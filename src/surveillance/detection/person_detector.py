"""Absolute-pixel detections, deterministic filtering, and backend protocol."""

from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from surveillance.detection.config import DetectionConfig


def _image_size(size):
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("image_size must be (height,width)")
    if any(type(value) is not int or value <= 0 for value in size):
        raise ValueError("image_size dimensions must be positive integers")
    return tuple(size)


def _tensors(boxes, scores, class_ids):
    boxes, scores, class_ids = (
        torch.as_tensor(value).detach().cpu() for value in (boxes, scores, class_ids)
    )
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N,4]")
    if scores.shape != (len(boxes),) or class_ids.shape != (len(boxes),):
        raise ValueError("scores and class_ids must have shape [N] matching boxes")
    if (
        scores.is_complex()
        or not torch.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("scores must be finite probabilities in [0,1]")
    if (
        class_ids.dtype == torch.bool
        or class_ids.is_complex()
        or not torch.isfinite(class_ids).all()
        or (class_ids < 0).any()
        or (class_ids != class_ids.long()).any()
    ):
        raise ValueError("class_ids must be nonnegative integers")
    if boxes.is_complex():
        raise ValueError("boxes must be real coordinates")
    return boxes.float(), scores.float(), class_ids.long()


def canonical_order(boxes: Tensor, scores: Tensor, class_ids: Tensor) -> list[int]:
    """Left-to-right centers, then vertical centers, coordinates, score and class."""
    return sorted(
        range(len(boxes)),
        key=lambda i: (
            float(boxes[i, 0] + boxes[i, 2]),
            float(boxes[i, 1] + boxes[i, 3]),
            *boxes[i].tolist(),
            -float(scores[i]),
            int(class_ids[i]),
        ),
    )


@dataclass(frozen=True)
class DetectionResult:
    """Validated CPU float32 absolute xyxy [N,4], scores [N], int64 classes [N]."""

    boxes: Tensor
    scores: Tensor
    class_ids: Tensor
    image_size: tuple[int, int]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        size = _image_size(self.image_size)
        boxes, scores, classes = _tensors(self.boxes, self.scores, self.class_ids)
        height, width = size
        if (
            not torch.isfinite(boxes).all()
            or (boxes[:, 2:] <= boxes[:, :2]).any()
            or (boxes < 0).any()
            or (boxes[:, [0, 2]] > width).any()
            or (boxes[:, [1, 3]] > height).any()
        ):
            raise ValueError("Detection boxes must be finite positive absolute xyxy inside image")
        if not isinstance(self.metadata, dict):
            raise ValueError("Detection metadata must be a dict")
        for key, value in (
            ("boxes", boxes),
            ("scores", scores),
            ("class_ids", classes),
            ("image_size", size),
        ):
            object.__setattr__(self, key, value)


@runtime_checkable
class PersonDetector(Protocol):
    def detect(self, image: Tensor) -> DetectionResult:
        """Detect persons in finite RGB float [3,H,W] image with values in [0,1]."""
        ...


def filter_detections(
    boxes: Tensor,
    scores: Tensor,
    class_ids: Tensor,
    image_size: tuple[int, int],
    config: DetectionConfig | None = None,
) -> DetectionResult:
    """Filter person boxes, clamp, apply stable CPU NMS, cap, then sort spatially.

    Invalid input shapes and scores fail. Invalid raw box geometry is counted and
    dropped. Per-stage drop counts are mutually exclusive and sum to before-after.
    """
    config = config or DetectionConfig()
    height, width = _image_size(image_size)
    boxes, scores, classes = _tensors(boxes, scores, class_ids)
    before = len(boxes)
    counts = {}

    def retain(mask, reason):
        nonlocal boxes, scores, classes
        counts[reason] = int((~mask).sum())
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

    retain(classes == config.person_class_id, "non_person")
    retain(scores >= config.confidence_threshold, "low_confidence")
    retain(torch.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1), "invalid_geometry")
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
    sizes = boxes[:, 2:] - boxes[:, :2]
    retain((sizes > 0).all(1), "outside_image")
    sizes = boxes[:, 2:] - boxes[:, :2]
    retain(
        (sizes[:, 0] >= config.min_box_width)
        & (sizes[:, 1] >= config.min_box_height)
        & (sizes.prod(1) >= config.min_box_area),
        "too_small",
    )
    # CPU float64 IoU and a geometry tie-break avoid device-dependent torchvision NMS ties.
    order = sorted(range(len(boxes)), key=lambda i: (-float(scores[i]), *boxes[i].tolist()))
    kept = []
    while order:
        index = order.pop(0)
        kept.append(index)
        remaining = []
        box = boxes[index].double()
        for other in order:
            candidate = boxes[other].double()
            intersection = (
                (torch.minimum(box[2:], candidate[2:]) - torch.maximum(box[:2], candidate[:2]))
                .clamp(min=0)
                .prod()
            )
            union = (box[2:] - box[:2]).prod() + (candidate[2:] - candidate[:2]).prod()
            if float(intersection / (union - intersection)) <= config.nms_iou_threshold:
                remaining.append(other)
        order = remaining
    counts["nms"] = len(boxes) - len(kept)
    counts["max_actors"] = max(0, len(kept) - config.max_actors)
    kept = kept[: config.max_actors]
    boxes, scores, classes = boxes[kept], scores[kept], classes[kept]
    order = canonical_order(boxes, scores, classes)
    return DetectionResult(
        boxes[order],
        scores[order],
        classes[order],
        (height, width),
        {
            "before_count": before,
            "after_count": len(order),
            "num_detections_before_filter": before,
            "num_detections_after_filter": len(order),
            "person_class_id": config.person_class_id,
            "filter_config": asdict(config),
            "filter_counts": counts,
            "truncated": counts["max_actors"] > 0,
        },
    )
