"""Threshold-gated one-to-one detector matching and offline GT label transfer."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def _boxes(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"{name} must be a tensor of shape [N,4]")
    if value.is_complex() or value.dtype == torch.bool:
        raise ValueError(f"{name} must contain real coordinates")
    value = value.detach().to(dtype=torch.float64)
    if not torch.isfinite(value).all() or (value < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative absolute xyxy coordinates")
    if (value[:, 2:] <= value[:, :2]).any():
        raise ValueError(f"{name} must have positive width and height")
    return value


def _threshold(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("IoU threshold must be finite and in (0,1]")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("IoU threshold must be finite and in (0,1]") from error
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("IoU threshold must be finite and in (0,1]")
    return value


def pairwise_iou(gt_boxes: Tensor, det_boxes: Tensor) -> Tensor:
    """Return [G,D] IoUs for finite, positive-area absolute xyxy boxes.

    Float64 computation avoids overflow for ordinary image coordinates. Inputs
    must share a device; the output remains on that device, including empty axes.
    """
    gt, detections = _boxes(gt_boxes, "GT boxes"), _boxes(det_boxes, "Detection boxes")
    if gt.device != detections.device:
        raise ValueError("GT and detection boxes must be on the same device")
    # Scale extreme coordinates only; preserve exact integer pixel arithmetic
    # around inclusive thresholds for ordinary image dimensions.
    maximum = (
        torch.cat((gt.reshape(-1), detections.reshape(-1))).max()
        if (gt.numel() or detections.numel())
        else gt.new_tensor(1.0)
    )
    if maximum > 1e150 or maximum < 1e-150:
        gt, detections = gt / maximum, detections / maximum
    lower = torch.maximum(gt[:, None, :2], detections[None, :, :2])
    upper = torch.minimum(gt[:, None, 2:], detections[None, :, 2:])
    intersection = (upper - lower).clamp_min(0).prod(dim=-1)
    gt_area = (gt[:, 2:] - gt[:, :2]).prod(dim=-1)
    detection_area = (detections[:, 2:] - detections[:, :2]).prod(dim=-1)
    union = gt_area[:, None] + detection_area[None, :] - intersection
    result = intersection / union
    if not torch.isfinite(result).all():
        raise ValueError("Box coordinate magnitudes are too disparate for stable IoU")
    return result.clamp(0, 1)


@dataclass(frozen=True)
class MatchResult:
    """Indices reference original GT and detection ordering; each appears once."""

    gt_indices: list[int]
    detection_indices: list[int]
    ious: list[float]
    missed_gt: list[int]
    unmatched_detections: list[int]
    gt_count: int
    detection_count: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 0 for n in (self.gt_count, self.detection_count)):
            raise ValueError("GT and detection counts must be nonnegative integers")
        if not len(self.gt_indices) == len(self.detection_indices) == len(self.ious):
            raise ValueError("Matched GT indices, detection indices and IoUs must align")
        for matched, unmatched, count in (
            (self.gt_indices, self.missed_gt, self.gt_count),
            (self.detection_indices, self.unmatched_detections, self.detection_count),
        ):
            indices = [*matched, *unmatched]
            if any(type(i) is not int for i in indices) or sorted(indices) != list(range(count)):
                raise ValueError("Matched and unmatched indices must partition each input")
        if any(not math.isfinite(iou) or not 0 < iou <= 1 for iou in self.ious):
            raise ValueError("Matched IoUs must be finite and in (0,1]")

    def to_dict(self) -> dict:
        return asdict(self)


def match_boxes(gt_boxes: Tensor, det_boxes: Tensor, iou_threshold: float = 0.5) -> MatchResult:
    """Maximize eligible match count first, then total IoU, with inclusive gating.

    Each eligible edge earns min(G,D)+1 plus its IoU. That bonus exceeds the
    largest possible IoU-sum improvement from losing an eligible match. Ineligible
    edges earn zero and are removed after assignment. Fixed input order produces
    deterministic assignments, returned in ascending GT index order.
    """
    threshold = _threshold(iou_threshold)
    ious = pairwise_iou(gt_boxes, det_boxes).cpu()
    gt_count, detection_count = ious.shape
    eligible = ious >= threshold
    scores = torch.where(eligible, min(gt_count, detection_count) + 1 + ious, 0.0)
    gt_indices, detection_indices, matched_ious = [], [], []
    if gt_count and detection_count:
        rows, columns = linear_sum_assignment(scores.numpy(), maximize=True)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if eligible[row, column]:
                gt_indices.append(row)
                detection_indices.append(column)
                matched_ious.append(float(ious[row, column]))
    selected_gt, selected_detections = set(gt_indices), set(detection_indices)
    return MatchResult(
        gt_indices,
        detection_indices,
        matched_ious,
        [i for i in range(gt_count) if i not in selected_gt],
        [i for i in range(detection_count) if i not in selected_detections],
        gt_count,
        detection_count,
    )


def transfer_actor_labels(gt_labels: Tensor, result: MatchResult) -> Tensor:
    """Copy labels only to eligible matched detections; unmatched labels are -100."""
    if (
        not isinstance(gt_labels, Tensor)
        or gt_labels.dtype != torch.long
        or gt_labels.shape != (result.gt_count,)
    ):
        raise ValueError("GT labels must be a long tensor of shape [gt_count]")
    if ((gt_labels < 0) & (gt_labels != -100)).any():
        raise ValueError("GT labels must be nonnegative or the ignore index -100")
    labels = gt_labels.new_full((result.detection_count,), -100)
    if result.gt_indices:
        labels[result.detection_indices] = gt_labels[result.gt_indices]
    return labels


def match_manifest(
    manifest: Path,
    detections_path: Path,
    split: str | None = "test",
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Join GT and cached detections with strict scene/source/center identity checks.

    Other known GT splits may appear in the detection cache. Every selected GT
    scene must have a detection record, including explicit empty results; unknown
    scene keys are rejected. GT normalized boxes use the cache's image dimensions.
    This is an offline annotation tool, not a detector or production classifier.
    """
    from surveillance.datasets.collective import read_actor_manifest
    from surveillance.detection.records import read_detections

    threshold = _threshold(iou_threshold)
    if split not in {None, "train", "val", "test"}:
        raise ValueError("split must be train, val, test or None")
    records = read_actor_manifest(Path(manifest))
    gt_by_key = {(row.dataset, row.clip_id): row for row in records}
    detections = read_detections(Path(detections_path))
    detection_by_key = {}
    for detection in detections:
        key = detection.key
        if key in detection_by_key:
            raise ValueError(f"Duplicate detection scene: {key}")
        if key not in gt_by_key:
            raise ValueError(f"Unknown detection scene absent from GT manifest: {key}")
        row = gt_by_key[key]
        if (
            detection.video_id != row.video_id
            or detection.source_video_id != row.source_video_id
            or detection.frame_index != row.frame_indices[5]
        ):
            raise ValueError(f"Detection source/video/center frame mismatch: {key}")
        detection_by_key[key] = detection
    selected = [row for row in records if split is None or row.split == split]
    if not selected:
        raise ValueError(f"No GT scenes for split {split!r}")
    scenes = []
    for row in sorted(selected, key=lambda row: (row.dataset, row.clip_id)):
        key = (row.dataset, row.clip_id)
        if key not in detection_by_key:
            raise ValueError(f"Missing detection record for GT scene: {key}")
        detection = detection_by_key[key]
        result = detection.result
        height, width = result.image_size
        gt_boxes = torch.tensor(row.actor_boxes, dtype=torch.float64).reshape(-1, 4)
        gt_boxes *= torch.tensor([width, height, width, height], dtype=torch.float64)
        boxes = result.boxes.detach().cpu()
        matches = match_boxes(gt_boxes, boxes, threshold)
        labels = transfer_actor_labels(torch.tensor(row.actor_labels, dtype=torch.long), matches)
        scenes.append(
            {
                "dataset": row.dataset,
                "video_id": row.video_id,
                "source_video_id": row.source_video_id,
                "clip_id": row.clip_id,
                "split": row.split,
                "frame_index": detection.frame_index,
                "image_size": [height, width],
                "iou_threshold": threshold,
                "gt_boxes": gt_boxes.tolist(),
                "gt_labels": row.actor_labels,
                "detection_boxes": boxes.tolist(),
                "scores": result.scores.detach().cpu().tolist(),
                "class_ids": result.class_ids.detach().cpu().tolist(),
                "actor_labels": labels.tolist(),
                "ignore_index": -100,
                "matches": matches.to_dict(),
            }
        )
    return scenes
