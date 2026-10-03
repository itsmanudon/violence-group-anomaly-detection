"""Explicit xyxy transforms; size tuples are always (height, width)."""

import math

import torch
from torchvision.ops import roi_align


def _validate_boxes(boxes: torch.Tensor) -> None:
    if boxes.ndim < 2 or boxes.shape[-1] != 4 or not boxes.is_floating_point():
        raise ValueError("boxes must be floating-point [...,4] xyxy")
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes must contain finite coordinates")
    if ((boxes[..., 2:] - boxes[..., :2]) <= 0).any():
        raise ValueError("boxes must have positive width and height")


def _validate_size(width: float, height: float) -> None:
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("image width and height must be finite and positive")


def clip_boxes(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Clip absolute xyxy edges to [0,width] / [0,height], rejecting empty boxes.

    Validate before and after clipping: inverted boxes and boxes completely
    outside the image are errors, while partially visible boxes are retained.
    The input is never mutated; empty [...,4] tensors are supported.
    """
    _validate_size(width, height)
    _validate_boxes(boxes)
    bounds = boxes.new_tensor([width, height, width, height])
    clipped = torch.minimum(boxes.clamp_min(0), bounds)
    _validate_boxes(clipped)
    return clipped


def normalized_boxes_to_pixels(
    boxes: torch.Tensor, width: float, height: float, *, clip_outside: bool = False
) -> torch.Tensor:
    """Convert normalized xyxy edges to absolute coordinates, with explicit clipping."""
    _validate_size(width, height)
    _validate_boxes(boxes)
    pixels = boxes * boxes.new_tensor([width, height, width, height])
    if clip_outside:
        return clip_boxes(pixels, width, height)
    if ((boxes < 0) | (boxes > 1)).any():
        raise ValueError("valid actor boxes are outside [0,1]; opt into clip_outside=True")
    return pixels


def scale_boxes(
    boxes: torch.Tensor, source_size: tuple[int, int], target_size: tuple[int, int]
) -> torch.Tensor:
    """Scale absolute boxes independently in x and y between image/feature grids."""
    _validate_boxes(boxes)
    source_h, source_w = source_size
    target_h, target_w = target_size
    _validate_size(source_w, source_h)
    _validate_size(target_w, target_h)
    return boxes * boxes.new_tensor(
        [
            target_w / source_w,
            target_h / source_h,
            target_w / source_w,
            target_h / source_h,
        ]
    )


def validate_actor_layout(
    boxes: torch.Tensor, valid_mask: torch.Tensor, batch_size: int, device: torch.device
) -> None:
    if boxes.ndim != 3 or boxes.shape[0] != batch_size or boxes.shape[-1] != 4:
        raise ValueError("actor boxes must be [B,N,4] normalized xyxy")
    if valid_mask.shape != boxes.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean [B,N]")
    if boxes.device != device or valid_mask.device != device:
        raise ValueError("frames/features, boxes, and valid_mask must share a device")
    if not boxes.is_floating_point():
        raise ValueError("actor boxes must be floating-point")


def roi_align_actors(
    features: torch.Tensor,
    boxes: torch.Tensor,
    valid_mask: torch.Tensor,
    output_size: tuple[int, int],
    *,
    clip_outside: bool = False,
) -> torch.Tensor:
    """Return [sum(valid),C,h,w] in row-major actor order, ignoring padded boxes.

    Boxes are normalized to the full source image; mapping directly to feature
    width/height is equivalent to image->feature anisotropic scaling. RoIAlign
    uses aligned=True (half-pixel convention), sampling_ratio=2 and scale=1.
    """
    if features.ndim != 4 or not features.is_floating_point():
        raise ValueError("features must be floating-point [B,C,H,W]")
    validate_actor_layout(boxes, valid_mask, features.shape[0], features.device)
    height, width = features.shape[-2:]
    selected = boxes[valid_mask].to(dtype=features.dtype)
    pixels = normalized_boxes_to_pixels(selected, width, height, clip_outside=clip_outside)
    batch_indices = valid_mask.nonzero(as_tuple=False)[:, 0].to(dtype=features.dtype)
    rois = torch.cat([batch_indices[:, None], pixels], dim=1)
    return roi_align(
        features, rois, output_size=output_size, spatial_scale=1.0, sampling_ratio=2, aligned=True
    )
