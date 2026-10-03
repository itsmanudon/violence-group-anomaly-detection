"""Explicit C3D preprocessing; checkpoint provenance must match this policy."""

import cv2
import numpy as np
import torch


def c3d_transform(
    clip: np.ndarray, mean: tuple[float, float, float] = (0.0, 0.0, 0.0), channel_order: str = "bgr"
) -> torch.Tensor:
    """Resize to 128x171, center crop 112, subtract configured channel means.

    Returns [3,16,112,112] float32 in raw 0..255 scale before subtraction.
    Zero mean is an implementation default, NOT original C3D preprocessing.
    Use the chosen weights' training mean/order; no guessed pretrained mean.
    """
    if clip.shape[0] != 16 or clip.ndim != 4 or clip.shape[-1] != 3:
        raise ValueError("C3D requires [16,H,W,3] clips")
    if channel_order not in {"rgb", "bgr"} or len(mean) != 3 or not np.isfinite(mean).all():
        raise ValueError("Expected rgb/bgr channel order and three finite channel means")
    resized = np.stack([cv2.resize(frame, (171, 128)) for frame in clip])
    cropped = resized[:, 8:120, 29:141].astype(np.float32)
    if channel_order == "rgb":
        cropped = cropped[..., ::-1].copy()
    cropped -= np.asarray(mean, dtype=np.float32)
    return torch.from_numpy(cropped.transpose(3, 0, 1, 2).copy())
