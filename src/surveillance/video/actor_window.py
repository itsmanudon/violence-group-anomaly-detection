"""Deterministic clip-local ten-frame RGB windows for the frozen I3D path."""

from pathlib import Path

import cv2
import numpy as np
import torch


def centered_frame_indices(
    num_frames: int, center: int | None = None, start: int = 0, end: int | None = None
) -> list[int]:
    """Zero-based [-5,+4] offsets, replicated only inside [start,end)."""
    end = num_frames if end is None else end
    center = (start + end) // 2 if center is None else center
    if not 0 <= start <= center < end <= num_frames:
        raise ValueError("Actor window requires nonempty valid frame bounds and center")
    return [max(start, min(end - 1, center + offset)) for offset in range(-5, 5)]


def read_actor_window(
    path: Path,
    num_frames: int,
    center: int | None = None,
    start: int = 0,
    end: int | None = None,
) -> dict:
    """Decode native reference RGB and OpenCV-resized 480x720 RGB model frames.

    Matches Collective's existing OpenCV resize/color/scale preprocessing.
    No source-wide frame index is inferred from a derived clip number.
    """
    indices = centered_frame_indices(num_frames, center, start, end)
    cap = cv2.VideoCapture(str(path))
    decoded = {}
    wanted = set(indices)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open actor video: {path}")
        for index in range(max(indices) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Actor window decode failed at frame {index}: {path}")
            if index in wanted:
                decoded[index] = frame
    finally:
        cap.release()
    reference = cv2.cvtColor(decoded[indices[5]], cv2.COLOR_BGR2RGB)
    frames = np.stack(
        [cv2.cvtColor(cv2.resize(decoded[i], (720, 480)), cv2.COLOR_BGR2RGB) for i in indices]
    )
    return {
        "frame_indices": indices,
        "reference_frame": indices[5],
        "image_size": list(reference.shape[:2]),
        "reference_rgb": torch.from_numpy(reference.copy()).permute(2, 0, 1).float() / 255,
        "frames": torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255,
    }
