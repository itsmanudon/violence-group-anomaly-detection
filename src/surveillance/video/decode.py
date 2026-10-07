"""OpenCV decoding with bounded-memory clip iteration."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Container metadata; frame count can be approximate for some codecs."""

    fps: float
    num_frames: int
    duration_sec: float


def probe_video(path: Path) -> VideoMetadata:
    """Read metadata without decoding the complete video."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}. Install the dataset locally first.")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}; check codec support")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError(f"Invalid video metadata: {path}")
        return VideoMetadata(fps, count, count / fps)
    finally:
        capture.release()


def iter_clips(path: Path, clip_length: int = 16) -> Iterator[np.ndarray]:
    """Yield nonoverlapping BGR uint8 clips; pad final clip with its last frame."""
    if clip_length < 1:
        raise ValueError("clip_length must be positive")
    metadata = probe_video(path)
    capture = cv2.VideoCapture(str(path))
    frames = []
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            frames.append(frame)
            if len(frames) == clip_length:
                yield np.stack(frames)
                frames = []
        if not decoded:
            raise ValueError(f"Video contains no decodable frames: {path}")
        if decoded != metadata.num_frames:
            raise ValueError(
                f"Decoded frame count {decoded} differs from reported {metadata.num_frames}: "
                f"{path}. Exact timeline mapping requires a complete, consistent video."
            )
        if frames:
            frames.extend([frames[-1]] * (clip_length - len(frames)))
            yield np.stack(frames)
    finally:
        capture.release()
