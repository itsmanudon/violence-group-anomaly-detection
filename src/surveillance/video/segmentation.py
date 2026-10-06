"""Deterministic temporal partitioning and feature aggregation."""

import numpy as np


def c3d_segment_frame_ranges(
    num_frames: int, num_segments: int = 32, unit_frames: int = 16
) -> list[tuple[int, int]]:
    """Map exact FC6 unit partitions to zero-based half-open original frames.

    The final padded unit ends at the last real frame. Short videos repeat
    their unit ranges exactly as feature aggregation repeats temporal units.
    """
    if num_frames < 1 or unit_frames < 1:
        raise ValueError("Frame count and C3D unit size must be positive")
    units = (num_frames + unit_frames - 1) // unit_frames
    return [
        (int(group[0]) * unit_frames, min((int(group[-1]) + 1) * unit_frames, num_frames))
        for group in segment_indices(units, num_segments)
    ]


def segment_indices(length: int, num_segments: int = 32) -> list[np.ndarray]:
    """Partition every unit once if N >= S; repeat ordered units if N < S.

    It is impossible to create S disjoint nonempty segments from N < S units.
    In that case each output picks floor(i*N/S), explicitly duplicating units.
    For longer inputs boundaries are floor(i*N/S); no units are discarded.
    """
    if length < 1 or num_segments < 1:
        raise ValueError("length and num_segments must be positive")
    if length < num_segments:
        return [np.array([i * length // num_segments]) for i in range(num_segments)]
    edges = np.arange(num_segments + 1, dtype=np.int64) * length // num_segments
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])]


def aggregate_segments(features: np.ndarray, num_segments: int = 32) -> np.ndarray:
    """Mean each temporal partition then L2-normalize; zero vectors stay zero."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] == 0 or not np.isfinite(features).all():
        raise ValueError("features must be a finite nonempty [units, feature_dim] matrix")
    output = np.stack(
        [features[g].mean(axis=0) for g in segment_indices(len(features), num_segments)]
    )
    return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
