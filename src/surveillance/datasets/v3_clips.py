"""Indian pilot v3 window rules (runs/indian_pilot/V3_PROTOCOL.md).

Pure functions only: which 2 s windows of a labelled video become clips. No I/O.
"""

import math
import random
import re

import numpy as np

XD_VIOLENT = frozenset({"B1", "B4", "B5"})
XD_EXCLUDED = frozenset({"B2", "B6", "G"})


def window_frames(fps: float, seconds: float = 2.0) -> int:
    return round(seconds * fps)


def evenly(candidates: list, k: int) -> list:
    """k evenly spaced items, or all of them when there are at most k. Deterministic."""
    n = len(candidates)
    if n <= k:
        return list(candidates)
    return [candidates[math.floor((i + 0.5) * n / k)] for i in range(k)]


def grid_windows(labels: np.ndarray, width: int, margin: int) -> tuple[list[int], list[int]]:
    """Non-overlapping window starts from frame 0: (all-violent, matched-normal).

    Matched normal windows have no violent frame inside them or within `margin` frames of
    either edge.
    """
    labels = np.asarray(labels, dtype=bool)
    starts = range(0, len(labels) - width + 1, width)
    violent = [s for s in starts if labels[s : s + width].all()]
    normal = [s for s in starts if not labels[max(0, s - margin) : s + width + margin].any()]
    return violent, normal


def intervals_to_labels(num_frames: int, bounds: list[int]) -> np.ndarray:
    """XD-Violence `start end start end ...` frame intervals (inclusive) to per-frame labels."""
    if len(bounds) % 2:
        raise ValueError("Interval bounds must come in start/end pairs")
    labels = np.zeros(num_frames, dtype=bool)
    for start, end in zip(bounds[::2], bounds[1::2], strict=True):
        if not 0 <= start <= end < num_frames + 1:
            raise ValueError(f"Interval {start}-{end} outside {num_frames} frames")
        labels[start : end + 1] = True
    return labels


def xd_kind(name: str) -> str:
    """'violent', 'normal' or 'excluded' from an XD-Violence file name's label part."""
    labels = set(name.rsplit("_label_", 1)[1].split("-")) - {"0"}
    if labels == {"A"}:
        return "normal"
    if labels & XD_EXCLUDED or not labels <= XD_VIOLENT:
        return "excluded"
    return "violent"


def xd_source(name: str) -> str:
    """Movie title or YouTube ID: everything before `__#`."""
    return name.split("__#", 1)[0]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def split_sources(sources: dict[str, bool], val_fraction: float, seed: int) -> dict[str, str]:
    """Assign sources to train/val, stratified by whether a source contains violence."""
    assignment = {}
    rng = random.Random(seed)
    for violent in (False, True):
        members = sorted(s for s, v in sources.items() if v == violent)
        rng.shuffle(members)
        cut = round(val_fraction * len(members))
        assignment.update({s: "val" for s in members[:cut]})
        assignment.update({s: "train" for s in members[cut:]})
    return assignment
