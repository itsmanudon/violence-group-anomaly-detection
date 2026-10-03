"""Binary and temporal evaluation; bag-level labels are not frame annotations."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def project_scores(scores: Sequence[float], num_frames: int, mode: str = "repeat") -> np.ndarray:
    """Project uniform segment scores to frames using floor boundaries or interpolation.

    Repetition matches segment_indices for N>=S. For N<S select the nearest
    segment at each frame center. Interpolation uses segment/frame centers.
    This is a uniform-duration approximation for precomputed bags; extraction
    includes a padded last 16-frame clip, so exact clip/frame boundaries may differ.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or num_frames < 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite 1D scores and positive frame count")
    if mode == "interpolate":
        return np.interp(
            (np.arange(num_frames) + 0.5) / num_frames,
            (np.arange(len(scores)) + 0.5) / len(scores),
            scores,
        )
    if mode != "repeat":
        raise ValueError("mode must be repeat or interpolate")
    if num_frames < len(scores):
        return scores[
            np.minimum(
                ((np.arange(num_frames) + 0.5) * len(scores) / num_frames).astype(int),
                len(scores) - 1,
            )
        ]
    edges = np.arange(len(scores) + 1) * num_frames // len(scores)
    return np.repeat(scores, np.diff(edges))


def frame_labels(num_frames: int, fps: float, intervals: list[list[float]]) -> np.ndarray:
    """Create binary frame labels for [start,end) intervals in seconds."""
    if num_frames < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Positive frame count and FPS required")
    labels = np.zeros(num_frames, dtype=np.int64)
    times = np.arange(num_frames) / fps
    for start, end in intervals:
        if not 0 <= start < end:
            raise ValueError("Invalid temporal annotation")
        labels[(times >= start) & (times < end)] = 1
    return labels


def binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict:
    """Compute ROC and threshold metrics; one-class ROC is undefined (None)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Labels and scores must be same-length nonempty vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    result = dict(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        false_positive_rate=float(fp / (fp + tn)) if fp + tn else None,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        roc_auc=None,
        roc_curve=None,
        threshold=threshold,
        count=len(labels),
    )
    if len(np.unique(labels)) == 2:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        result.update(
            roc_auc=float(roc_auc_score(labels, scores)),
            roc_curve=dict(
                fpr=fpr.tolist(),
                tpr=tpr.tolist(),
                thresholds=[float(t) if np.isfinite(t) else None for t in thresholds],
            ),
        )
    return result
