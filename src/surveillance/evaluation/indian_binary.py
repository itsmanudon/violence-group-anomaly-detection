"""Binary clip metrics and causal temporal rules with explicit evaluation scope."""

import math

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from surveillance.evaluation.anomaly_metrics import binary_metrics


def evaluate_binary(labels, scores, threshold: float = 0.5) -> dict:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Binary scores must be probabilities in [0,1]")
    report = binary_metrics(labels, scores, threshold)
    tn, fp = report["confusion_matrix"][0]
    fn, tp = report["confusion_matrix"][1]
    report["accuracy"] = (tn + tp) / report["count"]
    report["false_negative_rate"] = fn / (fn + tp) if fn + tp else None
    two_classes = len(set(labels)) == 2
    precision, recall, _ = precision_recall_curve(labels, scores) if two_classes else ([], [], [])
    report.update(
        pr_auc=float(auc(recall, precision)) if two_classes else None,
        average_precision=float(average_precision_score(labels, scores)) if two_classes else None,
        pr_auc_definition="trapezoidal precision-recall curve; AP reported separately",
        population="binary_labeled_clips",
        false_alarms_per_camera_hour=None,
        detection_latency_seconds=None,
    )
    return report


def calibrate_threshold(
    labels, scores, *, split: str, min_recall: float, candidates: list[float]
) -> dict:
    """Max validation F1 subject to a prospectively declared recall floor."""
    if split != "val":
        raise ValueError("Threshold selection requires validation-only predictions")
    if not math.isfinite(min_recall) or not 0 < min_recall <= 1:
        raise ValueError("Declare a recall floor in (0,1]")
    if set(labels) != {0, 1} or not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Both validation classes and unique threshold candidates are required")
    trials = [evaluate_binary(labels, scores, value) for value in sorted(candidates)]
    feasible = [trial for trial in trials if trial["recall"] >= min_recall]
    if not feasible:
        raise ValueError("No candidate satisfies the validation recall floor")
    best = max(feasible, key=lambda r: (r["f1"], r["recall"], r["threshold"]))
    return {
        "split": "val",
        "min_recall": min_recall,
        "selected_threshold": best["threshold"],
        "objective": "maximum_f1_subject_to_recall_floor",
        "tie_break": ["higher_recall", "higher_threshold"],
        "trials": trials,
    }


def evaluate_abstention(labels, scores, *, min_confidence=0.5, threshold=0.5) -> dict:
    """Keep detector/low-confidence abstentions in the full population denominator.

    None means unavailable (e.g. no detected person). Confidence is the maximum
    binary probability and is not a calibrated correctness probability.
    """
    if (
        len(labels) != len(scores)
        or not labels
        or not set(labels) <= {0, 1}
        or not 0.5 <= min_confidence <= 1
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid binary abstention inputs")
    matrix = [[0, 0, 0], [0, 0, 0]]
    covered_labels, covered_scores = [], []
    for label, score in zip(labels, scores, strict=True):
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Available scores must be finite probabilities")
        if score is None or max(score, 1 - score) < min_confidence:
            matrix[label][2] += 1
        else:
            matrix[label][int(score >= threshold)] += 1
            covered_labels.append(label)
            covered_scores.append(score)
    positives, negatives = sum(matrix[1]), sum(matrix[0])
    tp, fp = matrix[1][1], matrix[0][1]
    return {
        "count": len(labels),
        "covered_count": len(covered_labels),
        "coverage": len(covered_labels) / len(labels),
        "confusion_columns": ["non_violence", "violence", "abstain"],
        "confusion_matrix": matrix,
        "all_positive_recall": tp / positives if positives else None,
        "all_negative_false_positive_rate": fp / negatives if negatives else None,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "conditional_metrics": evaluate_binary(covered_labels, covered_scores, threshold)
        if covered_labels
        else None,
    }


def temporal_alerts(
    intervals,
    scores,
    *,
    on: float = 0.5,
    off: float = 0.5,
    min_duration: float = 0.0,
    alpha: float = 1.0,
) -> list[dict]:
    """Causal EMA and hysteresis on contiguous, nonoverlapping scored intervals.

    min_duration measures contiguous active time. Decision time is the end of the
    first interval meeting persistence, not a retroactively backdated alert.
    Gaps reset smoothing/persistence. Call separately per camera/video. This is
    an offline ablation utility; duration and latency require genuine time units.
    """
    if (
        not all(math.isfinite(v) for v in (on, off, min_duration, alpha))
        or not 0 <= off <= on <= 1
        or min_duration < 0
        or not 0 < alpha <= 1
    ):
        raise ValueError("Invalid alert thresholds/duration/smoothing")
    if len(intervals) != len(scores):
        raise ValueError("Intervals and scores must align")
    result, start, decision, smooth, previous = [], None, None, None, None

    def close(end):
        if decision is not None:
            result.append({"start": start, "end": end, "decision_time": decision})

    for (left, right), score in zip(intervals, scores, strict=True):
        if (
            not all(math.isfinite(v) for v in (left, right, score))
            or not 0 <= left < right
            or not 0 <= score <= 1
            or (previous is not None and left < previous)
        ):
            raise ValueError("Expected ordered nonoverlapping finite intervals and probabilities")
        if previous is not None and left > previous:
            close(previous)
            start, decision, smooth = None, None, None
        smooth = score if smooth is None else alpha * score + (1 - alpha) * smooth
        if start is None and smooth >= on:
            start = left
        elif start is not None and smooth < off:
            close(left)
            start, decision = None, None
        if start is not None and decision is None and right - start >= min_duration:
            decision = right
        previous = right
    if previous is not None:
        close(previous)
    return result
