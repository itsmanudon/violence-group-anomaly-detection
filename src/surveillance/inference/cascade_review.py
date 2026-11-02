"""Optional review-evidence assessment; original anomaly alerts are never suppressed."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReviewPolicy:
    mode: str = "legacy"

    def __post_init__(self):
        if self.mode not in {"legacy", "coverage", "coverage_consistency"}:
            raise ValueError("Unknown review assessment mode")


def assess_review(result: dict, policy: ReviewPolicy = ReviewPolicy()) -> dict:
    """Assess offline windows, not actor truth, binary violence or live latency.

    Legacy mode intentionally returns the existing object without any schema change.
    Other modes add a sidecar and retain the complete original alerts/results.
    """
    if policy.mode == "legacy":
        return result
    if result.get("schema_version") != 1:
        raise ValueError("Review assessment requires cascade schema version 1")
    windows = result["behavior"]["windows"]
    raw_alert = result["final_alert"]["state"] != "no_anomaly"
    if raw_alert != bool(windows):
        raise ValueError("Routing and anomaly alert disagree")
    available = []
    for window in windows:
        start, end = window["interval"]
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError("Invalid review window interval")
        count = window["actor_count"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid actor count")
        if window["status"] == "no_actors":
            if count or window["probabilities"] is not None or window.get("prediction") is not None:
                raise ValueError("No-actor windows must abstain")
        elif window["status"] == "available" and count:
            p = np.asarray(window["probabilities"], dtype=float)
            if (
                p.shape != (6,)
                or not np.isfinite(p).all()
                or (p < 0).any()
                or (p > 1).any()
                or not np.isclose(p.sum(), 1, atol=1e-6)
            ):
                raise ValueError("Expected six finite class probabilities summing to one")
            prediction = int(p.argmax())
            if window.get("prediction", prediction) != prediction:
                raise ValueError("Class prediction differs from probabilities")
            available.append((window, prediction))
        else:
            raise ValueError("Unsupported actor coverage status")
    reason = "not_routed_actors_unexamined"
    chosen = None
    if windows:
        if not available:
            reason = "no_actors"
        elif len(available) != len(windows):
            reason = "partial_actor_coverage"
        elif policy.mode == "coverage_consistency" and len({p for _, p in available}) > 1:
            reason = "cross_window_class_disagreement"
        else:
            chosen = next(
                (
                    p
                    for w, p in available
                    if w["interval"] == result["final_alert"]["timestamp_interval"]
                ),
                available[0][1],
            )
            reason = "covered_suggestion_requires_human_review"
    return {
        **result,
        "review_assessment": {
            "schema_version": 1,
            "mode": policy.mode,
            "anomaly_alert_preserved": raw_alert,
            "routed_windows": len(windows),
            "covered_windows": len(available),
            "covered_class_votes": [p for _, p in available],
            "suggested_behavior_class": chosen,
            "behavior_abstained": chosen is None,
            "reason": reason,
            "confidence_calibrated": False,
            "binary_violence_decision": None,
            "human_review_required": raw_alert,
        },
    }


class ReviewAwarePipeline:
    """Explicit opt-in wrapper; the original loader/UI and its defaults are untouched."""

    def __init__(self, pipeline, policy: ReviewPolicy = ReviewPolicy()):
        self.pipeline, self.policy = pipeline, policy

    def predict_video(self, path):
        return assess_review(self.pipeline.predict_video(path), self.policy)
