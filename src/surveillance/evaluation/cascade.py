"""Coarse clip-label cascade accounting, separate from frame localization metrics."""

import numpy as np

from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.evaluation.anomaly_metrics import binary_metrics
from surveillance.evaluation.group_activity_metrics import classification_metrics


def held_out_cascade_rows(rows: list[dict], anomaly_protocol: dict) -> list[dict]:
    """Only declared human test sources protected from both model optimizations."""
    assignments = {}
    for row in rows:
        source, split = row["source_video_id"], row["split"]
        if source in assignments and assignments[source] != split:
            raise ValueError("Human manifest source crosses splits")
        assignments[source] = split
    held_out = sorted(
        [r for r in rows if r["split"] == "test" and r["selected"]], key=lambda r: r["clip_id"]
    )
    if not held_out or len({r["clip_id"] for r in held_out}) != len(held_out):
        raise ValueError("Held-out cascade clip population is empty or duplicated")
    excluded = {
        r["source_video_id"] for r in anomaly_protocol["excluded_author_train_held_out_dcsass"]
    }
    for row in held_out:
        source = row["source_video_id"]
        membership = anomaly_protocol["source_assignments"].get(source)
        if membership in {"train", "val"} or not (membership == "test" or source in excluded):
            raise ValueError(f"Cascade source is not held out of anomaly training: {source}")
    return held_out


def summarize_cascade(records: list[dict]) -> dict:
    """Use saved outputs only; never invent Normal labels for behavior abstentions."""
    if not records:
        raise ValueError("Cascade evaluation population is empty")
    identities = [row["clip_id"] for row in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate clips in cascade evaluation")
    first = records[0]["result"]
    labels, scores, group_targets, group_predictions = [], [], [], []
    routed = covered = no_actor_clips = window_count = no_actor_windows = 0
    timings = {}
    for row in records:
        result, label = row["result"], row["group_label"]
        if (
            result.get("schema_version") != 1
            or result.get("provenance") != first["provenance"]
            or result["policy"] != first["policy"]
        ):
            raise ValueError("Mixed model identities or routing policies in evaluation")
        if (
            type(label) is not int
            or label not in range(len(CLASSES))
            or row["binary_label"] != int(label != 0)
        ):
            raise ValueError("Inconsistent human-centric clip labels")
        score = result["anomaly"]["overall_score"]
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid anomaly score")
        alert = result["final_alert"]["state"] != "no_anomaly"
        if alert != (score >= result["policy"]["anomaly_threshold"]):
            raise ValueError("Final alert contradicts frozen anomaly routing")
        labels.append(row["binary_label"])
        scores.append(score)
        windows = result["behavior"]["windows"]
        if windows:
            routed += 1
        window_count += len(windows)
        available = []
        for window in windows:
            if window["status"] == "no_actors":
                if window["actor_count"] != 0 or window["probabilities"] is not None:
                    raise ValueError("No-actor outputs must abstain")
                no_actor_windows += 1
            elif window["status"] == "available" and window["actor_count"] > 0:
                probability = np.asarray(window["probabilities"], dtype=float)
                if (
                    probability.shape != (6,)
                    or not np.isfinite(probability).all()
                    or (probability < 0).any()
                    or (probability > 1).any()
                    or not np.isclose(probability.sum(), 1, atol=1e-6)
                ):
                    raise ValueError("Invalid six-class behavior probabilities")
                available.append(window)
            else:
                raise ValueError("Unsupported behavior coverage status")
        if available:
            covered += 1
            chosen = next(
                (
                    w
                    for w in available
                    if w["interval"] == result["final_alert"]["timestamp_interval"]
                ),
                available[0],
            )
            group_targets.append(label)
            group_predictions.append(int(np.argmax(chosen["probabilities"])))
        elif windows:
            no_actor_clips += 1
        for key, value in result["timings"].items():
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid inference timing")
            timings.setdefault(key, []).append(value)
    count = len(records)
    clip_alert = binary_metrics(labels, scores, first["policy"]["anomaly_threshold"])
    confusion = clip_alert["confusion_matrix"]
    clip_alert["accuracy"] = (confusion[0][0] + confusion[1][1]) / count
    return {
        "population_scope": "Held-out DCSASS weak clip labels; not interval/actor ground truth",
        "clip_alert": clip_alert,
        "conditional_behavior": classification_metrics(group_targets, group_predictions, 6)
        if group_targets
        else None,
        "coverage": {
            "total_clips": count,
            "source_ids": len({r["source_video_id"] for r in records}),
            "routed_clips": routed,
            "bypassed_clips": count - routed,
            "covered_clips": covered,
            "no_actor_routed_clips": no_actor_clips,
            "behavior_abstained_clips": count - covered,
            "coverage_overall": covered / count,
            "coverage_among_routed": covered / routed if routed else None,
            "analyzed_windows": window_count,
            "no_actor_windows": no_actor_windows,
        },
        "latency_seconds": {
            key: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for key, values in timings.items()
        },
        "provenance": first["provenance"],
        "policy": first["policy"],
        "classes": list(CLASSES),
        "temporal_localization_measured": False,
        "explanation": "Alerts preserve Sultani anomalies, including no actors or model "
        "disagreement. Behavior metrics use covered routed clips only, choosing "
        "the final-alert covered window or first covered window as in the UI. "
        "Clip labels do not establish correctness of individual analyzed intervals. "
        "Bypassed clips have unexamined actors, not zero detected actors.",
    }
