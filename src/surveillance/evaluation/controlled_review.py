"""Fixed-policy diagnostic study; behavior abstention never becomes a negative alert."""

from collections import Counter, defaultdict

import numpy as np

from surveillance.evaluation.cascade import summarize_cascade
from surveillance.inference.cascade_review import ReviewPolicy, assess_review

MODES = ("legacy", "coverage_consistency", "coverage")


def eligible_validation(rows: list[dict], receipt: dict, anomaly_protocol: dict) -> list[dict]:
    """Require the original validation split in BOTH model source protocols."""
    if receipt.get("source_overlap") or receipt.get("duplicate_content_overlap"):
        raise ValueError("Historical source/content overlap is unresolved")
    if anomaly_protocol.get("source_overlap") or anomaly_protocol.get(
        "cross_split_content_overlap"
    ):
        raise ValueError("Anomaly source/content overlap is unresolved")
    if len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate manifest IDs")
    hashes = {}
    for row in rows:
        if not row["selected"]:
            continue
        if receipt["source_assignments"].get(row["source_video_id"]) != row["split"]:
            raise ValueError("Human source assignment changed")
        previous = hashes.setdefault(row["sha256"], row["split"])
        if previous != row["split"]:
            raise ValueError("Exact footage crosses human splits")
    validation = sorted(
        [r for r in rows if r["selected"] and r["split"] == "val"], key=lambda r: r["clip_id"]
    )
    if not validation:
        raise ValueError("No eligible historical validation clips")
    for row in validation:
        if anomaly_protocol["source_assignments"].get(row["source_video_id"]) != "val":
            raise ValueError("Validation source is training/test/unknown for the anomaly model")
    # Known raw-source exact-content aliases must also stay outside optimization/test.
    val_sources = {r["source_video_id"] for r in validation}
    for group in anomaly_protocol.get("exact_content_duplicate_groups", []):
        if set(group) & val_sources and any(
            anomaly_protocol["source_assignments"].get(s) != "val" for s in group
        ):
            raise ValueError("Validation source alias crosses anomaly splits")
    return validation


def representative(result: dict):
    available = [w for w in result["behavior"]["windows"] if w["status"] == "available"]
    if not available:
        return None, None
    chosen = next(
        (w for w in available if w["interval"] == result["final_alert"]["timestamp_interval"]),
        available[0],
    )
    probabilities = chosen["probabilities"]
    return int(np.argmax(probabilities)), float(max(probabilities))


def diagnose(
    rows: list[dict], *, bootstrap_repetitions=2000, seed=20261026
) -> tuple[dict, list[dict]]:
    """Use fixed existing modes only; no optimization or alternate alert mapping."""
    rows = sorted(rows, key=lambda r: r["clip_id"])
    baseline = summarize_cascade(rows)
    comparisons, categories = [], defaultdict(list)
    vectors = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    for row in sorted(rows, key=lambda r: r["clip_id"]):
        result, target = row["result"], row["group_label"]
        raw_alert = result["final_alert"]["state"] != "no_anomaly"
        actual = row["binary_label"]
        outcome = ("TP" if actual else "FP") if raw_alert else ("FN" if actual else "TN")
        legacy, confidence = representative(result)
        predictions = [legacy]
        item = {
            "clip_id": row["clip_id"],
            "source_video_id": row["source_video_id"],
            "binary_target_abnormal_not_violence": actual,
            "six_class_target": target,
            "anomaly_score": result["anomaly"]["overall_score"],
            "legacy_final_state": result["final_alert"]["state"],
            "legacy_alert": raw_alert,
            "legacy_binary_outcome": outcome,
            "legacy_behavior": legacy,
            "legacy_max_softmax": confidence,
            "routed_windows": len(result["behavior"]["windows"]),
        }
        for mode in MODES[1:]:
            assessed = assess_review(result, ReviewPolicy(mode))
            if (
                assessed["final_alert"] != result["final_alert"]
                or assessed["alerts"] != result["alerts"]
            ):
                raise ValueError("Assessment unexpectedly changed raw alert decisions")
            sidecar = assessed["review_assessment"]
            item.update(
                {
                    f"{mode}_alert": raw_alert,
                    f"{mode}_behavior": sidecar["suggested_behavior_class"],
                    f"{mode}_reason": sidecar["reason"],
                }
            )
            predictions.append(sidecar["suggested_behavior_class"])
        item["behavior_suggestion_withheld_a1"] = legacy is not None and predictions[1] is None
        item["legacy_behavior_correct"] = legacy == target if legacy is not None else None
        windows = result["behavior"]["windows"]
        item["no_actor_windows"] = sum(w["status"] == "no_actors" for w in windows)
        item["covered_class_disagreement"] = (
            len({w["prediction"] for w in windows if w["status"] == "available"}) > 1
        )
        item["anomaly_normal_disagreement"] = raw_alert and any(
            w.get("prediction") == 0 for w in windows
        )
        for index, prediction in enumerate(predictions):
            vectors[row["source_video_id"]][index] += [
                1,
                int(prediction is not None),
                int(prediction is not None and prediction == target),
            ]
        labels = [
            "false_positives_retained"
            if outcome == "FP"
            else "true_abnormal_positives_preserved"
            if outcome == "TP"
            else "false_negatives_retained"
            if outcome == "FN"
            else "true_negatives_preserved"
        ]
        if windows and item["no_actor_windows"] == len(windows):
            labels.append("routed_no_actors")
        if item["anomaly_normal_disagreement"]:
            labels.append("anomaly_vs_normal_behavior")
        if item["behavior_suggestion_withheld_a1"]:
            labels.append(
                "legacy_wrong_behavior_withheld"
                if legacy != target
                else "legacy_correct_behavior_withheld"
            )
        if legacy is not None and legacy != target and predictions[1] is not None:
            labels.append("wrong_behavior_retained")
        for label in labels:
            if len(categories[label]) < 5:
                categories[label].append(item.copy())
        comparisons.append(item)
    summaries = {}
    for index, mode in enumerate(MODES):
        values = [
            item["legacy_behavior"] if mode == "legacy" else item[f"{mode}_behavior"]
            for item in comparisons
        ]
        covered = sum(p is not None for p in values)
        correct = sum(
            p is not None and p == r["group_label"] for p, r in zip(values, rows, strict=True)
        )
        withheld_wrong = sum(
            i["legacy_behavior"] is not None and p is None and not i["legacy_behavior_correct"]
            for i, p in zip(comparisons, values, strict=True)
        )
        withheld_correct = sum(
            i["legacy_behavior"] is not None and p is None and i["legacy_behavior_correct"]
            for i, p in zip(comparisons, values, strict=True)
        )
        summaries[mode] = {
            "count": len(rows),
            "sources": len(vectors),
            "binary_changed_decisions": 0,
            "binary_changed_percent": 0.0,
            "correct_binary_changes": 0,
            "incorrect_binary_changes": 0,
            "binary_abstentions": 0,
            "binary_decision_coverage": 1.0,
            "false_alerts_prevented": 0,
            "true_abnormal_detections_lost": 0,
            "false_negatives_rescued": 0,
            "true_violence_detections_lost": None,
            "behavior_covered": covered,
            "behavior_abstained": len(rows) - covered,
            "behavior_abstention_rate": 1 - covered / len(rows),
            "behavior_coverage": covered / len(rows),
            "conditional_behavior_accuracy": correct / covered if covered else None,
            "correct_behavior_fraction_full_population": correct / len(rows),
            "wrong_legacy_behavior_withheld": withheld_wrong,
            "correct_legacy_behavior_withheld": withheld_correct,
            "assessment_reason_counts": dict(
                Counter(i.get(f"{mode}_reason", "legacy") for i in comparisons)
            ),
            "withheld_legacy_behavior_errors_by_true_class": dict(
                Counter(
                    i["six_class_target"]
                    for i, p in zip(comparisons, values, strict=True)
                    if i["legacy_behavior"] is not None
                    and p is None
                    and not i["legacy_behavior_correct"]
                )
            ),
            "binary_metrics_identical_to_legacy": baseline["clip_alert"],
        }
    matrix = baseline["clip_alert"]["confusion_matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    for summary in summaries.values():
        summary["binary_counts"] = dict(TN=tn, FP=fp, FN=fn, TP=tp)
        summary["specificity"] = tn / (tn + fp) if tn + fp else None
        summary["false_negative_rate"] = fn / (fn + tp) if fn + tp else None
    if bootstrap_repetitions < 1:
        raise ValueError("Positive source-bootstrap repetitions required")
    source_array = np.stack([vectors[s] for s in sorted(vectors)])
    rng = np.random.default_rng(seed)
    deltas = defaultdict(list)
    for _ in range(bootstrap_repetitions):
        total = source_array[rng.integers(len(vectors), size=len(vectors))].sum(axis=0)
        for index, mode in enumerate(MODES[1:], 1):
            n, c, k = total[index]
            bn, bc, bk = total[0]
            deltas[f"{mode}_coverage_delta"].append(c / n - bc / bn)
            deltas[f"{mode}_full_correct_fraction_delta"].append(k / n - bk / bn)
            if c and bc:
                deltas[f"{mode}_conditional_accuracy_delta"].append(k / c - bk / bc)
    intervals = {
        key: {"percentile_95": np.quantile(v, [0.025, 0.975]).tolist(), "valid_replicates": len(v)}
        for key, v in deltas.items()
    }
    return {
        "outcome": "C: diagnostic capability, no alert-decision improvement demonstrated",
        "scope": "historical_validation_diagnostics_not_independent_confirmatory_evidence",
        "binary_label_semantics": (
            "DCSASS abnormal/non-Normal weak clip labels, not violence ground truth"
        ),
        "modes": summaries,
        "examples_first_five_by_id": dict(categories),
        "zero_count_categories": {
            k: 0
            for k in (
                "false_positives_resolved",
                "true_abnormal_positives_lost",
                "false_negatives_rescued",
            )
        },
        "uncertainty": {
            "method": "paired original-source cluster percentile bootstrap",
            "seed": seed,
            "repetitions": bootstrap_repetitions,
            "source_clusters": len(vectors),
            "intervals": intervals,
            "limits": (
                "conditional on original-source grouping; shared incidents across recordings "
                "unknown; reused checkpoint-selection validation; not causal or confirmatory; "
                "no clip-independent p-values"
            ),
            "binary_delta": "exactly zero by unchanged-alert contract, no significance test",
        },
        "uncertainty_truth": (
            "NOT AVAILABLE: behavioral abstention is not a labeled uncertainty class"
        ),
    }, comparisons
