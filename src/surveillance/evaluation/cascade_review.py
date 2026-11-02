"""Matched validation-only comparison of behavior assessment policies."""

from surveillance.evaluation.cascade import summarize_cascade
from surveillance.evaluation.group_activity_metrics import classification_metrics
from surveillance.evaluation.indian_binary import evaluate_binary
from surveillance.inference.cascade_review import ReviewPolicy, assess_review


def compare_review_modes(rows: list[dict], manifest: list[dict], provenance: dict) -> dict:
    assignments = {}
    manifest_ids = [r["clip_id"] for r in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("Duplicate clip IDs in comparison manifest")
    for row in manifest:
        source, split = row["source_video_id"], row["split"]
        if source in assignments and assignments[source] != split:
            raise ValueError("Source leakage in comparison manifest")
        assignments[source] = split
    expected = [r for r in manifest if r["split"] == "val" and r.get("selected", True)]
    expected_by_id = {r["clip_id"]: r for r in expected}
    if not expected or len(expected_by_id) != len(expected):
        raise ValueError("Empty or duplicate validation manifest")
    if len(rows) != len(expected) or {r["clip_id"] for r in rows} != set(expected_by_id):
        raise ValueError("Comparison requires exact validation membership")
    for row in rows:
        expected_row = expected_by_id[row["clip_id"]]
        if row.get("split") != "val":
            raise ValueError("Comparison refuses protected test or undeclared split")
        for key in ("source_video_id", "group_label", "binary_label", "video_sha256"):
            if key not in expected_row or row[key] != expected_row[key]:
                raise ValueError(f"Comparison manifest mismatch: {key}")
            if key in {"source_video_id", "video_sha256"} and not expected_row[key]:
                raise ValueError(f"Missing comparison identity: {key}")
        if row["result"]["provenance"] != provenance:
            raise ValueError("Comparison model/preprocessing identity mismatch")
    summary = summarize_cascade(rows)
    variants = {}
    for mode in ("legacy", "coverage_consistency", "coverage"):
        matrix = [[0] * 7 for _ in range(6)]
        for row in rows:
            result = assess_review(row["result"], ReviewPolicy(mode))
            if mode == "legacy":
                available = [w for w in result["behavior"]["windows"] if w["status"] == "available"]
                chosen = next(
                    (
                        w
                        for w in available
                        if w["interval"] == result["final_alert"]["timestamp_interval"]
                    ),
                    available[0] if available else None,
                )
                prediction = (
                    chosen["probabilities"].index(max(chosen["probabilities"])) if chosen else 6
                )
            else:
                prediction = result["review_assessment"]["suggested_behavior_class"]
                prediction = 6 if prediction is None else prediction
            matrix[row["group_label"]][prediction] += 1
        covered = sum(sum(r[:6]) for r in matrix)
        correct = sum(matrix[i][i] for i in range(6))
        covered_targets = [i for i in range(6) for j in range(6) for _ in range(matrix[i][j])]
        covered_predictions = [j for i in range(6) for j in range(6) for _ in range(matrix[i][j])]
        variants[mode] = {
            "count": len(rows),
            "covered": covered,
            "coverage": covered / len(rows),
            "conditional_accuracy": correct / covered if covered else None,
            "conditional_error_rate": 1 - correct / covered if covered else None,
            "conditional_six_class_metrics": classification_metrics(
                covered_targets, covered_predictions, 6
            )
            if covered
            else None,
            "correct_fraction_all_clips": correct / len(rows),
            "confusion_rows_6_classes_columns_6_plus_abstain": matrix,
            "per_class_recall_all_clips": [
                matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None for i in range(6)
            ],
        }
    return {
        "scope": "matched_validation_only_no_threshold_tuning",
        "sources": len({r["source_video_id"] for r in rows}),
        "raw_clip_alert_all_modes": evaluate_binary(
            [r["binary_label"] for r in rows],
            [r["result"]["anomaly"]["overall_score"] for r in rows],
            summary["policy"]["anomaly_threshold"],
        ),
        "variants": variants,
        "uncertainty": "NOT ESTIMATED: report independent source support before inference",
    }
