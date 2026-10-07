"""Descriptive prediction audits and strict same-population seed aggregation."""

import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from surveillance.datasets.collective import CLASSES
from surveillance.detection.matching import MatchResult
from surveillance.evaluation.group_activity_metrics import classification_metrics


def _finite_tree(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")


def read_json(path: Path):
    """Read strict JSON, rejecting nonstandard NaN/Infinity input."""

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)
    _finite_tree(value)
    return value


def _class(value, classes, nullable=False):
    if value is None and nullable:
        return None
    if type(value) is not int or value not in range(len(classes)):
        raise ValueError("Class labels must be integers within the class vocabulary")
    return value


def confusion_summary(labels, predictions, classes=CLASSES) -> dict:
    """Keep the actual class vocabulary and rank directed off-diagonal errors."""
    classes = list(classes)
    if len(labels) != len(predictions) or not classes or len(set(classes)) != len(classes):
        raise ValueError("Confusion inputs require aligned labels and a unique vocabulary")
    matrix = [[0] * len(classes) for _ in classes]
    for truth, predicted in zip(labels, predictions, strict=True):
        matrix[_class(truth, classes)][_class(predicted, classes)] += 1
    pairs = [
        {
            "true_label": i,
            "true_class": classes[i],
            "predicted_label": j,
            "predicted_class": classes[j],
            "count": matrix[i][j],
        }
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and matrix[i][j]
    ]
    pairs.sort(key=lambda pair: (-pair["count"], pair["true_label"], pair["predicted_label"]))
    return {
        "classes": classes,
        "matrix": matrix,
        "count": len(labels),
        "off_diagonal_count": sum(pair["count"] for pair in pairs),
        "confusions": pairs,
    }


def confidence_summary(values, *, include_values=True) -> dict:
    """Population spread is descriptive; absent confidence stays unknown."""
    known = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in known):
        raise ValueError("Confidence must be finite and in [0,1]")
    result = {
        "count": len(known),
        "unknown_count": len(values) - len(known),
        "mean": statistics.mean(known) if known else None,
        "std": statistics.pstdev(known) if known else None,
        "quantiles": {
            str(q): float(np.quantile(known, q)) if known else None
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
    }
    if include_values:
        result["values"] = known
    return result


def _confidence(scene):
    if scene.get("group_prediction") is None:
        return None
    value = scene.get("group_confidence")
    if value is None and scene.get("group_probabilities") is not None:
        probabilities = scene["group_probabilities"]
        value = probabilities[scene["group_prediction"]]
    if value is not None:
        confidence_summary([value])
    return value


def _identity(scene):
    metadata = scene.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Scene metadata must be an object")
    if not isinstance(metadata.get("clip_id"), str) or not metadata["clip_id"]:
        raise ValueError("Scene metadata requires a nonempty clip_id")
    return tuple(metadata.get(key) for key in ("dataset", "clip_id", "source_video_id"))


def _empty_metrics(classes):
    return {
        "count": 0,
        "accuracy": None,
        "macro_f1": None,
        "per_class_accuracy": [None] * len(classes),
        "per_class_f1": [None] * len(classes),
        "support": [0] * len(classes),
        "confusion_matrix": [[0] * len(classes) for _ in classes],
    }


def _prediction_audit(scenes, classes):
    labels, predictions, actor_labels, actor_predictions = [], [], [], []
    correct_confidences, incorrect_confidences, abstained = [], [], []
    for scene in scenes:
        truth = _class(scene["group_label"], classes)
        prediction = _class(scene.get("group_prediction"), classes, nullable=True)
        confidence = _confidence(scene)
        if prediction is None:
            abstained.append(confidence)
        else:
            labels.append(truth)
            predictions.append(prediction)
            destination = correct_confidences if truth == prediction else incorrect_confidences
            destination.append(confidence)
        for actor in scene.get("actors", []):
            if actor.get("label") in (None, -100):
                continue
            actor_labels.append(_class(actor["label"], classes))
            actor_predictions.append(_class(actor["prediction"], classes))
    return {
        "group": classification_metrics(labels, predictions, len(classes))
        if labels
        else _empty_metrics(classes),
        "actor": classification_metrics(actor_labels, actor_predictions, len(classes))
        if actor_labels
        else _empty_metrics(classes),
        "confusion": confusion_summary(labels, predictions, classes),
        "confidence": {
            "correct": confidence_summary(correct_confidences),
            "incorrect": confidence_summary(incorrect_confidences),
            "abstained": confidence_summary(abstained),
        },
        "scene_count": len(scenes),
        "abstention_count": len(abstained),
        "accuracy_all_scenes_abstentions_incorrect": (
            sum(truth == prediction for truth, prediction in zip(labels, predictions, strict=True))
            / len(scenes)
            if scenes
            else None
        ),
    }


def analyze_predictions(
    gt: list[dict],
    detected: list[dict] | None = None,
    classes=CLASSES,
    high_confidence: float = 0.8,
    low_coverage: float = 0.5,
) -> dict:
    """Audit ordered scene pairs, without inferring why predictions changed.

    Unmatched detections can be unannotated people; they do not establish false
    detections or a behavioral cause. No-actor scenes retain null predictions.
    """
    classes = list(classes)
    confidence_summary([high_confidence, low_coverage])
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Classes must be a nonempty unique vocabulary")
    _finite_tree(gt)
    if detected is not None:
        _finite_tree(detected)
        if len(gt) != len(detected):
            raise ValueError("GT/detected scene count mismatch")
    identities = [_identity(scene) for scene in gt]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate scene identity")
    records = []
    for index, scene in enumerate(gt):
        truth = _class(scene["group_label"], classes)
        prediction = _class(scene.get("group_prediction"), classes, nullable=True)
        gt_correct = prediction == truth
        confidence = _confidence(scene)
        categories = ["GT_correct" if gt_correct else "GT_group_wrong"]
        record = {
            "scene_id": scene["metadata"]["clip_id"],
            "metadata": {
                key: scene["metadata"].get(key) for key in ("dataset", "clip_id", "source_video_id")
            },
            "true_label": truth,
            "true_class": classes[truth],
            "gt_prediction": prediction,
            "gt_predicted_class": classes[prediction] if prediction is not None else None,
            "gt_confidence": confidence,
            "gt_actor_count": len(scene.get("actors", [])),
            "gt_correct": gt_correct,
        }
        if not gt_correct and confidence is not None and confidence >= high_confidence:
            categories.append("GT_high_confidence_wrong")
        if detected is not None:
            det = detected[index]
            if _identity(det) != identities[index] or any(
                scene["metadata"].get(key) != det["metadata"].get(key)
                for key in ("video_id", "split", "frame_indices")
            ):
                raise ValueError("GT/detected scene identity or ordering mismatch")
            if det.get("group_label") != truth:
                raise ValueError("GT/detected group labels differ")
            det_prediction = _class(det.get("group_prediction"), classes, nullable=True)
            det_confidence = _confidence(det)
            status = det.get("status", "ok" if det_prediction is not None else "unknown")
            no_actors = status == "no_actors_detected"
            if no_actors and (det.get("actors") or det_prediction is not None):
                raise ValueError("No-actor status requires empty actors and a null prediction")
            det_correct = det_prediction == truth
            if gt_correct and not det_correct:
                categories.append("GT_correct_detected_failed")
            elif not gt_correct and not det_correct:
                categories.append("both_failed")
            elif det_correct:
                categories.append("detected_correct")
            if no_actors:
                categories.append("no_actors_detected")
            elif not det_correct:
                categories.append("detected_group_wrong")
            if not det_correct and det_confidence is not None and det_confidence >= high_confidence:
                categories.append("detected_high_confidence_wrong")
            matching = det.get("matching")
            coverage, ious, matched_predictions = None, [], []
            if matching is not None:
                match = MatchResult(**matching)
                if match.gt_count != len(scene.get("actors", [])) or (
                    match.detection_count != len(det.get("actors", []))
                ):
                    raise ValueError("Matching counts differ from prediction actor populations")
                coverage = len(match.gt_indices) / match.gt_count if match.gt_count else None
                ious = list(match.ious)
                for gi, di, iou in zip(
                    match.gt_indices, match.detection_indices, match.ious, strict=True
                ):
                    actor, detected_actor = scene["actors"][gi], det["actors"][di]
                    if actor.get("label") != detected_actor.get("label"):
                        raise ValueError("Matched actor labels differ from GT")
                    matched_predictions.append(
                        {
                            "gt_index": gi,
                            "detection_index": di,
                            "iou": iou,
                            "label": actor.get("label"),
                            "gt_prediction": actor.get("prediction"),
                            "detected_prediction": detected_actor.get("prediction"),
                        }
                    )
            if coverage is not None and coverage < low_coverage:
                categories.append("low_coverage")
            record.update(
                detected_prediction=det_prediction,
                detected_predicted_class=classes[det_prediction]
                if det_prediction is not None
                else None,
                detected_confidence=det_confidence,
                detected_correct=det_correct,
                detected_status=status,
                detected_actor_count=len(det.get("actors", [])),
                detection_coverage=coverage,
                matched_ious=ious,
                mean_matched_iou=statistics.mean(ious) if ious else None,
                matched_actor_predictions=matched_predictions,
                missed_gt_indices=matching["missed_gt"] if matching else None,
                unmatched_detection_indices=matching["unmatched_detections"] if matching else None,
            )
        record["categories"] = categories
        records.append(record)
    counts = Counter(category for record in records for category in record["categories"])
    return {
        "schema_version": 1,
        "classes": classes,
        "scene_count": len(records),
        "thresholds": {"high_confidence": high_confidence, "low_coverage": low_coverage},
        "category_counts": dict(sorted(counts.items())),
        "scenes": records,
        "ground_truth": _prediction_audit(gt, classes),
        "detected": _prediction_audit(detected, classes) if detected is not None else None,
        "interpretation": "Categories describe observed predictions, not behavioral causes. "
        "Unmatched detections may be unannotated people. Actor metrics use annotated actors; "
        "detected actor metrics are conditional on matching. Null confidence remains unknown.",
    }


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def error_report_markdown(report: dict) -> str:
    lines = [
        "# Collective prediction error analysis",
        "",
        report["interpretation"],
        "",
        f"Scenes: {report['scene_count']}",
        "",
        "| Category | Scenes |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {_cell(key)} | {count} |" for key, count in report["category_counts"].items())
    lines.extend(
        [
            "",
            "| Scene | Truth | GT prediction | Detected prediction | Coverage | Categories |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in report["scenes"]:
        values = (
            row["scene_id"],
            row["true_class"],
            row["gt_predicted_class"],
            row.get("detected_predicted_class"),
            row.get("detection_coverage"),
            ", ".join(row["categories"]),
        )
        cells = (_cell(v) if v is not None else "unknown" for v in values)
        lines.append("| " + " | ".join(cells) + " |")
    for source in ("ground_truth", "detected"):
        audit = report.get(source)
        if audit is None:
            continue
        lines.extend(
            [
                "",
                f"## {source.replace('_', ' ').title()}",
                "",
                "| Group predictions | Known confidence count | Mean | Std | Unknown |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for category, confidence in audit["confidence"].items():
            values = (
                category,
                confidence["count"],
                confidence["mean"],
                confidence["std"],
                confidence["unknown_count"],
            )
            cells = (_cell(v) if v is not None else "unknown" for v in values)
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(["", "| True group | Predicted group | Error count |", "| --- | --- | ---: |"])
        for pair in audit["confusion"]["confusions"]:
            lines.append(
                f"| {_cell(pair['true_class'])} | "
                f"{_cell(pair['predicted_class'])} | {pair['count']} |"
            )
        lines.extend(
            [
                "",
                "| Actor class | Annotated support | Accuracy | F1 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for index, name in enumerate(report["classes"]):
            actor = audit["actor"]
            values = (
                name,
                actor["support"][index],
                actor["per_class_accuracy"][index],
                actor["per_class_f1"][index],
            )
            cells = (_cell(v) if v is not None else "unknown" for v in values)
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_error_report(report: dict, output: Path) -> dict:
    """Write machine JSON and human Markdown; do not read or copy source imagery."""
    output = Path(output)
    json_path = output if output.suffix.lower() == ".json" else output / "errors.json"
    markdown_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(error_report_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _flatten(value, prefix="metrics"):
    if not isinstance(value, dict):
        raise ValueError("Run metrics must be an object")
    leaves = {}
    for key, child in value.items():
        path = f"{prefix}.{key}"
        if isinstance(child, dict):
            leaves.update(_flatten(child, path))
        else:
            leaves[path] = child
    return leaves


def _validate_run(run):
    required = (
        "schema_version",
        "protocol_hash",
        "experiment",
        "seed",
        "evidence_kind",
        "dry_run",
        "population",
        "metrics",
        "config_hash",
    )
    if not isinstance(run, dict) or any(key not in run for key in required):
        raise ValueError("Run is missing required experiment/provenance fields")
    if run["schema_version"] != 1 or type(run["seed"]) is not int:
        raise ValueError("Unsupported run schema or invalid seed")
    if run["evidence_kind"] not in {"real", "synthetic"} or type(run["dry_run"]) is not bool:
        raise ValueError("Run must declare real/synthetic evidence and boolean dry_run")
    for key in ("protocol_hash", "experiment", "config_hash"):
        if not isinstance(run[key], str) or not run[key]:
            raise ValueError(f"Run requires nonempty {key}")
    population = run["population"]
    fields = (
        "hash",
        "scene_ids",
        "scene_count",
        "actor_count",
        "box_source",
        "split",
        "actor_population",
    )
    if not isinstance(population, dict) or any(key not in population for key in fields):
        raise ValueError("Run requires explicit population identity, counts and policy")
    ids = population["scene_ids"]
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("Population scene_ids must be nonempty strings")
    if len(set(ids)) != len(ids) or population["scene_count"] != len(ids):
        raise ValueError("Population scene IDs are duplicated or scene_count differs")
    for key in ("scene_count", "actor_count"):
        if type(population[key]) is not int or population[key] < 0:
            raise ValueError("Population counts must be nonnegative integers")
    if not isinstance(population["hash"], str) or not population["hash"]:
        raise ValueError("Population hash must be nonempty")
    if population["split"] not in {"val", "test"}:
        raise ValueError("Aggregation requires a val/test population")
    if population["actor_population"] not in {"all_gt", "matched_only"}:
        raise ValueError("Actor population must be all_gt or matched_only")
    _finite_tree(run)
    return _flatten(run["metrics"])


def aggregate_experiments(paths: list[Path]) -> dict:
    """Aggregate only one experiment with identical protocol and population.

    Scalars report finite observations, null observations and sample standard
    deviation. Counts, arrays and nonnumeric leaves are retained per seed only.
    Config hashes may differ because they include the seed.
    """
    if not paths:
        raise ValueError("At least one run is required")
    runs = [read_json(path) for path in paths]
    leaves = [_validate_run(run) for run in runs]
    first = runs[0]
    for run in runs[1:]:
        for key in ("protocol_hash", "experiment", "evidence_kind", "dry_run", "population"):
            if run[key] != first[key]:
                raise ValueError(f"Incompatible runs: {key} differs")
        if run.get("feature_mode") != first.get("feature_mode"):
            raise ValueError("Incompatible runs: feature_mode differs")
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("Duplicate seed in aggregate")
    if any(set(leaf) != set(leaves[0]) for leaf in leaves[1:]):
        raise ValueError("Metric paths differ between runs")
    ordered = sorted(zip(runs, leaves, paths, strict=True), key=lambda item: item[0]["seed"])
    scalar_metrics, individual_metrics = {}, {}
    for path in sorted(leaves[0]):
        observations = [{"seed": run["seed"], "value": leaf[path]} for run, leaf, _ in ordered]
        values = [item["value"] for item in observations]
        scalar = all(value is None or type(value) in (int, float) for value in values)
        name = path.rsplit(".", 1)[-1]
        individual_only = (
            name == "count"
            or name.endswith(("_count", "_counts"))
            or name
            in {
                "matches",
                "missed_gt_actors",
                "unmatched_detections",
                "empty_frames",
                "confusion_matrix",
                "support",
                "per_class_accuracy",
                "per_class_f1",
            }
        )
        if scalar and not individual_only:
            known = [value for value in values if value is not None]
            scalar_metrics[path] = {
                "count": len(known),
                "null_count": len(values) - len(known),
                "mean": statistics.mean(known) if known else None,
                "std": statistics.stdev(known) if len(known) >= 2 else None,
                "seed_values": observations,
            }
        else:
            individual_metrics[path] = observations
    return {
        "schema_version": 1,
        "protocol_hash": first["protocol_hash"],
        "experiment": first["experiment"],
        "feature_mode": first.get("feature_mode"),
        "evidence_kind": first["evidence_kind"],
        "dry_run": first["dry_run"],
        "population": first["population"],
        "seed_count": len(runs),
        "seeds": [run["seed"] for run, _, _ in ordered],
        "metrics": scalar_metrics,
        "individual_metrics": individual_metrics,
        "runs": [
            {
                "seed": run["seed"],
                "config_hash": run["config_hash"],
                "selected_checkpoint": run.get("selected_checkpoint"),
                "validation_metrics": run.get("validation_metrics"),
                "environment": run.get("environment"),
                "metrics_path": str(path),
            }
            for run, _, path in ordered
        ],
        "aggregation_policy": "Same experiment, protocol, evidence, dry-run mode and full "
        "population identity only. Sample std uses ddof=1; null is not zero. Counts and "
        "arrays remain individual. Matched-only population hashes must include match identity.",
    }


def aggregate_table_markdown(aggregates: list[dict]) -> str:
    """Produce explicit, possibly partial rows without merging different experiments."""
    lines = [
        "# Collective experiment aggregation",
        "",
        "Null values are unknown. Spread is sample standard deviation across seeds.",
        "Detected accuracy counts empty-scene abstentions as incorrect; detected macro F1 "
        "uses supported scenes. GT rows use all scenes. Read coverage alongside these rows.",
        "",
        "| Experiment | Mode | Evidence | Dry run | Seeds | Group accuracy | Macro F1 | "
        "Population |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]

    def metric(row, key):
        value = row["metrics"].get(key)
        if value is None:
            return "unknown"
        if value["mean"] is None:
            return "unknown"
        spread = "unknown" if value["std"] is None else f"{value['std']:.4f}"
        return f"{value['mean']:.4f} ± {spread} (n={value['count']})"

    for row in aggregates:
        population = row["population"]
        detected = population["box_source"] == "detections"
        accuracy = (
            "metrics.detected_boxes.group.accuracy_all_scenes_abstentions_incorrect"
            if detected
            else "metrics.group.accuracy"
        )
        macro_f1 = "metrics.detected_boxes.group.macro_f1" if detected else "metrics.group.macro_f1"
        identity = (
            f"{population['box_source']}/{population['actor_population']}; "
            f"{population['scene_count']} scenes, {population['actor_count']} actors; "
            f"{population['hash']}; protocol {row['protocol_hash']}"
        )
        values = (
            row["experiment"],
            row.get("feature_mode") or "unknown",
            row["evidence_kind"],
            row["dry_run"],
            row["seed_count"],
            metric(row, accuracy),
            metric(row, macro_f1),
            identity,
        )
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def select_feature_mode(paths: list[Path]) -> dict:
    """Select among available modes using validation macro F1, never test results.

    All candidates must have the same validation population and seed set. A
    partial mode table is allowed and its scope is explicit in the receipt.
    """
    if not paths:
        raise ValueError("Feature selection requires validation runs")
    candidates = {}
    reference_population, reference_evidence, reference_dry = None, None, None
    for path in paths:
        run = read_json(path)
        _validate_run(run)
        population = run.get("validation_population")
        if not isinstance(population, dict) or population.get("split") != "val":
            raise ValueError("Feature selection requires an explicit validation_population")
        # Reuse the population gate independently of the evaluation population.
        validation_run = {**run, "population": population}
        _validate_run(validation_run)
        if reference_population is None:
            reference_population = population
            reference_evidence, reference_dry = run["evidence_kind"], run["dry_run"]
        elif population != reference_population:
            raise ValueError("Validation population identity differs across seeds or modes")
        if run["evidence_kind"] != reference_evidence or run["dry_run"] != reference_dry:
            raise ValueError("Validation candidates have incompatible evidence or dry-run modes")
        mode = run.get("feature_mode")
        if not isinstance(mode, str) or not mode:
            raise ValueError("Validation run requires feature_mode")
        try:
            group = run["validation_metrics"]["group"]
            accuracy, macro_f1 = group["accuracy"], group["macro_f1"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Validation metrics require group.accuracy and group.macro_f1"
            ) from error
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in (accuracy, macro_f1)
        ):
            raise ValueError("Feature selection requires finite validation accuracy and macro F1")
        experiment = run["experiment"]
        candidate = candidates.setdefault(
            experiment,
            {
                "experiment": experiment,
                "feature_mode": mode,
                "protocol_hash": run["protocol_hash"],
                "seed_values": [],
            },
        )
        if candidate["feature_mode"] != mode or candidate["protocol_hash"] != run["protocol_hash"]:
            raise ValueError("Validation candidate mode or protocol differs across seeds")
        if any(value["seed"] == run["seed"] for value in candidate["seed_values"]):
            raise ValueError("Duplicate validation seed for a candidate")
        candidate["seed_values"].append(
            {
                "seed": run["seed"],
                "group_accuracy": accuracy,
                "group_macro_f1": macro_f1,
                "config_hash": run["config_hash"],
                "selected_checkpoint": run.get("selected_checkpoint"),
                "metrics_path": str(path),
            }
        )
    reference_seeds = None
    for candidate in candidates.values():
        candidate["seed_values"].sort(key=lambda value: value["seed"])
        seeds = [value["seed"] for value in candidate["seed_values"]]
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
            raise ValueError("Validation candidates must use identical seed sets")
        candidate["seed_count"] = len(seeds)
        candidate["mean_validation_group_macro_f1"] = statistics.mean(
            value["group_macro_f1"] for value in candidate["seed_values"]
        )
        candidate["mean_validation_group_accuracy"] = statistics.mean(
            value["group_accuracy"] for value in candidate["seed_values"]
        )
    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate["mean_validation_group_macro_f1"],
            -candidate["mean_validation_group_accuracy"],
            candidate["experiment"],
        ),
    )
    winner = ranked[0]
    return {
        "schema_version": 1,
        "selected_experiment": winner["experiment"],
        "feature_mode": winner["feature_mode"],
        "mode": winner["feature_mode"],
        "selection_metric": "mean_validation_group_macro_f1",
        "selection_value": winner["mean_validation_group_macro_f1"],
        "tie_break": "mean validation group accuracy descending, then experiment name ascending",
        "split": "val",
        "validation_population": reference_population,
        "evidence_kind": reference_evidence,
        "dry_run": reference_dry,
        "seeds": reference_seeds,
        "seed_values": winner["seed_values"],
        "candidates": ranked,
        "scope": "Selection is limited to the available validated candidates in this receipt; "
        "it does not establish the strongest mode among untested candidates.",
    }
