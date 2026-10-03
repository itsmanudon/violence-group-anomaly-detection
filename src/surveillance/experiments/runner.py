"""Protocol-bound orchestration of existing trainers, with no test-set selection."""

import copy
import json
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

from surveillance.datasets.collective import (
    CLASSES,
    read_actor_manifest,
    write_actor_manifest,
)
from surveillance.datasets.collective_validation import validate_collective, validate_manifest
from surveillance.datasets.common import resolve_path
from surveillance.datasets.detected_actors import source_fingerprint
from surveillance.detection.config import DetectionConfig
from surveillance.detection.records import read_detections, write_detections
from surveillance.evaluation.box_robustness import evaluate_box_robustness
from surveillance.evaluation.group_activity_metrics import classification_metrics
from surveillance.experiments.environment import environment_metadata
from surveillance.experiments.protocol import (
    actor_config,
    content_hash,
    feature_expectations,
    file_hash,
    freeze_receipt,
    verify_receipt,
)
from surveillance.experiments.reporting import (
    analyze_predictions,
    select_feature_mode,
    write_error_report,
)
from surveillance.experiments.thresholds import tune_detector_thresholds
from surveillance.features.provenance import scene_fingerprint, validate_feature_caches
from surveillance.inference.group_activity_pipeline import GroupActivityPipeline
from surveillance.training.actor_transformer_trainer import load_checkpoint, train
from surveillance.training.sultani_trainer import select_device


def write_json(path: Path, value: dict | list) -> None:
    """Write finite, human-readable research artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def receipt_path(protocol: dict, experiment: str) -> Path:
    return Path(protocol["output_root"]) / "protocols" / f"{experiment}.json"


def validate_inputs(
    protocol: dict, experiment: str, *, allow_subset: bool = False
) -> tuple[dict, dict]:
    """Check split integrity and every cache, including empty detected scenes."""
    spec = protocol["experiments"][experiment]
    manifest = Path(spec["manifest"])
    report = validate_manifest(
        manifest,
        protocol["dataset"]["validation_sequences"],
        require_full_split=protocol["evidence_kind"] == "real" and not allow_subset,
        check_images=protocol["evidence_kind"] == "real",
    )
    if protocol["evidence_kind"] == "real":
        report["raw_data"] = validate_collective(
            Path(protocol["dataset"]["root"]),
            validation_sequences=protocol["dataset"]["validation_sequences"],
            manifest=Path(protocol["dataset"]["manifest"]) if allow_subset else manifest,
        )
        if allow_subset:
            authority = Path(protocol["dataset"]["manifest"])
            original = {(row.dataset, row.clip_id): row for row in read_actor_manifest(authority)}
            for row in read_actor_manifest(manifest):
                reference = original.get((row.dataset, row.clip_id))
                if reference is None or (
                    row.split,
                    scene_fingerprint(row),
                    source_fingerprint(row, manifest),
                ) != (
                    reference.split,
                    scene_fingerprint(reference),
                    source_fingerprint(reference, authority),
                ):
                    raise ValueError(
                        "Dry-run subset differs from the validated annotation authority"
                    )
    caches = validate_feature_caches(
        manifest,
        actor_config(protocol, experiment, protocol["seeds"][0]),
        feature_expectations(protocol, spec["mode"]),
        Path(spec["detections"]) if spec["box_source"] == "detections" else None,
    )
    if spec["box_source"] == "detections":
        validate_feature_caches(
            manifest,
            actor_config(protocol, experiment, protocol["seeds"][0]),
            feature_expectations(protocol, spec["mode"]),
        )
    return report, caches


def detector_selection(protocol: dict) -> dict:
    """Recompute a selection from validation candidates and reject altered receipts."""
    detector = protocol["detector"]
    selection = json.loads(Path(detector["selection"]).read_text(encoding="utf-8"))
    root = Path(protocol["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="selection_check_", dir=root) as temporary:
        actual = tune_detector_thresholds(
            Path(protocol["dataset"]["manifest"]),
            Path(detector["candidates"]),
            Path(temporary) / "receipt.json",
            detector["confidence_values"],
            DetectionConfig(**detector["filters"]),
            detector["matching_iou"],
            detector["min_recall"],
            protocol["dataset"]["validation_sequences"],
        )
    if selection != actual:
        raise ValueError("Detector receipt differs from the validation-only sweep")
    expected = detector["checkpoint_sha256"]
    if detector["checkpoint"]:
        digest = file_hash(Path(detector["checkpoint"]))
        if expected and expected != digest:
            raise ValueError("Detector checkpoint changed from protocol fingerprint")
        expected = digest
    if expected is None or actual["detector_checkpoint_sha256"] != expected:
        raise ValueError("Validation candidates use a different detector checkpoint")
    if protocol["evidence_kind"] == "real":
        manifest = Path(protocol["dataset"]["manifest"])
        rows = {(row.dataset, row.clip_id): row for row in read_actor_manifest(manifest)}
        for record in read_detections(Path(detector["candidates"])):
            image = resolve_path(rows[record.key].frame_paths[5], manifest)
            if record.result.metadata.get("reference_image_sha256") != file_hash(image):
                raise ValueError("Validation candidate image bytes changed or provenance is absent")
    return actual


def freeze_experiment(protocol: dict, experiment: str) -> dict:
    """Bind caches, configuration and validation selection before test evaluation."""
    validation, caches = validate_inputs(protocol, experiment)
    spec = protocol["experiments"][experiment]
    paths = {Path(spec["manifest"])}
    selection = None
    if spec["box_source"] == "detections":
        selection = detector_selection(protocol)
        candidate_runs = []
        for name, candidate in protocol["experiments"].items():
            if candidate["box_source"] != "ground_truth":
                continue
            runs = [
                Path(protocol["output_root"]) / name / f"seed_{seed}" / "metrics.json"
                for seed in protocol["seeds"]
            ]
            if all(path.exists() for path in runs):
                candidate_runs.extend(runs)
        winner = select_feature_mode(candidate_runs)
        if (
            any(
                json.loads(path.read_text(encoding="utf-8"))["protocol_hash"]
                != content_hash(protocol)
                for path in candidate_runs
            )
            or winner["dry_run"]
        ):
            raise ValueError("Mode selection requires completed non-dry runs of this protocol")
        if winner["selected_experiment"] != spec["checkpoint_experiment"]:
            raise ValueError("Detected experiment must use the strongest available validation mode")
        paths.update(candidate_runs)
        for path in candidate_runs:
            run = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = Path(run["selected_checkpoint"])
            if file_hash(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError("Validation checkpoint changed after measurement")
            paths.add(checkpoint)
        selection = {"detector": selection, "feature_mode": winner}
        paths.update(Path(protocol["detector"][key]) for key in ("candidates", "selection"))
        paths.add(Path(protocol["dataset"]["manifest"]))
        paths.add(Path(spec["detections"]))
        gt_receipt = verify_receipt(
            protocol,
            spec["checkpoint_experiment"],
            receipt_path(protocol, spec["checkpoint_experiment"]),
        )
        paths.update(Path(path) for path in gt_receipt["artifacts"])
        paths.add(receipt_path(protocol, spec["checkpoint_experiment"]))
        selected_hash = content_hash(selection["detector"])
        authority = Path(spec["manifest"])
        gt_rows = {(row.dataset, row.clip_id): row for row in read_actor_manifest(authority)}
        for record in read_detections(Path(spec["detections"])):
            if (
                record.result.metadata.get("detector_selection_hash") != selected_hash
                or record.result.metadata.get("filter_config")
                != selection["detector"]["selected_filters"]
                or record.result.metadata.get("checkpoint_sha256")
                != selection["detector"]["detector_checkpoint_sha256"]
                or record.result.metadata.get("untrained") is not False
            ):
                raise ValueError("Detected cache must use the frozen validation detector filters")
            if protocol["evidence_kind"] == "real":
                reference = resolve_path(gt_rows[record.key].frame_paths[5], authority)
                if record.result.metadata.get("reference_image_sha256") != file_hash(reference):
                    raise ValueError("Detected reference image changed or provenance is absent")
    rows = read_actor_manifest(Path(spec["manifest"]))
    if protocol["evidence_kind"] == "real":
        paths.update(
            resolve_path(frame, Path(spec["manifest"])) for row in rows for frame in row.frame_paths
        )
        paths.update(Path(protocol["dataset"]["root"]).glob("seq*/annotations.txt"))
    records = read_detections(Path(spec["detections"])) if selection else rows
    source = Path(spec["detections"]) if selection else Path(spec["manifest"])
    for record in records:
        for modality in feature_expectations(protocol, spec["mode"]):
            path = resolve_path(getattr(record, f"{modality}_feature_path"), source)
            paths.update((path, path.with_suffix(".json")))
    for modality in feature_expectations(protocol, spec["mode"]):
        checkpoint = protocol["features"][modality]["checkpoint"]
        if checkpoint:
            paths.add(Path(checkpoint))
    if selection and protocol["detector"]["checkpoint"]:
        paths.add(Path(protocol["detector"]["checkpoint"]))
    receipt = freeze_receipt(
        protocol,
        experiment,
        {str(p.resolve()): file_hash(p) for p in sorted(paths)},
        selection,
        receipt_path(protocol, experiment),
    )
    output = Path(protocol["output_root"]) / "protocols"
    write_json(output / f"{experiment}_dataset.json", validation)
    write_json(output / f"{experiment}_features.json", caches)
    return receipt


def population(manifest: Path, split: str, detected: list[dict] | None = None) -> dict:
    """Bind ordered scenes and GT supervision, or exact matched actor identities."""
    rows = [row for row in read_actor_manifest(manifest) if row.split == split]
    payload = [{"scene": scene_fingerprint(row), "split": row.split} for row in rows]
    actors = sum(len(row.actor_labels) for row in rows)
    if detected is not None:
        by_id = {
            (scene["metadata"]["dataset"], scene["metadata"]["clip_id"]): scene
            for scene in detected
        }
        if set(by_id) != {(row.dataset, row.clip_id) for row in rows}:
            raise ValueError("Detected comparison population differs from GT")
        actors = 0
        for row, value in zip(rows, payload, strict=True):
            scene = by_id[(row.dataset, row.clip_id)]
            value["matching"] = scene["matching"]
            value["detected_boxes"] = [actor["box_pixels"] for actor in scene["actors"]]
            actors += len(scene["matching"]["gt_indices"])
    return {
        "hash": content_hash(payload),
        "scene_ids": [row.clip_id for row in rows],
        "scene_count": len(rows),
        "actor_count": actors,
        "split": split,
        "box_source": "detections" if detected is not None else "ground_truth",
        "actor_population": "matched_only" if detected is not None else "all_gt",
    }


def prediction_metrics(scenes: list[dict]) -> dict:
    actors = [actor for scene in scenes for actor in scene["actors"]]
    return {
        "group": classification_metrics(
            [s["group_label"] for s in scenes],
            [s["group_prediction"] for s in scenes],
            len(CLASSES),
        ),
        "actor": classification_metrics(
            [a["label"] for a in actors], [a["prediction"] for a in actors], len(CLASSES)
        ),
    }


def run_experiment(
    protocol: dict,
    experiment: str,
    seed: int,
    *,
    dry_run: bool = False,
    max_scenes: int | None = None,
    max_iterations: int | None = None,
) -> dict:
    """Train GT once; evaluate detected boxes with its checkpoint and unchanged model.

    Dry runs evaluate only validation scenes and live in ``preflight``. Real test
    evaluation requires an unchanged freeze receipt. Existing run outputs are
    never overwritten. Bounds are permitted only for explicitly marked dry runs.
    """
    if not dry_run and (max_scenes is not None or max_iterations is not None):
        raise ValueError("Subset/iteration overrides require --dry-run")
    if dry_run and (not max_scenes or not max_iterations or min(max_scenes, max_iterations) < 1):
        raise ValueError("Dry runs require positive max_scenes and max_iterations")
    config = actor_config(protocol, experiment, seed)
    spec = protocol["experiments"][experiment]
    manifest = Path(spec["manifest"])
    detections = Path(spec["detections"]) if spec["box_source"] == "detections" else None
    root = Path(protocol["output_root"])
    if dry_run:
        root /= "preflight"
    output = root / experiment / f"seed_{seed}"
    if output.exists():
        raise FileExistsError(f"Run already exists; choose a new output_root: {output}")
    validate_inputs(protocol, experiment, allow_subset=dry_run)
    if not dry_run:
        verify_receipt(protocol, experiment, receipt_path(protocol, experiment))
    output.mkdir(parents=True)
    if dry_run:
        config = copy.deepcopy(config)
        config["training"]["max_iterations"] = max_iterations
        rows = read_actor_manifest(manifest)
        limited = []
        for split in ("train", "val"):
            selected = [row for row in rows if row.split == split][:max_scenes]
            if not selected:
                raise ValueError("Dry run requires nonempty training and validation sources")
            limited.extend(selected)
        # Absolute references keep subset relocation faithful to the original manifest.
        limited = [
            replace(
                row,
                frame_paths=[str(resolve_path(p, manifest)) for p in row.frame_paths],
                pose_feature_path=str(resolve_path(row.pose_feature_path, manifest))
                if row.pose_feature_path
                else None,
                rgb_feature_path=str(resolve_path(row.rgb_feature_path, manifest))
                if row.rgb_feature_path
                else None,
            )
            for row in limited
        ]
        manifest = output / "subset.jsonl"
        write_actor_manifest(limited, manifest)
        if detections is not None:
            keys = {(row.dataset, row.clip_id) for row in limited}
            records = [
                replace(
                    record,
                    pose_feature_path=str(resolve_path(record.pose_feature_path, detections))
                    if record.pose_feature_path
                    else None,
                    rgb_feature_path=str(resolve_path(record.rgb_feature_path, detections))
                    if record.rgb_feature_path
                    else None,
                )
                for record in read_detections(detections)
                if record.key in keys
            ]
            detections = output / "subset_detections.jsonl"
            write_detections(records, detections)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(output / "resolved_protocol.json", protocol)
    configuration_hash = content_hash(config)
    device = select_device(config["device"])
    environment = environment_metadata(configuration_hash, seed, str(device))
    write_json(output / "environment.json", environment)
    if spec["box_source"] == "ground_truth":
        train(config, manifest, output)
        checkpoint = output / "best.pt"
    else:
        reference = root / spec["checkpoint_experiment"] / f"seed_{seed}"
        checkpoint = reference / "best.pt"
        reference_run = json.loads((reference / "metrics.json").read_text(encoding="utf-8"))
        if (
            reference_run["protocol_hash"] != content_hash(protocol)
            or reference_run["seed"] != seed
            or reference_run["dry_run"] != dry_run
        ):
            raise ValueError("Detected evaluation must reuse the matching frozen GT seed run")
        if file_hash(checkpoint) != reference_run["checkpoint_sha256"]:
            raise ValueError("GT checkpoint changed since its baseline evaluation")
    system, saved = load_checkpoint(checkpoint, device)
    if saved["config"] != config:
        raise ValueError("Selected checkpoint configuration differs from resolved run")
    if not dry_run:
        # Training may be long: verify again at the evaluation boundary.
        verify_receipt(protocol, experiment, receipt_path(protocol, experiment))
        if spec["box_source"] == "detections":
            verify_receipt(
                protocol,
                spec["checkpoint_experiment"],
                receipt_path(protocol, spec["checkpoint_experiment"]),
            )
    pipeline = GroupActivityPipeline(system)
    validation_scenes = pipeline.predict_manifest(manifest, "val")
    split = "val" if dry_run else "test"
    if not dry_run:
        verify_receipt(protocol, experiment, receipt_path(protocol, experiment))
    detected_scenes = None
    if spec["box_source"] == "ground_truth":
        scenes = pipeline.predict_manifest(manifest, split)
        metrics = prediction_metrics(scenes)
    else:
        comparison = evaluate_box_robustness(
            checkpoint,
            manifest,
            detections,
            split,
            protocol["detector"]["matching_iou"],
            str(device),
        )
        scenes = comparison.pop("ground_truth_predictions")
        detected_scenes = comparison.pop("detected_predictions")
        metrics = comparison
        detection = comparison["detected_boxes"]
        detection["empty_scene_rate"] = detection["no_actor_scene_count"] / len(scenes)
        detection["mean_actors_per_scene"] = sum(len(s["actors"]) for s in detected_scenes) / len(
            scenes
        )
        write_json(output / "comparison.json", comparison)
    if not dry_run:
        verify_receipt(protocol, experiment, receipt_path(protocol, experiment))
    errors = analyze_predictions(scenes, detected_scenes, classes=CLASSES)
    write_error_report(errors, output / "errors.json")
    write_json(output / "predictions.json", scenes)
    if detected_scenes is not None:
        write_json(output / "detected_predictions.json", detected_scenes)
    result = {
        "schema_version": 1,
        "protocol_hash": content_hash(protocol),
        "experiment": experiment,
        "feature_mode": spec["mode"],
        "seed": seed,
        "evidence_kind": protocol["evidence_kind"],
        "dry_run": dry_run,
        "population": population(manifest, split, detected_scenes),
        "metrics": metrics,
        "scene_count": len(scenes),
        "gt_actor_count": sum(len(scene["actors"]) for scene in scenes),
        "invalid_scene_count": 0,
        "abstained_scene_count": sum(scene["status"] != "ok" for scene in detected_scenes)
        if detected_scenes is not None
        else 0,
        "validation_metrics": prediction_metrics(validation_scenes),
        "validation_population": population(manifest, "val"),
        "selected_checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_hash(checkpoint),
        "config_hash": configuration_hash,
        "environment": environment,
        "group_class_distribution": dict(Counter(CLASSES[s["group_label"]] for s in scenes)),
        "actor_class_distribution": dict(
            Counter(CLASSES[a["label"]] for s in scenes for a in s["actors"])
        ),
        "result_scope": "NON-BENCHMARK PREFLIGHT"
        if dry_run or protocol["evidence_kind"] == "synthetic"
        else "REAL COLLECTIVE BENCHMARK; paper-exact split unverified",
    }
    write_json(output / "metrics.json", result)
    (output / "summary.md").write_text(
        f"{result['result_scope']}\n\nExperiment: {experiment}; seed: {seed}; split: {split}.\n"
        f"Scenes: {len(scenes)}. Checkpoint: {checkpoint}.\n"
        "See metrics.json, errors.json, environment.json and history.jsonl.\n",
        encoding="utf-8",
    )
    return result
