"""Real orchestration with artificial inputs, including misses, extras and abstention."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from surveillance.datasets.collective import read_actor_manifest, write_actor_manifest
from surveillance.detection.config import DetectionConfig
from surveillance.experiments.protocol import load_protocol, verify_receipt
from surveillance.experiments.runner import (
    detector_selection,
    freeze_experiment,
    receipt_path,
    run_experiment,
)
from surveillance.experiments.synthetic import smoke_workflow, synthetic_protocol
from surveillance.experiments.thresholds import tune_detector_thresholds


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    destination = tmp_path_factory.mktemp("orchestration") / "smoke"
    smoke_workflow(destination)
    return load_protocol(destination / "protocol.yaml")


def result(protocol, experiment, seed=0):
    return json.loads(
        (Path(protocol["output_root"]) / experiment / f"seed_{seed}" / "metrics.json").read_text()
    )


def test_two_seed_gt_and_detected_runs_reuse_identical_checkpoints(workflow):
    for seed in workflow["seeds"]:
        gt = result(workflow, "pose_gt", seed)
        det = result(workflow, "detected_pose", seed)
        assert gt["selected_checkpoint"] == det["selected_checkpoint"]
        assert gt["checkpoint_sha256"] == det["checkpoint_sha256"]
        assert gt["population"]["scene_count"] == det["population"]["scene_count"] == 2
        assert gt["population"]["actor_count"] == 5
        assert det["population"]["actor_count"] == 1
        assert det["population"]["actor_population"] == "matched_only"
        assert gt["validation_population"]["hash"] == det["validation_population"]["hash"]
        assert gt["result_scope"].startswith("NON-BENCHMARK")


def test_variable_actors_misses_extras_and_empty_scene_are_recorded(workflow):
    det = result(workflow, "detected_pose")
    comparison = det["metrics"]
    detection = comparison["detected_boxes"]
    assert detection["no_actor_scene_count"] == 1
    assert detection["empty_scene_rate"] == 0.5
    assert detection["actor_matched_only"]["count"] == 1
    assert detection["detection"]["missed_gt_actors"] == 4
    assert detection["detection"]["unmatched_detections"] == 1
    assert comparison["delta"]["group_accuracy_all_scenes"] is not None
    path = Path(workflow["output_root"]) / "detected_pose" / "seed_0" / "errors.json"
    errors = json.loads(path.read_text())
    assert errors["category_counts"]["no_actors_detected"] == 1
    assert len(errors["scenes"]) == 2


def test_aggregation_and_environment_are_saved(workflow):
    root = Path(workflow["output_root"])
    aggregate = json.loads((root / "pose_gt_aggregate.json").read_text())
    assert aggregate["seed_count"] == 2
    metric = aggregate["metrics"]["metrics.group.accuracy"]
    assert metric["std"] is not None and len(metric["seed_values"]) == 2
    environment = result(workflow, "pose_gt")["environment"]
    assert "hostname" not in environment and "username" not in environment
    assert (root / "ablations.md").exists()


def test_completed_run_is_never_overwritten(workflow):
    with pytest.raises(FileExistsError):
        run_experiment(workflow, "pose_gt", 0)


def test_detector_receipt_rejects_test_selection_tampering(tmp_path):
    protocol = synthetic_protocol(tmp_path / "fixture")
    path = Path(protocol["detector"]["selection"])
    receipt = json.loads(path.read_text())
    receipt["selected_confidence"] = 0.3
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="validation-only"):
        detector_selection(protocol)


def test_official_test_sources_cannot_be_relabeled_validation(tmp_path):
    protocol = synthetic_protocol(tmp_path / "fixture")
    manifest = Path(protocol["dataset"]["manifest"])
    rows = [
        replace(row, split="val") if row.split == "test" else row
        for row in read_actor_manifest(manifest)
    ]
    write_actor_manifest(rows, manifest)
    with pytest.raises(ValueError, match="Official test sources"):
        tune_detector_thresholds(
            manifest,
            Path(protocol["detector"]["candidates"]),
            tmp_path / "invalid.json",
            [0.5],
            DetectionConfig(),
        )


def test_detected_receipt_binds_ground_truth_features(workflow):
    source = Path(workflow["dataset"]["manifest"])
    feature = Path(read_actor_manifest(source)[-1].pose_feature_path)
    original = feature.read_bytes()
    try:
        feature.write_bytes(original + b"modified after detected freeze")
        with pytest.raises(ValueError, match="Frozen artifact changed"):
            verify_receipt(workflow, "detected_pose", receipt_path(workflow, "detected_pose"))
    finally:
        feature.write_bytes(original)


def test_detector_weights_cannot_differ_from_selected_candidate_weights(workflow):
    from surveillance.detection.records import read_detections, write_detections

    path = Path(workflow["experiments"]["detected_pose"]["detections"])
    original = path.read_bytes()
    records = read_detections(path)
    record = records[0]
    changed = replace(
        record.result, metadata={**record.result.metadata, "checkpoint_sha256": "b" * 64}
    )
    try:
        write_detections([replace(record, result=changed), *records[1:]], path)
        with pytest.raises(ValueError, match="frozen validation detector filters"):
            freeze_experiment(workflow, "detected_pose")
    finally:
        path.write_bytes(original)


def test_artifact_changed_during_training_prevents_test_evaluation(tmp_path, monkeypatch):
    import surveillance.experiments.runner as runner

    protocol = synthetic_protocol(tmp_path / "fixture")
    protocol["actor"]["training"]["max_iterations"] = 1
    freeze_experiment(protocol, "pose_gt")
    feature = Path(read_actor_manifest(Path(protocol["dataset"]["manifest"]))[-1].pose_feature_path)
    original_train = runner.train

    def changed_after_training(*args, **kwargs):
        checkpoint = original_train(*args, **kwargs)
        feature.write_bytes(feature.read_bytes() + b"changed during training")
        return checkpoint

    monkeypatch.setattr(runner, "train", changed_after_training)
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        runner.run_experiment(protocol, "pose_gt", 0)
    assert not (Path(protocol["output_root"]) / "pose_gt" / "seed_0" / "metrics.json").exists()
