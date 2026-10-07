"""Population/provenance gates and descriptive scene reports use tiny synthetic data."""

import copy
import json
import math

import pytest

from surveillance.experiments.environment import environment_metadata
from surveillance.experiments.reporting import (
    aggregate_experiments,
    aggregate_table_markdown,
    analyze_predictions,
    confidence_summary,
    confusion_summary,
    select_feature_mode,
    write_error_report,
)


def _run(seed=1):
    return {
        "schema_version": 1,
        "protocol_hash": "protocol",
        "experiment": "pose_only",
        "seed": seed,
        "evidence_kind": "synthetic",
        "dry_run": True,
        "population": {
            "hash": "population",
            "scene_ids": ["a", "b"],
            "scene_count": 2,
            "actor_count": 3,
            "box_source": "ground_truth",
            "split": "test",
            "actor_population": "all_gt",
        },
        "metrics": {
            "group": {
                "accuracy": 0.5,
                "macro_f1": None,
                "count": 2,
                "confusion_matrix": [[1, 0], [1, 0]],
            }
        },
        "config_hash": f"config-{seed}",
        "selected_checkpoint": "best.pt",
        "validation_metrics": {"group_accuracy": 0.5},
    }


def _write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_aggregation_sample_std_null_and_individual_arrays(tmp_path):
    left, right = _run(), _run(2)
    right["metrics"]["group"]["accuracy"] = 1.0
    paths = [_write(tmp_path, "left.json", left), _write(tmp_path, "right.json", right)]
    aggregate = aggregate_experiments(paths)
    accuracy = aggregate["metrics"]["metrics.group.accuracy"]
    assert accuracy["count"] == 2 and accuracy["mean"] == 0.75
    assert accuracy["std"] == pytest.approx(math.sqrt(0.125))
    assert accuracy["seed_values"] == [{"seed": 1, "value": 0.5}, {"seed": 2, "value": 1.0}]
    assert aggregate["metrics"]["metrics.group.macro_f1"]["mean"] is None
    assert "metrics.group.count" in aggregate["individual_metrics"]
    assert "metrics.group.confusion_matrix" in aggregate["individual_metrics"]
    assert aggregate["dry_run"] is True and aggregate["evidence_kind"] == "synthetic"
    markdown = aggregate_table_markdown([aggregate])
    assert "synthetic" in markdown and "True" in markdown and "population" in markdown
    assert aggregate_experiments(paths[:1])["metrics"]["metrics.group.accuracy"]["std"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_hash", "different"),
        ("experiment", "rgb_only"),
        ("evidence_kind", "real"),
        ("dry_run", False),
        ("seed", 1),
    ],
)
def test_incompatible_runs_and_duplicate_seed_rejected(tmp_path, field, value):
    left, right = _run(), _run(2)
    right[field] = value
    paths = [_write(tmp_path, "left.json", left), _write(tmp_path, "right.json", right)]
    with pytest.raises(ValueError):
        aggregate_experiments(paths)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hash", "changed-matched-subset"),
        ("scene_ids", ["a", "c"]),
        ("actor_count", 2),
        ("box_source", "detections"),
        ("actor_population", "matched_only"),
    ],
)
def test_population_differences_rejected_even_with_same_hash(tmp_path, field, value):
    left, right = _run(), _run(2)
    right["population"][field] = value
    with pytest.raises(ValueError, match="population"):
        aggregate_experiments(
            [_write(tmp_path, "left.json", left), _write(tmp_path, "right.json", right)]
        )


def test_nonfinite_and_metric_structure_rejected(tmp_path):
    left = _run()
    left["metrics"]["group"]["accuracy"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        aggregate_experiments([_write(tmp_path, "nan.json", left)])
    left, right = _run(), _run(2)
    del right["metrics"]["group"]["macro_f1"]
    with pytest.raises(ValueError, match="Metric paths"):
        aggregate_experiments(
            [_write(tmp_path, "left.json", left), _write(tmp_path, "right.json", right)]
        )


def _scene(scene_id="a", predicted=0):
    return {
        "metadata": {
            "dataset": "collective",
            "clip_id": scene_id,
            "source_video_id": "seq05",
            "frame_indices": [1] * 10,
        },
        "group_label": 0,
        "group_prediction": predicted,
        "group_probabilities": [0.9, 0.1] if predicted == 0 else [0.1, 0.9],
        "actors": [
            {"index": 0, "label": 0, "prediction": 0},
            {"index": 1, "label": 1, "prediction": 0},
        ],
    }


def _det(scene):
    det = copy.deepcopy(scene)
    det["status"] = "ok"
    det["group_prediction"] = 1
    det["group_probabilities"] = [0.1, 0.9]
    det["actors"] = det["actors"][:1]
    det["matching"] = {
        "gt_indices": [0],
        "detection_indices": [0],
        "ious": [0.8],
        "missed_gt": [1],
        "unmatched_detections": [],
        "gt_count": 2,
        "detection_count": 1,
    }
    return det


def test_error_records_confusion_and_confidence(tmp_path):
    gt = _scene()
    report = analyze_predictions(
        [gt], [_det(gt)], classes=["crossing", "walking"], low_coverage=0.6
    )
    row = report["scenes"][0]
    assert "GT_correct_detected_failed" in row["categories"]
    assert "detected_high_confidence_wrong" in row["categories"]
    assert "low_coverage" in row["categories"]
    assert row["detection_coverage"] == 0.5 and row["mean_matched_iou"] == 0.8
    assert row["matched_actor_predictions"][0]["label"] == 0
    assert report["detected"]["actor"]["count"] == 1
    assert report["detected"]["confidence"]["incorrect"]["mean"] == 0.9
    assert report["detected"]["confusion"]["confusions"][0]["true_class"] == "crossing"
    paths = write_error_report(report, tmp_path / "reports")
    assert json.loads((tmp_path / "reports/errors.json").read_text())["scene_count"] == 1
    assert "unknown" not in paths["json"]
    assert "unannotated people" in (tmp_path / "reports/errors.md").read_text()


def test_zero_detection_and_unknown_confidence_stay_null():
    gt = _scene()
    det = _det(gt)
    det.update(
        status="no_actors_detected", group_prediction=None, group_probabilities=None, actors=[]
    )
    det["matching"].update(
        gt_indices=[], detection_indices=[], ious=[], missed_gt=[0, 1], detection_count=0
    )
    report = analyze_predictions([gt], [det], classes=["crossing", "walking"])
    row = report["scenes"][0]
    assert row["detected_prediction"] is None and row["detected_confidence"] is None
    assert row["detected_status"] == "no_actors_detected"
    assert row["detection_coverage"] == 0
    assert report["detected"]["confidence"]["abstained"]["unknown_count"] == 1
    assert report["detected"]["group"]["accuracy"] is None


@pytest.mark.parametrize("mutation", ["identity", "label", "actor_label", "order"])
def test_paired_prediction_identity_and_labels_are_checked(mutation):
    gt = [_scene("a"), _scene("b")]
    det = [_det(scene) for scene in gt]
    if mutation == "identity":
        det[0]["metadata"]["source_video_id"] = "other"
    elif mutation == "label":
        det[0]["group_label"] = 1
    elif mutation == "actor_label":
        det[0]["actors"][0]["label"] = 1
    else:
        det.reverse()
    with pytest.raises(ValueError):
        analyze_predictions(gt, det, classes=["crossing", "walking"])


def test_both_fail_and_confusion_uses_actual_vocabulary():
    gt = _scene(predicted=1)
    report = analyze_predictions([gt], [_det(gt)], classes=["crossing", "walking"])
    assert report["category_counts"]["both_failed"] == 1
    assert confusion_summary([0, 1, 0], [1, 0, 1], ["a", "b"])["confusions"][0] == {
        "true_label": 0,
        "true_class": "a",
        "predicted_label": 1,
        "predicted_class": "b",
        "count": 2,
    }
    assert confidence_summary([None])["mean"] is None


def test_environment_allowlist_contains_no_private_identifiers():
    metadata = environment_metadata("config", 7, "cpu")
    assert metadata["seed"] == 7 and metadata["config_hash"] == "config"
    assert isinstance(metadata["torch"], str)
    assert metadata["torchvision"] is None or isinstance(metadata["torchvision"], str)
    assert set(metadata) == {
        "python",
        "torch",
        "torchvision",
        "cuda_available",
        "cuda_version",
        "cuda_device",
        "device",
        "os",
        "git_commit",
        "config_hash",
        "seed",
    }
    assert set(metadata["os"]) == {"system", "release"}


def _validation_run(mode, seed, f1=0.5, accuracy=0.5):
    run = _run(seed)
    run["experiment"] = mode
    run["feature_mode"] = mode
    run["validation_population"] = {**run["population"], "split": "val", "hash": "val"}
    run["validation_metrics"] = {"group": {"accuracy": accuracy, "macro_f1": f1}}
    return run


def test_feature_selection_uses_validation_only_and_states_partial_scope(tmp_path):
    paths = []
    for mode, validation_f1, test_accuracy in (("pose_only", 0.8, 0.1), ("rgb_only", 0.7, 1.0)):
        for seed in (1, 2):
            run = _validation_run(mode, seed, validation_f1)
            run["metrics"]["group"]["accuracy"] = test_accuracy
            paths.append(_write(tmp_path, f"{mode}-{seed}.json", run))
    receipt = select_feature_mode(paths)
    assert receipt["selected_experiment"] == "pose_only"
    assert receipt["selection_value"] == 0.8
    assert receipt["split"] == "val" and receipt["seeds"] == [1, 2]
    assert len(receipt["candidates"]) == 2 and "untested" in receipt["scope"]
    assert select_feature_mode(paths[:1])["selected_experiment"] == "pose_only"


def test_feature_selection_ties_are_deterministic(tmp_path):
    runs = [
        _validation_run("rgb_only", 1, accuracy=0.9),
        _validation_run("pose_only", 1, accuracy=0.9),
    ]
    paths = [_write(tmp_path, f"tie-{index}.json", run) for index, run in enumerate(runs)]
    assert select_feature_mode(paths)["selected_experiment"] == "pose_only"


@pytest.mark.parametrize("mutation", ["population", "seed_set", "evidence", "dry_run", "null"])
def test_feature_selection_rejects_incompatible_candidates(tmp_path, mutation):
    left = _validation_run("pose_only", 1)
    right = _validation_run("rgb_only", 1)
    if mutation == "population":
        right["validation_population"]["actor_count"] = 9
    elif mutation == "seed_set":
        right["seed"] = 2
    elif mutation == "evidence":
        right["evidence_kind"] = "real"
    elif mutation == "dry_run":
        right["dry_run"] = False
    else:
        right["validation_metrics"]["group"]["macro_f1"] = None
    with pytest.raises(ValueError):
        select_feature_mode(
            [_write(tmp_path, "left.json", left), _write(tmp_path, "right.json", right)]
        )
