"""Robustness reports pair supported predictions and make missed coverage visible."""

import copy
import hashlib
import json

import numpy as np
import pytest
from test_detected_actor_inputs import result
from test_group_activity_inference import tiny_config, tiny_manifest

from surveillance.datasets.collective import read_actor_manifest
from surveillance.datasets.detected_actors import source_fingerprint
from surveillance.detection.matching import MatchResult
from surveillance.detection.records import DetectionRecord, detection_fingerprint, write_detections
from surveillance.evaluation.box_robustness import compare_predictions, evaluate_box_robustness
from surveillance.training.actor_transformer_trainer import train


def scenes():
    gt = [
        {
            "metadata": {"dataset": "fixture", "clip_id": str(i), "source_video_id": str(i)},
            "group_label": i,
            "group_prediction": i,
            "actors": [{"label": 0, "prediction": 0}, {"label": 1, "prediction": 0}],
        }
        for i in range(2)
    ]
    detected = copy.deepcopy(gt)
    detected[0].update(
        status="ok",
        actors=[
            {"label": None, "prediction": 1},
            {"label": 1, "prediction": 1},
        ],
    )
    detected[1].update(status="no_actors_detected", actors=[], group_prediction=None)
    matches = [
        MatchResult([1], [1], [0.8], [0], [0], 2, 2),
        MatchResult([], [], [], [0, 1], [], 2, 0),
    ]
    return gt, detected, matches


def test_paired_comparison_uses_same_actor_subset_and_counts_misses():
    report = compare_predictions(*scenes(), 2, 2)
    assert report["ground_truth_boxes"]["actor"]["accuracy"] == 0.5
    assert report["paired_ground_truth"]["actor_matched_subset"]["accuracy"] == 0
    assert report["detected_boxes"]["actor_matched_only"]["accuracy"] == 1
    assert report["delta"]["actor_accuracy_matched_subset"] == 1  # Not 1 - overall GT 0.5.
    coverage = report["detected_boxes"]["detection"]
    assert coverage["missed_gt_actors"] == 3 and coverage["unmatched_detections"] == 1
    assert coverage["recall"] == 0.25 and coverage["precision"] == 0.5
    assert report["detected_boxes"]["scene_coverage"] == 0.5
    assert report["detected_boxes"]["group"]["accuracy_all_scenes_abstentions_incorrect"] == 0.5
    assert report["delta"]["group_accuracy_all_scenes"] == -0.5
    assert report["delta"]["group_accuracy_supported_scenes"] == 0
    json.dumps(report, allow_nan=False)


def test_unmatched_detection_keeps_group_metrics_but_actor_metrics_are_null():
    gt, detected, _ = scenes()
    detected[0]["actors"] = [{"label": None, "prediction": 1}]
    matches = [MatchResult([], [], [], [0, 1], [0], 2, 1)]
    report = compare_predictions(gt[:1], detected[:1], matches, 2, 2)
    assert report["detected_boxes"]["group"]["count"] == 1
    assert report["detected_boxes"]["group"]["accuracy"] == 1
    actor = report["detected_boxes"]["actor_matched_only"]
    assert actor["count"] == 0 and actor["accuracy"] is None and actor["macro_f1"] is None
    assert actor["confusion_matrix"] is None
    assert report["delta"]["actor_accuracy_matched_subset"] is None
    assert report["detected_boxes"]["detection"]["recall"] == 0
    json.dumps(report, allow_nan=False)


def test_all_empty_predictions_have_undefined_supported_metrics_and_zero_all_accuracy():
    gt, detected, matches = scenes()
    report = compare_predictions(gt[1:], detected[1:], matches[1:], 2, 2)
    group = report["detected_boxes"]["group"]
    assert group["count"] == 0 and group["accuracy"] is None
    assert group["accuracy_all_scenes_abstentions_incorrect"] == 0
    assert report["detected_boxes"]["scene_coverage"] == 0
    assert report["delta"]["group_macro_f1_supported_scenes"] is None


@pytest.mark.parametrize("change", ["identity", "group_label", "actor_label"])
def test_comparison_rejects_unpaired_scenes_or_transferred_targets(change):
    gt, detected, matches = scenes()
    if change == "identity":
        detected[0]["metadata"]["clip_id"] = "wrong"
    elif change == "group_label":
        detected[0]["group_label"] = 1
    else:
        detected[0]["actors"][1]["label"] = 0
    with pytest.raises(ValueError):
        compare_predictions(gt, detected, matches, 2, 2)


def test_public_evaluator_uses_one_checkpoint_for_gt_and_detected_inputs(tmp_path):
    manifest = tiny_manifest(tmp_path)
    checkpoint = train(tiny_config(1), manifest, tmp_path / "trained")
    row = next(row for row in read_actor_manifest(manifest) if row.split == "test")
    detection = result([[20, 10, 100, 90], [140, 10, 180, 90]])
    features = tmp_path / "detected_features.npy"
    np.save(features, np.arange(16, dtype=np.float32).reshape(2, 8))
    metadata = {
        "detection_fingerprint": detection_fingerprint(detection),
        "source_fingerprint": source_fingerprint(row, manifest),
        "feature_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
    }
    detections = tmp_path / "detected.jsonl"
    write_detections(
        [
            DetectionRecord(
                row.dataset,
                row.video_id,
                row.source_video_id,
                row.clip_id,
                5,
                detection,
                pose_feature_path=features.name,
                feature_metadata={"pose": metadata},
            )
        ],
        detections,
    )
    report = evaluate_box_robustness(
        checkpoint,
        manifest,
        detections,
        device="cpu",
        return_attention=True,
    )
    assert report["checkpoint"] == str(checkpoint)
    assert report["ground_truth_boxes"]["actor"]["count"] == 3
    assert report["detected_boxes"]["actor_matched_only"]["count"] == 1
    assert report["detected_boxes"]["detection"]["missed_gt_actors"] == 2
    assert report["detected_boxes"]["detection"]["unmatched_detections"] == 1
    assert report["ground_truth_boxes"]["group"]["count"] == 1
    assert report["detected_boxes"]["group"]["count"] == 1
    assert len(report["ground_truth_predictions"][0]["actors"]) == 3
    assert len(report["detected_predictions"][0]["actors"]) == 2
    json.dumps(report, allow_nan=False)
