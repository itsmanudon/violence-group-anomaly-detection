import json

import numpy as np
import pytest
import torch

from surveillance.datasets.dcsass_audit import sha256
from surveillance.detection.person_detector import DetectionResult
from surveillance.experiments.dcsass_cache import DETECTOR_SHA256, I3D_SHA256, detection_payload


def test_feature_loader_rejects_changed_feature_bytes_and_excludes_no_actors(tmp_path):
    from surveillance.datasets.dcsass_behavior import load_behavior_sample

    row = {
        "clip_id": "clip",
        "source_video_id": "source",
        "sha256": "video",
        "num_frames": 30,
        "group_label": 3,
        "split": "train",
    }
    result = DetectionResult(
        torch.tensor([[0.0, 0.0, 16.0, 24.0]]),
        torch.tensor([0.9]),
        torch.tensor([1]),
        (24, 32),
        {"checkpoint_sha256": DETECTOR_SHA256, "filter_config": {"confidence_threshold": 0.7}},
    )
    payload = detection_payload(
        row,
        {"reference_frame": 15, "frame_indices": list(range(10, 20)), "image_size": [24, 32]},
        result,
    )
    (tmp_path / "detections").mkdir()
    (tmp_path / "detections/clip.json").write_text(json.dumps(payload))
    (tmp_path / "rgb").mkdir()
    feature = tmp_path / "rgb/clip.npy"
    np.save(feature, np.ones((1, 20800), dtype=np.float32))
    meta = {
        "clip_id": "clip",
        "source_video_id": "source",
        "video_sha256": "video",
        "reference_frame": 15,
        "frame_indices": list(range(10, 20)),
        "detection_fingerprint": payload["detection_fingerprint"],
        "detector_sha256": DETECTOR_SHA256,
        "i3d_sha256": I3D_SHA256,
        "shape": [1, 20800],
        "feature_sha256": sha256(feature),
    }
    feature.with_suffix(".json").write_text(json.dumps(meta))
    sample = load_behavior_sample(row, tmp_path)
    assert sample["group_label"] == 3
    assert "actor_labels" not in sample
    assert sample["actor_boxes"].tolist() == [[0.0, 0.0, 0.5, 1.0]]
    np.save(feature, np.zeros((1, 20800), dtype=np.float32))
    with pytest.raises(ValueError, match="feature"):
        load_behavior_sample(row, tmp_path)
    empty = DetectionResult(
        torch.empty(0, 4),
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        (24, 32),
        result.metadata,
    )
    payload = detection_payload(
        row,
        {"reference_frame": 15, "frame_indices": list(range(10, 20)), "image_size": [24, 32]},
        empty,
    )
    (tmp_path / "detections/clip.json").write_text(json.dumps(payload))
    assert load_behavior_sample(row, tmp_path) is None


def test_training_weights_use_only_given_training_targets():
    from surveillance.training.dcsass_behavior import training_class_weights

    weights, counts = training_class_weights([0, 0, 0, 1, 2, 3, 4, 5])
    assert counts == [3, 1, 1, 1, 1, 1]
    assert weights[0] == pytest.approx(8 / 18)
    assert weights[3] == pytest.approx(8 / 6)
    with pytest.raises(ValueError, match="training"):
        training_class_weights([0, 1, 2, 4, 5])


def test_multiclass_and_binary_views_keep_separate_correctness():
    from surveillance.training.dcsass_behavior import behavior_metrics

    records = [
        {"target": 0, "prediction": 0, "probabilities": [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]},
        {"target": 3, "prediction": 3, "probabilities": [0.2, 0.1, 0.1, 0.4, 0.1, 0.1]},
        {"target": 3, "prediction": 1, "probabilities": [0.3, 0.4, 0.1, 0.1, 0.05, 0.05]},
    ]
    report = behavior_metrics(records)
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["macro_f1"] == pytest.approx((1 + 2 / 3) / 6)
    assert report["balanced_accuracy"] == 0.75
    assert report["binary_anomaly"]["f1"] == 1.0
    assert report["binary_anomaly"]["threshold"] == 0.5
