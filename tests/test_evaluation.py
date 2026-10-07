from dataclasses import replace

import numpy as np
import pytest
import torch

from surveillance.datasets.common import Record, write_manifest
from surveillance.datasets.features import load_features
from surveillance.evaluation.anomaly_metrics import binary_metrics, frame_labels, project_scores
from surveillance.evaluation.evaluate import evaluate
from surveillance.models.sultani.model import SultaniScorer
from surveillance.visualization.timeline import save_timeline


def test_metrics_and_unknown_roc():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2])
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert binary_metrics([0, 0], [0.1, 0.2])["roc_auc"] is None
    np.testing.assert_array_equal(frame_labels(6, 2.0, [[0.5, 1.5]]), [0, 1, 1, 0, 0, 0])
    assert len(project_scores([0, 1], 1)) == 1
    assert np.all(np.diff(project_scores([0, 1], 10, "interpolate")) >= 0)


def test_feature_validation(tmp_path):
    path = tmp_path / "bag.npy"
    np.save(path, np.zeros((32, 4), dtype="float32"))
    assert load_features(path, 32, 4).shape == (32, 4)
    with pytest.raises(ValueError, match="expected finite"):
        load_features(path, 32, 8)
    np.save(path, np.full((32, 4), np.nan, dtype="float32"))
    with pytest.raises(ValueError, match="expected finite"):
        load_features(path, 32, 4)


def test_evaluation_does_not_invent_frame_annotations(tmp_path):
    feature = tmp_path / "bag.npy"
    np.save(feature, np.ones((32, 4), dtype="float32"))
    rows = [
        Record(
            "synthetic",
            "a",
            "a",
            "video.avi",
            "test",
            1,
            fps=2,
            num_frames=6,
            feature_path=str(feature),
        ),
        Record(
            "synthetic",
            "b",
            "b",
            "normal.avi",
            "test",
            0,
            fps=2,
            num_frames=6,
            feature_path=str(feature),
        ),
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(rows, manifest)
    model = SultaniScorer(feature_dim=4, hidden_dims=[3], dropout=0)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "config": {
                "feature_dim": 4,
                "num_segments": 32,
                "model": {"hidden_dims": [3], "dropout": 0},
            },
        },
        checkpoint,
    )
    assert evaluate(checkpoint, manifest)["count"] == 2
    evidence = evaluate(checkpoint, manifest, include_predictions=True)
    assert [r["video_id"] for r in evidence["predictions"]] == ["a", "b"]
    assert evidence["predictions"][0]["bag_label"] == 1
    assert len(evidence["predictions"][0]["segment_scores"]) == 32
    assert evidence["predictions"][0]["overall_score"] == max(
        evidence["predictions"][0]["segment_scores"]
    )
    with pytest.raises(ValueError, match="temporal annotations unavailable"):
        evaluate(checkpoint, manifest, mode="frame")
    write_manifest([replace(rows[0], temporal_annotations=[[0.5, 1.5]]), rows[1]], manifest)
    assert evaluate(checkpoint, manifest, mode="frame")["count"] == 12


def test_timeline_artifact(tmp_path):
    output = tmp_path / "timeline.png"
    save_timeline([(0.0, 1.0), (1.0, 2.0)], [0.1, 0.9], output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
