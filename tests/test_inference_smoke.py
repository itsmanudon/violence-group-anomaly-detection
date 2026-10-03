import numpy as np
import torch

from surveillance.datasets.common import Record, write_manifest
from surveillance.evaluation.anomaly_metrics import binary_metrics, project_scores
from surveillance.inference.anomaly_pipeline import AnomalyPipeline
from surveillance.models.sultani.model import SultaniScorer
from surveillance.training.sultani_trainer import load_checkpoint, train


def test_inference_and_projection():
    pipeline = AnomalyPipeline(SultaniScorer(feature_dim=4, hidden_dims=[3], dropout=0))
    result = pipeline.predict_features(
        np.ones((32, 4), dtype=np.float32), duration_sec=8, threshold=0
    )
    assert len(result.segment_scores) == len(result.timestamps) == 32
    assert result.suspicious_intervals == [(0.0, 8.0)]
    assert result.overall_score == max(result.segment_scores)
    projected = project_scores([0.0, 1.0], 5)
    np.testing.assert_array_equal(projected, [0, 0, 1, 1, 1])
    assert binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])["roc_auc"] == 1


def test_train_checkpoint_roundtrip(tmp_path):
    rows = []
    for i in range(8):
        feature = tmp_path / f"{i}.npy"
        np.save(feature, np.random.default_rng(i).normal(size=(32, 4)).astype("float32"))
        rows.append(
            Record(
                "synthetic",
                str(i),
                str(i),
                "unused.mp4",
                "train" if i < 4 else "val",
                i % 2,
                feature_path=str(feature),
            )
        )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(rows, manifest)
    config = {
        "seed": 7,
        "device": "cpu",
        "num_segments": 32,
        "feature_dim": 4,
        "model": {"hidden_dims": [3], "dropout": 0},
        "loss": {},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "optimizer": "adagrad"},
    }
    train(config, manifest, tmp_path / "run")
    model, saved = load_checkpoint(tmp_path / "run" / "best.pt")
    assert saved["epoch"] == 1
    x = torch.ones(1, 32, 4)
    restored, _ = load_checkpoint(tmp_path / "run" / "last.pt")
    torch.testing.assert_close(model(x), restored(x))
    config["training"]["epochs"] = 2
    train(config, manifest, tmp_path / "run", resume=tmp_path / "run" / "last.pt")
    resumed, resumed_saved = load_checkpoint(tmp_path / "run" / "last.pt")
    assert resumed_saved["epoch"] == 2
    train(config, manifest, tmp_path / "uninterrupted")
    uninterrupted, _ = load_checkpoint(tmp_path / "uninterrupted" / "last.pt")
    torch.testing.assert_close(resumed(x), uninterrupted(x), rtol=0, atol=0)
