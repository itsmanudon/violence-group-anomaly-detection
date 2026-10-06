import numpy as np
import pytest
import torch

from surveillance.datasets.common import Record, write_manifest
from surveillance.experiments.sultani_selection import freeze_sultani_selection
from surveillance.models.sultani.model import SultaniScorer


@pytest.mark.parametrize("problem", [None, "unfinished", "preflight", "changed_validation"])
def test_sultani_selection_verifies_completion_and_validation_without_reading_test(
    tmp_path, problem
):
    run = tmp_path / "run"
    run.mkdir()
    feature = tmp_path / "bag.npy"
    np.save(feature, np.ones((32, 4), dtype=np.float32))
    manifest = tmp_path / "features.jsonl"
    write_manifest(
        [
            Record("fixture", str(i), str(i), "unneeded.avi", "val", i, feature_path=str(feature))
            for i in [0, 1]
        ]
        + [
            Record(
                "fixture",
                "test",
                "test",
                "unneeded.avi",
                "test",
                1,
                feature_path=str(tmp_path / "must-not-open.npy"),
            )
        ],
        manifest,
    )
    config = {
        "feature_dim": 4,
        "num_segments": 32,
        "model": {"hidden_dims": [3], "dropout": 0},
        "training": {"epochs": 20},
    }
    model = SultaniScorer(feature_dim=4, hidden_dims=[3], dropout=0)
    saved = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "config": config,
        "selection_metric": "validation_bag_roc_auc",
        "best_value": 0.5,
        "epoch": 7,
    }
    if problem == "preflight":
        saved["dry_run"] = True
    elif problem == "changed_validation":
        saved["best_value"] = 0.7
    torch.save(saved, run / "best.pt")
    torch.save({**saved, "epoch": 19 if problem == "unfinished" else 20}, run / "last.pt")
    if problem:
        with pytest.raises(ValueError):
            freeze_sultani_selection(run, manifest, "fixture")
        assert not (run / "selection.json").exists()
    else:
        receipt = freeze_sultani_selection(run, manifest, "fixture")
        assert receipt["selection_frozen"] is True
        assert receipt["selected_epoch"] == 7
        assert receipt["validation"]["roc_auc"] == 0.5
        assert receipt["held_out_used_for_selection"] is False
        with pytest.raises(FileExistsError):
            freeze_sultani_selection(run, manifest, "fixture")
