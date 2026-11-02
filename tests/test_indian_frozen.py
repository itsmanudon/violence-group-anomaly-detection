import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_cascade_evaluation import record
from test_indian_split import split_fixture

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_split import freeze_split
from surveillance.evaluation.indian_frozen import replay_frozen_review, run_frozen_validation
from surveillance.inference.provenance import pipeline_identity


def frozen_fixture(tmp_path, monkeypatch):
    args = split_fixture(tmp_path)
    freeze_split(*args)
    config = yaml.safe_load(Path("configs/surveillance_demo_ucf_v1.yaml").read_text())
    for key in config["model_sha256"]:
        asset = tmp_path / f"{key}.pt"
        asset.write_bytes(b"synthetic weights placeholder; mocked inference only")
        config[key] = str(asset)
        config["model_sha256"][key] = sha256(asset)
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config))
    seen = []

    class FixturePipeline:
        behavior = SimpleNamespace(device="cpu")

        def predict_video(self, path):
            seen.append(path)
            result = record("fixture", 0, 0.8, "available", 4)["result"]
            result["provenance"] = pipeline_identity(config)
            return result

    monkeypatch.setattr(
        "surveillance.evaluation.indian_frozen.load_surveillance_pipeline",
        lambda *args: FixturePipeline(),
    )
    return args[-1], config_path, tmp_path / "baseline", seen


def test_unapproved_missing_split_blocks_before_loading_or_output(tmp_path, monkeypatch):
    def must_not_load(*args):
        raise AssertionError("Unapproved data must not load models")

    monkeypatch.setattr(
        "surveillance.evaluation.indian_frozen.load_surveillance_pipeline", must_not_load
    )
    with pytest.raises(FileNotFoundError):
        run_frozen_validation(
            tmp_path / "absent", tmp_path / "absent.yaml", tmp_path / "out", root=Path.cwd()
        )
    assert not (tmp_path / "out").exists()


def test_frozen_baseline_only_infers_val_and_replay_uses_identical_cached_results(
    tmp_path, monkeypatch
):
    split, config, output, seen = frozen_fixture(tmp_path, monkeypatch)
    report = run_frozen_validation(split, config, output, root=Path.cwd())
    assert report["count"] == len(seen) == 2
    receipt = json.loads((split / "split-receipt.json").read_text())
    assert {p.stem for p in seen} == {k for k, v in receipt["assignments"].items() if v == "val"}
    assert report["metrics"]["confusion_matrix"] == [[0, 1], [0, 1]]
    replay = replay_frozen_review(output, split, tmp_path / "replay")
    assert replay["identical_raw_binary_metrics_all_modes"] == report["metrics"]
    assert len(seen) == 2  # replay did not call the models
    assert replay["modes"]["coverage_consistency"]["count"] == 2
    with pytest.raises(FileExistsError):
        run_frozen_validation(split, config, output, root=Path.cwd())


@pytest.mark.parametrize("mutation", ["hash", "test", "duplicate"])
def test_replay_rejects_tampered_or_test_predictions(tmp_path, monkeypatch, mutation):
    split, config, output, _ = frozen_fixture(tmp_path, monkeypatch)
    run_frozen_validation(split, config, output, root=Path.cwd())
    path = output / "clips/000000.json"
    p = json.loads(path.read_text())
    p["split"] = "test"
    path.write_text(json.dumps(p))
    completion_path = output / "completion.json"
    completion = json.loads(completion_path.read_text())
    if mutation != "hash":
        completion["result_sha256"]["clips/000000.json"] = sha256(path)
    if mutation == "duplicate":
        p["split"] = "val"
        path.write_text(json.dumps(p))
        other = output / "clips/000001.json"
        other.write_text(json.dumps(p))
        completion["result_sha256"] = {
            "clips/000000.json": sha256(path),
            "clips/000001.json": sha256(other),
        }
    completion_path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="hash|membership"):
        replay_frozen_review(output, split, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()
