"""Professor path relocation, scientific lock, and truthful missing-asset checks."""

from pathlib import Path

import pytest
import yaml

from surveillance.demo_preflight import preflight
from surveillance.demo_profile import load_professor_profile, profile_hash

ROOT = Path(__file__).resolve().parents[1]


def test_professor_paths_preserve_scientific_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_MODELS", str(tmp_path))
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({"c3d_checkpoint": "$DEMO_MODELS/frozen.pt"}))
    baseline = load_professor_profile(ROOT)
    relocated = load_professor_profile(ROOT, local)
    assert relocated["c3d_checkpoint"] == str(tmp_path / "frozen.pt")
    assert profile_hash(baseline) == profile_hash(relocated)
    assert relocated["anomaly_threshold"] == 0.5
    assert relocated["professor_locked"]


@pytest.mark.parametrize("key", ["anomaly_threshold", "model_sha256", "device", "training"])
def test_local_profile_rejects_policy_edits(tmp_path, key):
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump({key: 0.1}))
    with pytest.raises(ValueError, match="paths only"):
        load_professor_profile(ROOT, local)


def test_preflight_missing_models_fails_without_substitution(tmp_path):
    config = load_professor_profile(ROOT)
    for key in config["model_sha256"]:
        config[key] = str(tmp_path / (key + ".pt"))
    config["outputs"] = str(tmp_path / "outputs")
    config["examples"] = str(tmp_path / "absent.json")
    report = preflight(config, ROOT)
    assert report["status"] == "FAIL"
    assert sum(c["status"] == "FAIL" for c in report["checks"]) >= 6
    assert next(c for c in report["checks"] if c["name"] == "model_archives")["status"] == "WARN"
