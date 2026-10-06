import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "demo_api_validation", Path(__file__).resolve().parents[1] / "scripts/validate_demo_api.py"
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_new_default_receipts_preserve_historical_deployment_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    legacy = tmp_path / "runs/dcsass/demo-api-validation.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("historical DCSASS evidence")
    config = {"outputs": "outputs/demo/ucf_v1"}
    first = validator.validation_receipt_path(config, None)
    first.parent.mkdir(parents=True)
    first.write_text("first UCF evidence")
    second = validator.validation_receipt_path(config, None)
    assert first.parent == tmp_path / "outputs/demo/ucf_v1/api-validation"
    assert second != first
    assert legacy.read_text() == "historical DCSASS evidence"
    assert first.read_text() == "first UCF evidence"


def test_explicit_receipt_cannot_replace_previous_validation(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text("verified evidence")
    with pytest.raises(FileExistsError, match="Preserve"):
        validator.validation_receipt_path({"outputs": "unused"}, path)
    assert path.read_text() == "verified evidence"
