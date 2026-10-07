"""Rehearsal predeclares external provenance and rejects post-inference label drift."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rehearsal(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "rehearse_demo", ROOT / "scripts/rehearse_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"test-only video identity")
    return {
        "clips": [
            {
                "id": "external_01",
                "file": "clip.mp4",
                "expected": {"behavior": "Normal", "anomaly_expected": False},
                "source": {
                    "description": "Synthetic unit fixture",
                    "license": "test",
                    "url": "local",
                },
                "provenance": {
                    key: False
                    for key in (
                        "used_for_training",
                        "used_for_validation",
                        "used_for_test",
                        "used_for_tuning",
                    )
                },
            }
        ]
    }


@pytest.mark.parametrize(
    "key", ["used_for_training", "used_for_validation", "used_for_test", "used_for_tuning"]
)
def test_rehearsal_rejects_research_population(rehearsal, tmp_path, key):
    data = manifest(tmp_path)
    data["clips"][0]["provenance"][key] = True
    with pytest.raises(ValueError, match="unused"):
        rehearsal.validate_manifest(data, tmp_path)


def test_rehearsal_saves_expectations_before_first_call_and_rejects_drift(
    rehearsal, tmp_path, monkeypatch
):
    path = tmp_path / "manifest.yaml"
    data = manifest(tmp_path)
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(rehearsal, "ROOT", tmp_path)
    monkeypatch.setattr(rehearsal, "load_professor_profile", lambda *args: {})
    monkeypatch.setattr(rehearsal, "profile_hash", lambda _: "frozen-unit-policy")
    calls = []

    def analyze(*args):
        declarations = list((tmp_path / "runs/demo_rehearsal").glob("*/predeclared.json"))
        assert len(declarations) == 1
        saved = json.loads(declarations[0].read_text())
        assert saved["manifest"]["clips"][0]["expected"] == data["clips"][0]["expected"]
        calls.append(args)
        return {
            "final_alert": {"state": "no_anomaly", "behavior": None},
            "timings": {"request_total_seconds": 0.01},
        }

    monkeypatch.setattr(rehearsal, "DemoService", lambda *args: SimpleNamespace(analyze=analyze))
    monkeypatch.setattr(sys, "argv", ["rehearse_demo", "--manifest", str(path)])
    rehearsal.main()
    assert len(calls) == 1
    data["clips"][0]["expected"]["behavior"] = "Fighting"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Expectations changed"):
        rehearsal.main()
    assert len(calls) == 1
