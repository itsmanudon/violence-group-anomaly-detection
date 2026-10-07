import json

import pytest

from surveillance.datasets.dcsass_audit import sha256
from surveillance.experiments import demo_cache
from surveillance.inference.provenance import ASSET_KEYS, pipeline_identity


def setup_example(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture video bytes")
    entries = [
        {
            "name": "fixture",
            "video": "video.mp4",
            "video_sha256": sha256(video),
            "result_cache": "result.json",
        }
    ]
    (tmp_path / "examples.json").write_text(json.dumps(entries))
    config = {
        "examples": "examples.json",
        "model_sha256": {key: "a" * 64 for key in ASSET_KEYS},
        "training_population": {"anomaly": "fixture", "behavior": "fixture"},
        "c3d_mean": [104, 117, 128],
        "c3d_channel_order": "rgb",
        "anomaly_threshold": 0.5,
        "max_behavior_windows": 3,
    }
    return config


def test_cache_generation_records_inference_once_and_resumes_offline(tmp_path, monkeypatch):
    config = setup_example(tmp_path)
    calls = []

    class Pipeline:
        def predict_video(self, path):
            calls.append(path)
            return {
                "schema_version": 1,
                "provenance": pipeline_identity(config),
                "timings": {"total_seconds": 1.25},
                "final_alert": {"state": "generic_anomaly"},
            }

    monkeypatch.setattr(demo_cache, "load_surveillance_pipeline", lambda *args: Pipeline())
    result = demo_cache.cache_examples(config, tmp_path)
    assert result["generated"] == 1
    assert len(calls) == 1
    cached = json.loads((tmp_path / "result.json").read_text())
    assert cached["result"]["execution_mode"] == "precomputed_on_demand"
    monkeypatch.setattr(
        demo_cache,
        "load_surveillance_pipeline",
        lambda *args: pytest.fail("Valid offline caches need no model loading"),
    )
    assert demo_cache.cache_examples(config, tmp_path)["reused"] == 1


def test_cache_generation_rejects_changed_video_before_loading_models(tmp_path, monkeypatch):
    config = setup_example(tmp_path)
    (tmp_path / "video.mp4").write_bytes(b"different bytes")
    monkeypatch.setattr(
        demo_cache,
        "load_surveillance_pipeline",
        lambda *args: pytest.fail("Changed video must be rejected first"),
    )
    with pytest.raises(ValueError, match="video"):
        demo_cache.cache_examples(config, tmp_path)
