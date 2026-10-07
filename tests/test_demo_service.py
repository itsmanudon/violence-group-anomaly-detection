import json

import cv2
import numpy as np
import pytest

from surveillance.datasets.dcsass_audit import sha256
from surveillance.demo_service import DemoService
from surveillance.inference.provenance import ASSET_KEYS, pipeline_identity


def demo_config(examples):
    return {
        "examples": str(examples),
        "model_sha256": {key: "a" * 64 for key in ASSET_KEYS},
        "training_population": {"anomaly": "test fixture", "behavior": "test fixture"},
        "c3d_mean": [104, 117, 128],
        "c3d_channel_order": "rgb",
        "anomaly_threshold": 0.5,
        "max_behavior_windows": 3,
    }


@pytest.mark.parametrize("changed", ["video", "policy", "model", "preflight"])
def test_cached_example_preserves_abstention_and_rejects_changed_provenance(tmp_path, changed):
    video = tmp_path / "example.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    result = {
        "schema_version": 1,
        "anomaly": {"overall_score": 0.9},
        "behavior": {
            "windows": [{"status": "no_actors", "prediction": None, "probabilities": None}]
        },
        "final_alert": {"state": "generic_anomaly"},
    }
    config = demo_config(tmp_path / "examples.json")
    result["provenance"] = pipeline_identity(config)
    artifact = tmp_path / "result.json"
    artifact.write_text(json.dumps({"video_sha256": sha256(video), "result": result}))
    examples = tmp_path / "examples.json"
    examples.write_text(
        json.dumps(
            [
                {
                    "name": "limitation",
                    "video": str(video),
                    "result_cache": str(artifact),
                    "video_sha256": sha256(video),
                }
            ]
        )
    )
    service = DemoService(config, tmp_path)
    loaded = service.analyze(None, "limitation", use_cache=True)
    assert loaded["execution_mode"] == "cached_example"
    assert loaded["behavior"]["windows"][0]["prediction"] is None
    assert loaded["final_alert"]["state"] == "generic_anomaly"
    if changed == "video":
        video.write_bytes(b"changed")
    else:
        if changed == "policy":
            result["provenance"]["policy"]["anomaly_threshold"] = 0.9
        elif changed == "model":
            result["provenance"]["models"]["behavior_checkpoint"] = "b" * 64
        else:
            result["provenance"]["training_status"] = "preflight"
        artifact.write_text(json.dumps({"video_sha256": sha256(video), "result": result}))
    with pytest.raises(ValueError, match="video" if changed == "video" else "provenance"):
        service.analyze(None, "limitation", use_cache=True)


def test_no_video_returns_actionable_error_before_loading_models(tmp_path):
    service = DemoService({"examples": "missing.json"}, tmp_path)
    with pytest.raises(ValueError, match="Upload"):
        service.analyze(None, "Upload a video", use_cache=False)


def test_changed_bundled_video_is_rejected_before_live_decode_or_model_loading(
    tmp_path, monkeypatch
):
    video = tmp_path / "example.mp4"
    video.write_bytes(b"original example fixture")
    examples = tmp_path / "examples.json"
    examples.write_text(
        json.dumps([{"name": "Normal", "video": str(video), "video_sha256": sha256(video)}])
    )
    service = DemoService({"examples": str(examples)}, tmp_path)
    video.write_bytes(b"different example fixture")
    monkeypatch.setattr(
        "surveillance.demo_service.probe_video",
        lambda *args: pytest.fail("Frozen example identity must be checked before live decoding"),
    )
    with pytest.raises(ValueError, match="example video"):
        service.analyze(None, "Normal", use_cache=False)
