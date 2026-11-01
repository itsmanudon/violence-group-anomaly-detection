"""React transport keeps exact-example caches separate from genuine upload inference."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from surveillance.demo_api import create_api
from surveillance.demo_profile import load_professor_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(tmp_path):
    config = load_professor_profile(ROOT)
    config["outputs"] = str(tmp_path / "outputs")
    return create_api(config, ROOT, frontend=tmp_path / "unbuilt")


def wait_job(client, job_id):
    for _ in range(100):
        result = client.get("/api/jobs/" + job_id).json()
        if result["status"] in {"complete", "failed"}:
            return result
        time.sleep(0.01)
    pytest.fail("Background inference did not finish")


def test_api_status_exposes_real_asset_readiness_and_cases(app):
    with TestClient(app) as client:
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert len(data["examples"]) == 9
        assert any(e["kind"] == "limitation" for e in data["examples"])
        assert len(data["profile_sha256"]) == 64


def test_api_rejects_paths_and_external_origins(app):
    with TestClient(app) as client:
        assert client.get("/api/media/../../README.md").status_code == 404
        assert client.get("/api/media/not-registered").status_code == 404
        assert client.post("/api/analyze", json={}).status_code == 400
        assert client.post("/api/analyze", json={"upload_id": "/etc/passwd"}).status_code == 404
        assert client.post("/api/analyze", json={"example": "not-indexed"}).status_code == 404
        assert (
            client.post(
                "/api/analyze", json={}, headers={"Origin": "https://untrusted.example"}
            ).status_code
            == 403
        )


def test_upload_ignores_client_cache_request_and_reports_failure(app, monkeypatch, tmp_path):
    calls = []

    def analyze(video, example, use_cache):
        calls.append((video, example, use_cache))
        raise ValueError("frozen checkpoint unavailable")

    monkeypatch.setattr(app.state.service, "analyze", analyze)
    import cv2
    import numpy as np

    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    video_bytes = video.read_bytes()
    with TestClient(app) as client:
        upload = client.post("/api/upload", files={"file": ("clip.avi", video_bytes)}).json()
        response = client.post(
            "/api/analyze", json={"upload_id": upload["upload_id"], "use_cache": True}
        )
        assert response.status_code == 202
        job = wait_job(client, response.json()["job_id"])
        assert job == {"status": "failed", "error": "frozen checkpoint unavailable"}
        assert calls[0][1:] == ("Upload a video", False)
        assert client.get(upload["video_url"]).content == video_bytes


def test_upload_requires_video_extension_and_nonempty_bytes(app):
    with TestClient(app) as client:
        assert client.post("/api/upload", files={"file": ("secret.txt", b"x")}).status_code == 400
        assert client.post("/api/upload", files={"file": ("empty.mp4", b"")}).status_code == 400


def test_cached_example_serializes_genuine_service_result(app, monkeypatch, tmp_path):
    video = tmp_path / "example.mp4"
    video.write_bytes(b"unit fixture")
    calls = []

    def analyze(path, example, cached):
        calls.append((path, example, cached))
        return {
            "schema_version": 1,
            "execution_mode": "cached_example",
            "video_path": str(video),
            "behavior": {"windows": []},
        }

    monkeypatch.setattr(app.state.service, "analyze", analyze)
    with TestClient(app) as client:
        response = client.post("/api/analyze", json={"example": "Normal example"})
        result = wait_job(client, response.json()["job_id"])
        assert result["status"] == "complete"
        assert calls == [(None, "Normal example", True)]
        assert client.get(result["download_url"]).json()["execution_mode"] == "cached_example"
        assert client.get(result["video_url"]).content == b"unit fixture"


def test_api_serializes_model_execution_and_releases_lock_after_error(app, monkeypatch):
    from threading import Event

    started, release = Event(), Event()

    def analyze(*args):
        started.set()
        assert release.wait(3)
        raise ValueError("unit test error")

    monkeypatch.setattr(app.state.service, "analyze", analyze)
    with TestClient(app) as client:
        first = client.post("/api/analyze", json={"example": "Normal example"}).json()
        assert started.wait(3)
        try:
            assert (
                client.post("/api/analyze", json={"example": "Normal example"}).status_code == 409
            )
        finally:
            release.set()
        assert wait_job(client, first["job_id"])["status"] == "failed"
        next_job = client.post("/api/analyze", json={"example": "Normal example"})
        assert next_job.status_code == 202
        assert wait_job(client, next_job.json()["job_id"])["status"] == "failed"
