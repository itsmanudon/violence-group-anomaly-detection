"""HTTP, frontend runtime and offline acceptance; no models or real footage required."""

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from test_municipal_platform import manifest, result, video

from surveillance.inference.provenance import pipeline_identity
from surveillance.municipal.api import create_api
from surveillance.municipal.service import MunicipalService, preflight

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def municipal(tmp_path):
    config = yaml.safe_load((ROOT / "configs/surveillance_demo.yaml").read_text())

    def runner(path, mode):
        return {**result(path, mode), "provenance": pipeline_identity(config)}

    service = MunicipalService(tmp_path / "app", config, ROOT, runner=runner)
    with TestClient(create_api(service, start_worker=False), base_url="http://127.0.0.1") as client:
        yield service, client


def test_http_upload_process_review_reference_export(municipal, tmp_path):
    service, client = municipal
    path = video(tmp_path / "fixture.avi")
    response = client.post(
        "/api/uploads?filename=fixture.avi",
        content=path.read_bytes(),
        headers={"X-Municipal-Request": "local-upload"},
    )
    assert response.status_code == 200, response.text
    clip = response.json()["clip_id"]
    assert client.get("/api/clips").json()[0]["assignment"] == "upload"
    assert "file_path" not in client.get("/api/clips").json()[0]
    submit = client.post("/api/jobs", json={"clip_ids": [clip], "idempotency_key": "test"})
    assert submit.status_code == 200
    service.queue.process_one()
    jobs = client.get("/api/jobs").json()
    item = jobs[0]["items"][0]["item_id"]
    assert client.get(f"/api/items/{item}").json()["result"]["execution_mode"] == "on_demand"
    payload = {"verified": True, "evidence": "synthetic fixture only", "behavior": "Fighting"}
    assert (
        client.post(
            f"/api/references/{clip}",
            json={
                "payload": payload,
                "author": "test",
                "reason": "supplied fixture annotation",
                "expected_version": 0,
            },
        ).status_code
        == 200
    )
    assert "reference" not in client.get(f"/api/items/{item}").json()
    review = {"item_id": item, "reviewer": "operator", "decision": "uncertain"}
    assert client.post("/api/reviews", json=review).status_code == 200
    assert client.post(f"/api/items/{item}/reveal-reference").json()["version"] == 1
    assert client.post("/api/reviews", json=review).status_code == 200
    with service.catalog.connect() as db:
        assert [r[0] for r in db.execute("SELECT blinded FROM reviews ORDER BY created_at")] == [
            1,
            0,
        ]
    exported = client.get("/api/exports/json").json()
    assert exported["records"][0]["reference"] is None
    assert exported["evaluation"]["population"]["unique_clips"] == 0
    assert client.get("/api/exports/json?include_references=true").json()["records"][0]["reference"]
    assert "clip_id" in client.get("/api/exports/csv").text
    report = client.get("/api/exports/html").text
    assert "<video" not in report and "<img" not in report
    assert client.get(f"/api/clips/{clip}/video").content == path.read_bytes()
    assert client.post("/api/backup").status_code == 200


@pytest.mark.parametrize("name", ["../evil.avi", "..\\evil.avi", "C:evil.mp4", "clip.py", ""])
def test_upload_filename_security(municipal, name):
    _, client = municipal
    assert (
        client.post(
            "/api/uploads",
            params={"filename": name},
            content=b"bad",
            headers={"X-Municipal-Request": "local-upload"},
        ).status_code
        == 400
    )


def test_http_boundaries_corruption_limits_unknowns(municipal):
    service, client = municipal
    assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 403
    assert (
        client.post("/api/preflight", headers={"Origin": "https://attacker.example"}).status_code
        == 403
    )
    assert client.post("/api/uploads?filename=clip.avi", content=b"bad").status_code == 400
    headers = {"X-Municipal-Request": "local-upload"}
    assert (
        client.post("/api/uploads?filename=clip.avi", content=b"bad", headers=headers).status_code
        == 400
    )
    service.catalog.max_bytes = 2
    assert (
        client.post("/api/uploads?filename=clip.avi", content=b"bad", headers=headers).status_code
        == 413
    )
    assert list(service.catalog.temporary.iterdir()) == []
    assert client.get("/api/items/missing").status_code == 400
    assert (
        client.post("/api/jobs", json={"clip_ids": [], "idempotency_key": "x"}).status_code == 422
    )
    assert (
        client.post(
            "/api/jobs", json={"clip_ids": ["x"], "idempotency_key": "x", "extra": "reject"}
        ).status_code
        == 422
    )


def test_manifest_dry_run_is_nonmutating_and_checks_cross_clip_leakage(municipal, tmp_path):
    service, _ = municipal
    a, b = video(tmp_path / "a.avi", 1), video(tmp_path / "b.avi", 2)
    doc = {"schema_version": 1, "clips": [manifest(a), manifest(b)]}
    assert service.import_manifest(doc, tmp_path)["count"] == 2
    assert service.catalog.clips() == []
    doc["clips"][1].update(source_video_id="a", assignment="train")
    with pytest.raises(ValueError, match="Source"):
        service.import_manifest(doc, tmp_path)
    assert service.catalog.clips() == []


def test_missing_asset_preflight_never_loads_or_downloads(tmp_path):
    config = yaml.safe_load((ROOT / "configs/surveillance_demo.yaml").read_text())
    report = preflight(config, tmp_path)
    assert report["ready"] is False
    assert set(report["assets"].values()) == {"missing"}


def test_installed_frontend_static_build_and_six_screens(municipal, monkeypatch):
    import gradio as gr

    from surveillance.municipal.console import create_console

    service, _ = municipal
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(service.catalog.temporary / "gradio"))
    ui = create_console(service)
    app = gr.mount_gradio_app(
        create_api(service, start_worker=False),
        ui,
        path="/",
        theme=ui.municipal_theme,
        css=ui.municipal_css,
        allowed_paths=[str(service.catalog.media)],
        footer_links=[],
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/")
        assert response.status_code == 200
        config = client.get("/config").json()
        tabs = [c["props"]["label"] for c in config["components"] if c["type"] == "tabitem"]
        assert tabs == [
            "Dashboard",
            "Video library",
            "Processing queue",
            "Analysis workspace",
            "Human review",
            "Results and exports",
        ]
        assert config["analytics_enabled"] is False
        assert "fonts.googleapis.com" not in json.dumps(config)
        import re

        assets = re.findall(r'(?:src|href)="(\.?/?assets/[^\"]+\.(?:js|css))"', response.text)
        assert assets, "Installed frontend must include built JS/CSS assets"
        for asset in assets:
            assert client.get("/" + asset.lstrip("./")).status_code == 200
        assert client.get("/api/health").status_code == 200


def test_analysis_and_timed_review_callbacks_use_existing_pipeline(
    municipal, tmp_path, monkeypatch
):
    from test_surveillance_pipeline import FixedAnomaly, FixedBehavior

    from surveillance.inference.surveillance_pipeline import SurveillancePipeline
    from surveillance.municipal.console import create_console

    service, _ = municipal
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(service.catalog.temporary / "gradio"))
    path = video(tmp_path / "callback.avi", 11)
    clip = service.catalog.import_clip(manifest(path), tmp_path)["clip_id"]
    pipeline = SurveillancePipeline(
        FixedAnomaly(0.9), FixedBehavior(0), provenance=service.queue.identity
    )
    service.queue.runner = lambda p, mode: {**pipeline.predict_video(p), "execution_mode": mode}
    service.queue.submit([clip], "callback")
    service.queue.process_one()
    item_id = service.queue.jobs()[0]["items"][0]["item_id"]
    ui = create_console(service)
    callbacks = {entry.fn.__name__: entry.fn for entry in ui.fns.values()}
    evidence = callbacks["inspect"](item_id)
    assert len(evidence) == 12
    assert "on_demand" in evidence[1]
    assert evidence[8] == item_id and evidence[9] is None
    assert evidence[11] == []
    receipt = callbacks["save_review"](
        item_id, "operator", "incorrect", "fixture review", [[0.1, 0.5]]
    )
    assert receipt["review_id"]
    with pytest.raises(ValueError):
        callbacks["save_review"](item_id, "operator", "incorrect", "fixture", "[[0.1,99]]")


def test_schema_migration_preserves_existing_import(municipal, tmp_path):
    from surveillance.municipal.catalog import Catalog

    service, _ = municipal
    path = video(tmp_path / "migration.avi", 14)
    clip = service.catalog.import_clip(manifest(path), tmp_path)["clip_id"]
    with service.catalog.connect() as db:
        db.execute("ALTER TABLE jobs DROP COLUMN config_sha256")
        db.execute("PRAGMA user_version=2")
    migrated = Catalog(service.catalog.directory)
    assert migrated.clip(clip)["sha256"] == service.catalog.clip(clip)["sha256"]
    with migrated.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
