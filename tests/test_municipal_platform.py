"""Synthetic media exercises software contracts, never benchmark performance."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from surveillance.demo_profile import load_professor_profile
from surveillance.municipal.catalog import Catalog, digest
from surveillance.municipal.queue import JobQueue

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def catalog(tmp_path):
    return Catalog(tmp_path / "municipal")


def make_clip(catalog, index=0, **changes):
    path = catalog.import_root / f"clip-{index}.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for frame in range(3):
        writer.write(np.full((24, 32, 3), (index + frame) % 255, np.uint8))
    writer.release()
    return dict(
        clip_id=f"clip-{index}",
        source_video_id=f"recording-{index}",
        file_path=path.name,
        sha256=digest(path),
        source="Synthetic integration fixture",
        license="Test fixture only",
        population="synthetic",
        selection_method="Generated for software validation",
        **changes,
    )


def import_clips(catalog, specs, dry_run=False):
    return catalog.import_manifest(dict(schema_version=1, clips=specs), dry_run=dry_run)


def service(catalog):
    config = load_professor_profile(ROOT)
    instance = SimpleNamespace(config=config)
    queue = JobQueue(catalog, instance)

    def analyze(path, example, use_cache):
        assert use_cache is False
        return dict(
            execution_mode="on_demand",
            video_sha256=digest(path),
            **queue.profile,
            anomaly={"segment_scores": [0.8] * 32, "timestamps": [[0, 0.3]] * 32},
            behavior={
                "windows": [
                    {
                        "status": "no_actors",
                        "prediction": None,
                        "model_disagreement": False,
                        "actor_count": 0,
                    }
                ]
            },
            final_alert={"state": "generic_anomaly", "behavior": None},
            timings={"total_seconds": 0.1},
            policy={"anomaly_threshold": 0.5},
        )

    instance.analyze = analyze
    return instance


def test_100_clip_manifest_dry_run_dedup_and_migration(catalog):
    specs = [make_clip(catalog, index) for index in range(100)]
    assert len(import_clips(catalog, specs, True)["imported"]) == 100
    assert catalog.clips() == []
    assert len(import_clips(catalog, specs)["imported"]) == 100
    assert len(import_clips(catalog, specs)["duplicates"]) == 100
    reopened = Catalog(catalog.directory)
    assert len(reopened.clips()) == 100
    with reopened.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
    backup = catalog.backup()
    import sqlite3

    with sqlite3.connect(backup) as db:
        assert db.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 100


@pytest.mark.parametrize(
    "path", ["../clip.avi", "/etc/passwd", "nested/../../clip.avi", "..\\x.avi"]
)
def test_import_path_security(catalog, path):
    spec = make_clip(catalog)
    spec["file_path"] = path
    with pytest.raises(ValueError, match="path|directory"):
        import_clips(catalog, [spec])


def test_symlink_and_corrupt_video(catalog, tmp_path):
    spec = make_clip(catalog)
    outside = tmp_path / "outside.avi"
    outside.write_bytes(b"not video")
    link = catalog.import_root / "link.avi"
    link.symlink_to(outside)
    spec["file_path"] = link.name
    with pytest.raises(ValueError, match="escapes"):
        import_clips(catalog, [spec])
    spec["file_path"] = "corrupt.avi"
    (catalog.import_root / spec["file_path"]).write_bytes(b"not video")
    with pytest.raises(ValueError, match="metadata"):
        import_clips(catalog, [spec])
    spec["file_path"] = "missing.avi"
    with pytest.raises(ValueError, match="Missing"):
        import_clips(catalog, [spec])


def test_source_split_identity_and_unknown_labels(catalog):
    a, b = make_clip(catalog), make_clip(catalog, 1)
    a["assignment"] = "test"
    import_clips(catalog, [a])
    b.update(source_video_id=a["source_video_id"], assignment="train")
    with pytest.raises(ValueError, match="Source"):
        import_clips(catalog, [b])
    a["assignment"] = "train"
    with pytest.raises(ValueError, match="Duplicate"):
        import_clips(catalog, [a])
    assert catalog.clip(a["clip_id"])["spec"]["annotation_status"] == "unlabeled"
    with catalog.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM reference_versions").fetchone()[0] == 0


def test_queue_success_failure_cancellation_and_idempotency(catalog):
    specs = [make_clip(catalog, i) for i in range(3)]
    import_clips(catalog, specs)
    queue = JobQueue(catalog, service(catalog))
    job_id = queue.submit([s["clip_id"] for s in specs], "submission")
    assert queue.submit([s["clip_id"] for s in specs], "submission") == job_id
    with pytest.raises(ValueError, match="Idempotency"):
        queue.submit([specs[0]["clip_id"]], "submission")
    assert queue.process_one()
    # Deleting an input does not lose queue state or produce an invented prediction.
    (catalog.import_root / specs[1]["file_path"]).unlink()
    queue.process_one()
    queue.cancel(job_id)
    job = queue.job(job_id)
    assert [i["status"] for i in job["items"]] == ["completed", "failed", "cancelled"]
    assert job["progress"] == 1
    assert job["items"][0]["result"]["behavior"]["windows"][0]["status"] == "no_actors"
    assert queue.retry(job_id, "retry") != job_id


def test_restart_recovery_and_frozen_profile(catalog):
    spec = make_clip(catalog)
    import_clips(catalog, [spec])
    instance = service(catalog)
    queue = JobQueue(catalog, instance)
    job_id = queue.submit([spec["clip_id"]], "first")
    assert queue.claim()
    resumed = JobQueue(Catalog(catalog.directory), instance)
    resumed.recover()
    resumed.process_one()
    assert resumed.job(job_id)["status"] == "completed"
    changed = queue.submit([spec["clip_id"]], "changed")
    queue.profile["profile_sha256"] = "changed"
    queue.process_one()
    assert queue.job(changed)["status"] == "failed"


def test_bounded_retry_and_cache_rejection(catalog):
    spec = make_clip(catalog)
    import_clips(catalog, [spec])
    instance = service(catalog)
    good = instance.analyze
    calls = []

    def timeout(*args):
        calls.append(1)
        raise TimeoutError()

    instance.analyze = timeout
    queue = JobQueue(catalog, instance)
    job_id = queue.submit([spec["clip_id"]], "timeout")
    queue.process_one()
    queue.process_one()
    assert len(calls) == 2
    assert queue.job(job_id)["status"] == "failed"
    instance.analyze = lambda *a: {**good(*a), "execution_mode": "cached_example"}
    job_id = queue.submit([spec["clip_id"]], "cache")
    queue.process_one()
    assert queue.job(job_id)["status"] == "failed"


def test_review_and_reference_append_only(catalog):
    spec = make_clip(catalog)
    import_clips(catalog, [spec])
    assert (
        catalog.reference(spec["clip_id"], {"class_label": "Fighting"}, "curator", "verified", 0)
        == 1
    )
    with pytest.raises(ValueError, match="version changed"):
        catalog.reference(spec["clip_id"], {"class_label": "Normal"}, "curator", "edit", 0)
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="immutable"), catalog.connect() as db:
        db.execute("UPDATE reference_versions SET version=2")
    queue = JobQueue(catalog, service(catalog))
    job_id = queue.submit([spec["clip_id"]], "review")
    item = queue.job(job_id)["items"][0]["item_id"]
    with pytest.raises(ValueError, match="completed"):
        catalog.review(item, "correct", "", "official", [])
    queue.process_one()
    catalog.review(item, "uncertain", "No actors", "official", [[0, 0.2]])
    with catalog.connect() as db:
        assert json.loads(db.execute("SELECT payload FROM reference_versions").fetchone()[0]) == {
            "class_label": "Fighting"
        }
    with pytest.raises(ValueError, match="duration"):
        catalog.review(item, "incorrect", "", "official", [[0, 999]])


def test_metric_population_and_exports(catalog):
    from surveillance.municipal.reporting import csv_export, evaluation, export_rows

    specs = [make_clip(catalog, i) for i in range(4)]
    specs[0].update(
        population="evaluation",
        assignment="test",
        training_overlap="excluded",
        evaluation_eligibility="Synthetic test of eligibility declarations only",
    )
    specs[1].update(population="curated_demo", assignment="demo")
    specs[2].update(population="upload")
    specs[3].update(
        population="evaluation",
        assignment="test",
        training_overlap="excluded",
        evaluation_eligibility="Synthetic test of eligibility declarations only",
    )
    import_clips(catalog, specs)
    instance = service(catalog)
    good = instance.analyze

    def predict(*args):
        result = good(*args)
        if "clip-0" in str(args[0]):
            result["final_alert"]["behavior"] = "Fighting"
        return result

    instance.analyze = predict
    queue = JobQueue(catalog, instance)
    job = queue.submit([s["clip_id"] for s in specs], "metrics")
    for _ in specs:
        queue.process_one()
    for s in specs:
        catalog.reference(
            s["clip_id"],
            {
                "class_label": "Fighting",
                "violence": True,
                "events": [[0, 0.15]],
                "temporal_complete": True,
            },
            "curator",
            "fixture",
            0,
        )
    report = evaluation(catalog, queue.profile["profile_sha256"])
    assert report["eligible_inferred_clips"] == 2
    assert report["class_labeled"] == 2
    assert report["behavior"]["n"] == 1
    assert report["behavior"]["accuracy"] == 1
    assert report["class_coverage"] == 0.5
    assert report["class_abstentions"] == 1
    assert report["temporal"]["false_alert_fraction"] == pytest.approx(1.0)
    item = queue.job(job)["items"][0]["item_id"]
    catalog.review(item, "incorrect", "=HYPERLINK(bad)", "reviewer", [])
    payload = export_rows(catalog)
    assert "reference" not in payload["records"][0]
    assert "video_path" not in json.dumps(payload)
    assert "'=HYPERLINK" in csv_export(payload)
    assert export_rows(catalog, external=True)["records"] == []
    assert export_rows(catalog, external=True)["excluded_rights"] == 4
    assert export_rows(catalog, references=True)["records"][0]["reference"]["version"] == 1
    # A second run of the same clip is not a second metric observation.
    queue.submit([specs[0]["clip_id"]], "repeat")
    queue.process_one()
    assert evaluation(catalog, queue.profile["profile_sha256"])["behavior"]["n"] == 1


def test_evaluation_requires_predeclared_overlap_evidence(catalog):
    spec = make_clip(catalog)
    spec.update(population="evaluation", assignment="test")
    with pytest.raises(ValueError, match="exclusion evidence"):
        import_clips(catalog, [spec])


def test_blinded_api_upload_security_and_review(catalog, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from surveillance.demo_api import create_api

    spec = make_clip(catalog)
    import_clips(catalog, [spec])
    catalog.reference(spec["clip_id"], {"class_label": "Fighting"}, "curator", "fixture", 0)
    config = load_professor_profile(ROOT)
    config.update(outputs=str(tmp_path / "demo"), municipal_directory=str(catalog.directory))
    app = create_api(config, ROOT)
    monkeypatch.setattr(app.state.service, "analyze", service(catalog).analyze)
    with TestClient(app) as client:
        bad = client.post("/api/platform/uploads", files={"file": ("../../x.avi", b"bad")})
        assert bad.status_code == 400
        assert list(catalog.uploads.iterdir()) == []
        assert (
            client.post("/api/platform/uploads", files={"file": ("x.txt", b"a")}).status_code == 400
        )
        video = (catalog.import_root / spec["file_path"]).read_bytes()
        upload = client.post("/api/platform/uploads", files={"file": ("../../x.avi", video)})
        assert upload.status_code == 201
        assert upload.json()["clip_id"] == spec["clip_id"]
        assert list(catalog.uploads.iterdir()) == []  # dedup removes only managed copy
        response = client.post(
            "/api/platform/jobs",
            json={
                "clip_ids": [spec["clip_id"]],
                "idempotency_key": "api-job",
            },
        )
        assert response.status_code == 202
        import time

        job_id = response.json()["job_id"]
        for _ in range(100):
            job = client.get(f"/api/platform/jobs/{job_id}").json()
            if job["status"] == "completed":
                break
            time.sleep(0.02)
        assert job["status"] == "completed"
        item = job["items"][0]["item_id"]
        assert "reference" not in client.get(f"/api/platform/items/{item}").json()
        assert client.get(f"/api/platform/items/{item}/reference").status_code == 409
        review = client.post(
            f"/api/platform/items/{item}/reviews",
            json={
                "decision": "uncertain",
                "reviewer": "official",
                "notes": "Review fixture",
            },
        )
        assert review.status_code == 201
        assert client.get(f"/api/platform/items/{item}/reference").json()["version"] == 1
        assert "reference" not in client.get("/api/platform/export").json()["records"][0]
        assert "Fighting" in client.get("/api/platform/export?references=true").text
        assert client.get("/api/platform/export?format=csv").status_code == 200
        assert client.get("/api/platform/health").json()["worker_alive"]
        assert client.delete(f"/api/platform/clips/{spec['clip_id']}/upload").status_code == 400
        assert (
            client.post(
                "/api/platform/jobs", json={}, headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )
        assert (
            client.get("/api/platform/health", headers={"host": "evil.example"}).status_code == 400
        )


def test_owned_upload_retention_and_active_job_protection(catalog):
    spec = make_clip(catalog)
    path = catalog.uploads / "owned.avi"
    path.write_bytes((catalog.import_root / spec["file_path"]).read_bytes())
    clip_id = catalog.add_upload(path)
    queue = JobQueue(catalog, service(catalog))
    job_id = queue.submit([clip_id], "retention")
    with pytest.raises(ValueError, match="active jobs"):
        catalog.delete_upload(clip_id)
    queue.cancel(job_id)
    catalog.delete_upload(clip_id)
    assert not path.exists()
    assert catalog.clip(clip_id)["status"] == "deleted"
    with pytest.raises(ValueError, match="unavailable"):
        queue.submit([clip_id], "deleted")


def test_worker_process_ownership(catalog):
    queue = JobQueue(catalog, service(catalog))
    queue.start()
    other = JobQueue(catalog, service(catalog))
    try:
        with pytest.raises(RuntimeError, match="Another"):
            other.start()
    finally:
        queue.close()


def test_disagreement_persisted_and_no_synthetic_metrics(catalog):
    from surveillance.municipal.reporting import evaluation

    spec = make_clip(catalog)
    import_clips(catalog, [spec])
    instance = service(catalog)
    good = instance.analyze

    def predict(*args):
        result = good(*args)
        result["behavior"]["windows"][0].update(
            model_disagreement=True, status="available", actor_count=2
        )
        result["final_alert"]["behavior"] = "Normal"
        return result

    instance.analyze = predict
    queue = JobQueue(catalog, instance)
    job = queue.submit([spec["clip_id"]], "disagreement")
    queue.process_one()
    assert queue.job(job)["items"][0]["result"]["behavior"]["windows"][0]["model_disagreement"]
    assert evaluation(catalog, queue.profile["profile_sha256"])["behavior"]["accuracy"] is None


def test_schema_v1_upgrade_preserves_rows_and_frozen_results(tmp_path):
    import sqlite3

    from surveillance.municipal.catalog import SCHEMA

    directory = tmp_path / "old"
    directory.mkdir()
    with sqlite3.connect(directory / "catalog.sqlite3") as db:
        db.executescript(SCHEMA)
        db.execute(
            "INSERT INTO audit(action,subject,actor,detail,created_at) VALUES(?,?,?,?,?)",
            ("old_event", "catalog", "operator", "{}", "2026-01-01"),
        )
    catalog = Catalog(directory)
    with catalog.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert db.execute("SELECT action FROM audit").fetchone()[0] == "old_event"
    spec = make_clip(catalog)
    import_clips(catalog, [spec])
    queue = JobQueue(catalog, service(catalog))
    queue.submit([spec["clip_id"]], "immutable")
    queue.process_one()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"), catalog.connect() as db:
        db.execute("UPDATE items SET result='{}' WHERE status='completed'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"), catalog.connect() as db:
        db.execute("UPDATE clips SET assignment='train'")


def test_cancel_between_clips_and_serialized_workers(catalog):
    import time
    from threading import Event, Lock

    specs = [make_clip(catalog, i) for i in range(3)]
    import_clips(catalog, specs)
    instance = service(catalog)
    good = instance.analyze
    started, release = Event(), Event()
    guard, active, maximum = Lock(), [0], [0]

    def infer(*args):
        with guard:
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
        started.set()
        assert release.wait(5)
        result = good(*args)
        with guard:
            active[0] -= 1
        return result

    instance.analyze = infer
    queue = JobQueue(catalog, instance, concurrency=3)
    job_id = queue.submit([s["clip_id"] for s in specs], "cancel-active")
    queue.start()
    try:
        assert started.wait(5)
        queue.cancel(job_id)
        release.set()
        for _ in range(100):
            if queue.job(job_id)["status"] == "cancelled":
                break
            time.sleep(0.02)
        assert [i["status"] for i in queue.job(job_id)["items"]] == [
            "completed",
            "cancelled",
            "cancelled",
        ]
        assert maximum[0] == 1
    finally:
        release.set()
        queue.close()


def test_upload_size_limit_and_decode_failure_cleanup(catalog, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from surveillance.demo_api import create_api

    config = load_professor_profile(ROOT)
    config.update(
        outputs=str(tmp_path / "demo"),
        municipal_directory=str(catalog.directory),
        municipal_max_upload_bytes=8,
    )
    with TestClient(create_api(config, ROOT)) as client:
        assert (
            client.post("/api/platform/uploads", files={"file": ("x.avi", b"x" * 9)}).status_code
            == 413
        )
        assert (
            client.post("/api/platform/uploads", files={"file": ("x.avi", b"")}).status_code == 400
        )
        assert client.post("/api/upload", files={"file": ("x.avi", b"bad")}).status_code == 400
        assert list(catalog.uploads.iterdir()) == []


def test_preflight_missing_and_corrupt_assets_are_explicit(tmp_path, monkeypatch):
    from surveillance.municipal import __main__ as command

    config = load_professor_profile(ROOT)
    config.update(municipal_directory=str(tmp_path / "catalog"), municipal_import_root=None)
    config["sultani_checkpoint"] = str(tmp_path / "missing.pt")
    config["c3d_checkpoint"] = str(tmp_path / "corrupt.pt")
    Path(config["c3d_checkpoint"]).write_bytes(b"invalid checkpoint")
    report = command.preflight(config, load_models=False)
    assert report["status"] == "FAIL"
    assert report["model_readiness_verified"] is False
    checks = {c["name"]: c for c in report["checks"]}
    assert "Missing frozen asset" in checks["sultani_checkpoint"]["detail"]
    assert "substitution forbidden" in checks["c3d_checkpoint"]["detail"]


def test_failed_import_rolls_back_whole_manifest(catalog):
    specs = [make_clip(catalog, 0), make_clip(catalog, 1)]
    specs[1]["source_video_id"] = specs[0]["source_video_id"]
    specs[1]["assignment"] = "test"
    with pytest.raises(ValueError, match="Source recording"):
        import_clips(catalog, specs)
    assert catalog.clips() == []
