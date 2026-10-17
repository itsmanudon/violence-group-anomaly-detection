"""Synthetic-only acceptance tests; these results are not model performance."""

import csv
import io
import json
import sqlite3

import cv2
import numpy as np
import pytest

from surveillance.municipal.catalog import Catalog
from surveillance.municipal.evaluation import export_csv, export_json, metrics
from surveillance.municipal.queue import JobQueue

IDENTITY = {"fixture": True, "models": {"fixture": "a" * 64}}


def video(path, number=0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (32, 24))
    for frame in range(8):
        pixels = np.random.default_rng(number * 8 + frame).integers(
            0, 255, (24, 32, 3), dtype=np.uint8
        )
        writer.write(pixels)
    writer.release()
    return path


def manifest(path, **changes):
    return {
        "file_path": str(path),
        "source_video_id": path.stem,
        "source": "synthetic generator",
        "license": "test fixture only",
        "external_handover_permission": "no",
        "training_permission": "no",
        "assignment": "test",
        "cohort": "synthetic_fixture",
        "selection_method": "deterministic synthetic test",
        **changes,
    }


def result(path, mode, label="Fighting"):
    return {
        "schema_version": 1,
        "provenance": IDENTITY,
        "execution_mode": mode,
        "video_metadata": {"fps": 8},
        "video_path": str(path),
        "anomaly": {"segment_scores": [0.8] * 32},
        "behavior": {"windows": []},
        "alerts": [{"timestamp_interval": [0.0, 0.5]}],
        "final_alert": {"behavior": label},
        "timings": {"total_seconds": 0.01},
        "policy": {},
    }


@pytest.fixture
def setup(tmp_path):
    catalog = Catalog(tmp_path / "app")
    source = video(tmp_path / "fixture.avi")
    clip = catalog.import_clip(manifest(source), tmp_path)["clip_id"]
    queue = JobQueue(catalog, IDENTITY, result)
    yield catalog, source, clip, queue
    queue.close()


def test_import_owned_metadata_dedup_and_hash(setup, tmp_path):
    catalog, source, clip_id, _ = setup
    clip = catalog.clip(clip_id)
    assert clip["annotation_status"] == "unlabeled"
    assert clip["resolution"] == [32, 24]
    assert clip["duration"] == 1
    assert catalog.media_path(clip_id) != source
    assert catalog.import_clip(manifest(source), tmp_path)["deduplicated"]
    with pytest.raises(ValueError, match="rights"):
        catalog.import_clip(manifest(source, external_handover_permission="yes"), tmp_path)
    with pytest.raises(ValueError, match="Duplicate|Source"):
        catalog.import_clip(manifest(source, assignment="train"), tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        catalog.import_clip(manifest(source, sha256="a" * 64), tmp_path)


def test_hundred_unique_fixture_clips(tmp_path):
    catalog = Catalog(tmp_path / "app")
    for index in range(100):
        path = video(tmp_path / f"fixture{index}.avi", index)
        catalog.import_clip(manifest(path, clip_id=f"fixture{index}"), tmp_path)
    assert len(catalog.clips()) == 100
    assert len(catalog.clips(search="fixture99")) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"assignment": "unknown"},
        {"external_handover_permission": None},
        {"label": "Normal"},
        {"geography": {"confidence": "verified", "location": "India"}},
        {"clip_id": "../escape"},
        {"cohort": "untouched_evaluation", "assignment": "demo"},
    ],
)
def test_invalid_manifests(setup, tmp_path, change):
    catalog, source, _, _ = setup
    with pytest.raises(ValueError):
        catalog.import_clip(manifest(source, **change), tmp_path)


def test_path_corrupt_missing_and_size(tmp_path):
    catalog = Catalog(tmp_path / "app", max_bytes=100)
    outside = tmp_path / "outside"
    outside.mkdir()
    path = video(outside / "fixture.avi")
    with pytest.raises(ValueError, match="outside"):
        catalog.import_clip(manifest(path), catalog.media)
    with pytest.raises(ValueError, match="size"):
        catalog.import_clip(manifest(path), tmp_path)
    catalog.max_bytes = 10000
    path.write_bytes(b"invalid video")
    with pytest.raises(ValueError):
        catalog.import_clip(manifest(path), tmp_path)
    with pytest.raises(ValueError):
        catalog.import_clip(manifest(tmp_path / "missing.mp4"), tmp_path)


def test_queue_success_idempotence_failure_cancel_retry(setup, tmp_path):
    catalog, _, clip, queue = setup
    job = queue.submit([clip, clip], "request")
    assert queue.submit([clip], "request") == job
    with pytest.raises(ValueError, match="different"):
        queue.submit([clip], "request", "cached_example")
    assert queue.process_one()
    assert queue.jobs()[0]["state"] == "completed"
    broken = catalog.import_clip(manifest(video(tmp_path / "second.avi", 2)), tmp_path)["clip_id"]
    catalog.media_path(broken).unlink()
    failed = queue.submit([broken], "broken")
    queue.process_one()
    assert next(j for j in queue.jobs() if j["job_id"] == failed)["state"] == "failed"
    queue.submit([clip], "cancel")
    cancel = queue.jobs()[0]["job_id"]
    queue.cancel(cancel)
    assert queue.jobs()[0]["state"] == "cancelled"
    assert queue.retry(cancel, "retry") != cancel


def test_restart_recovery_lease_and_profile(setup):
    catalog, _, clip, queue = setup
    job = queue.submit([clip], "recover")
    queue.acquire()
    other = JobQueue(catalog, IDENTITY, result)
    with pytest.raises(RuntimeError, match="owns"):
        other.acquire()
    with catalog.connect() as db:
        db.execute("UPDATE items SET state='running' WHERE job_id=?", (job,))
    queue.close()
    try:
        other.acquire()
        assert other.jobs()[0]["state"] == "queued"
        other.process_one()
        assert other.jobs()[0]["state"] == "completed"
    finally:
        other.close()
    changed = JobQueue(catalog, {"changed": True}, result)
    try:
        queue.submit([clip], "profile")
        changed.process_one()
        assert changed.jobs()[0]["state"] == "failed"
    finally:
        changed.close()


def test_transient_retries_bounded_and_execution_mode(setup):
    _, _, clip, queue = setup
    queue.runner = lambda *_: (_ for _ in ()).throw(TimeoutError())
    queue.submit([clip], "transient")
    queue.process_one()
    assert queue.jobs()[0]["state"] == "queued"
    queue.process_one()
    assert queue.jobs()[0]["state"] == "failed"
    assert queue.jobs()[0]["items"][0]["attempts"] == 2
    queue.runner = lambda path, mode: result(path, "cached_example")
    queue.submit([clip], "must-be-fresh")
    queue.process_one()
    assert queue.jobs()[0]["state"] == "failed"


def test_reference_review_and_metrics_integrity(setup):
    catalog, _, clip, queue = setup
    queue.submit([clip], "fixture", mode="synthetic_fixture")
    queue.process_one()
    item = queue.jobs()[0]["items"][0]["item_id"]
    catalog.review(item, "operator", "uncertain", "=untrusted note")
    assert metrics(catalog, queue.profile_hash)["population"]["unique_clips"] == 0
    before = metrics(
        catalog, queue.profile_hash, cohort="synthetic_fixture", mode="synthetic_fixture"
    )
    assert before["behavior_all_labeled_abstentions_incorrect"]["accuracy"] is None
    payload = {
        "verified": True,
        "evidence": "fixture generator",
        "behavior": "Fighting",
        "violence": True,
        "event_intervals": [[0.2, 0.4]],
        "temporal_complete": True,
    }
    catalog.add_reference(clip, payload, "curator", "initial fixture annotation", 0)
    assert catalog.clip(clip)["annotation_status"] == "verified"
    assert catalog.clips()[0]["reference_version"] == 1
    with pytest.raises(ValueError, match="version"):
        catalog.add_reference(clip, payload, "curator", "stale update", 0)
    for table in ("reference_versions", "reviews", "audit"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"), catalog.connect() as db:
            db.execute(f"DELETE FROM {table}")
    after = metrics(
        catalog, queue.profile_hash, cohort="synthetic_fixture", mode="synthetic_fixture"
    )
    assert after["behavior_all_labeled_abstentions_incorrect"]["accuracy"] == 1
    assert after["temporal_false_alerts"]["false_alert_seconds"] == pytest.approx(0.3)
    assert json.loads(export_json(catalog, queue.profile_hash))["records"][0]["reference"] is None
    row = list(csv.DictReader(io.StringIO(export_csv(catalog, include_references=True))))[0]
    assert json.loads(row["reference"])["version"] == 1
    assert "untrusted note" in row["review"]


def test_backup_dry_run_and_scoped_retention(setup, tmp_path):
    catalog, _, _, _ = setup
    new = video(tmp_path / "dry.avi", 1)
    count = len(catalog.clips())
    catalog.import_clip(manifest(new), tmp_path, dry_run=True)
    assert len(catalog.clips()) == count
    temp = catalog.temporary / "upload.bin"
    temp.write_bytes(b"temporary")
    assert catalog.purge_temporary(0) == 1
    backup = catalog.backup(catalog.directory / "exports" / "backup.sqlite3")
    assert backup.is_file()
    with pytest.raises(ValueError, match="outside"):
        catalog.backup(tmp_path / "escape.sqlite3")


def test_duplicate_source_alias_cannot_leak_into_training(setup, tmp_path):
    catalog, source, _, _ = setup
    catalog.import_clip(manifest(source, source_video_id="alternate-source"), tmp_path)
    second = video(tmp_path / "other.avi", 3)
    with pytest.raises(ValueError, match="Source"):
        catalog.import_clip(
            manifest(second, source_video_id="alternate-source", assignment="train"), tmp_path
        )


@pytest.mark.parametrize("no_actors", [True, False])
def test_queued_pipeline_preserves_no_actors_and_disagreement(setup, no_actors):
    from test_surveillance_pipeline import FixedAnomaly, FixedBehavior

    from surveillance.inference.surveillance_pipeline import SurveillancePipeline

    catalog, _, clip, queue = setup
    pipeline = SurveillancePipeline(
        FixedAnomaly(0.9), FixedBehavior(0, no_actors=no_actors), provenance=IDENTITY
    )
    queue.runner = lambda path, mode: {**pipeline.predict_video(path), "execution_mode": mode}
    queue.submit([clip], "pipeline")
    queue.process_one()
    with catalog.connect() as db:
        saved = json.loads(db.execute("SELECT result FROM items").fetchone()[0])
    assert len(saved["anomaly"]["segment_scores"]) == 32
    assert saved["final_alert"]["state"] == "generic_anomaly"
    assert saved["behavior"]["windows"][0]["model_disagreement"] is not no_actors
    assert saved["behavior"]["windows"][0]["predicted_behavior"] == (
        None if no_actors else "Normal"
    )


def test_multiclip_cancel_during_running_clip_and_recover_cancelled(setup, tmp_path):
    catalog, _, clip, queue = setup
    other = catalog.import_clip(manifest(video(tmp_path / "other.avi", 3)), tmp_path)["clip_id"]
    job = queue.submit([clip, other], "multi")

    def runner(path, mode):
        queue.cancel(job)
        return result(path, mode)

    queue.runner = runner
    queue.process_one()
    assert sorted(i["state"] for i in queue.jobs()[0]["items"]) == ["cancelled", "completed"]
    assert queue.process_one() is False
    assert queue.jobs()[0]["progress"] == 1


def test_metrics_exclude_cache_uploads_unlabeled_and_count_abstention(setup, tmp_path):
    catalog, _, _, queue = setup
    path = video(tmp_path / "evaluation.avi", 12)
    clip = catalog.import_clip(manifest(path, cohort="untouched_evaluation"), tmp_path)["clip_id"]
    catalog.add_reference(
        clip,
        {"verified": True, "evidence": "fixture", "behavior": "Normal"},
        "fixture",
        "initial",
        0,
    )
    queue.runner = lambda p, m: result(p, m, None)
    queue.submit([clip], "fresh")
    queue.process_one()
    queue.submit([clip], "cache", "cached_example")
    queue.process_one()
    data = metrics(catalog, queue.profile_hash)
    assert data["population"]["unique_clips"] == 1
    assert data["population"]["coverage"] == 0
    assert data["behavior_all_labeled_abstentions_incorrect"]["accuracy"] == 0
    assert data["behavior_covered_only"]["accuracy"] is None
    assert (
        metrics(catalog, queue.profile_hash, mode="cached_example")["population"]["unique_clips"]
        == 1
    )


def test_recursive_temporary_cleanup_and_newer_schema_rejection(setup):
    catalog, _, _, _ = setup
    nested = catalog.temporary / "gradio" / "generated"
    nested.mkdir(parents=True)
    (nested / "upload.tmp").write_bytes(b"fixture")
    assert catalog.purge_temporary(0) == 1
    with catalog.connect() as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError, match="newer"):
        Catalog(catalog.directory)
