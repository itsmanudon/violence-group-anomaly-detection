"""Versioned local catalog with owned media and append-only assessment records."""

import hashlib
import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import cv2

from surveillance.datasets.dcsass_audit import sha256
from surveillance.video.decode import probe_video

ASSIGNMENTS = {"train", "validation", "test", "demo", "upload"}
COHORTS = {"curated_demo", "untouched_evaluation", "arbitrary_upload", "synthetic_fixture"}
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DECISIONS = {"correct", "incorrect", "uncertain", "not_assessable"}


def now():
    return datetime.now(UTC).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def bounded_text(value, name, limit=4000):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ValueError(f"Invalid {name}")
    return value


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path is outside the authorized media directory")
    return path


def inspect_video(path, max_bytes):
    if path.suffix.lower() not in EXTENSIONS or not path.is_file():
        raise ValueError("A supported local video file is required")
    if not 0 < path.stat().st_size <= max_bytes:
        raise ValueError("Video is empty or exceeds the configured size limit")
    metadata = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Video has no decodable first frame")
        height, width = frame.shape[:2]
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    return {
        "duration": metadata.duration_sec,
        "fps": metadata.fps,
        "resolution": [width, height],
        "num_frames": metadata.num_frames,
        "codec_fourcc": codec,
        "browser_playback": "verify_in_browser; OpenCV decoding alone is insufficient",
    }


def intervals(value, duration):
    if not isinstance(value, list):
        raise ValueError("Intervals must be a list")
    result = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Expected [start_seconds, end_seconds]")
        a, b = pair
        if (
            type(a) not in (int, float)
            or type(b) not in (int, float)
            or not math.isfinite(a + b)
            or not 0 <= a < b <= duration
        ):
            raise ValueError("Event interval lies outside the clip")
        result.append([a, b])
    return result


MIGRATION_1 = """
CREATE TABLE clips (
 clip_id TEXT PRIMARY KEY, source_video_id TEXT NOT NULL,
 sha256 TEXT UNIQUE NOT NULL, assignment TEXT NOT NULL,
 cohort TEXT NOT NULL, file_path TEXT NOT NULL, metadata TEXT NOT NULL,
 imported_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ready');
CREATE INDEX clips_source ON clips(source_video_id);
CREATE TABLE reference_versions (
 clip_id TEXT NOT NULL REFERENCES clips(clip_id), version INTEGER NOT NULL,
 payload TEXT NOT NULL, author TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(clip_id,version));
CREATE TABLE reviews (
 review_id TEXT PRIMARY KEY, clip_id TEXT NOT NULL REFERENCES clips(clip_id),
 item_id TEXT NOT NULL REFERENCES items(item_id), reviewer TEXT NOT NULL,
 decision TEXT NOT NULL, notes TEXT NOT NULL, intervals TEXT NOT NULL,
 blinded INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE jobs (
 job_id TEXT PRIMARY KEY, submission_key TEXT UNIQUE NOT NULL,
 request_hash TEXT NOT NULL, profile_hash TEXT NOT NULL, provenance TEXT NOT NULL,
 execution_mode TEXT NOT NULL, created_at TEXT NOT NULL,
 cancel_requested INTEGER NOT NULL DEFAULT 0);
CREATE TABLE items (
 item_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id),
 clip_id TEXT NOT NULL REFERENCES clips(clip_id), state TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
 started_at TEXT, finished_at TEXT, error TEXT, result TEXT,
 UNIQUE(job_id,clip_id));
CREATE INDEX items_state ON items(state,created_at);
CREATE TABLE audit (
 audit_id INTEGER PRIMARY KEY, action TEXT NOT NULL, subject TEXT NOT NULL,
 payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TRIGGER refs_no_update BEFORE UPDATE ON reference_versions
 BEGIN SELECT RAISE(ABORT,'References are append-only'); END;
CREATE TRIGGER refs_no_delete BEFORE DELETE ON reference_versions
 BEGIN SELECT RAISE(ABORT,'References are append-only'); END;
CREATE TRIGGER reviews_no_update BEFORE UPDATE ON reviews
 BEGIN SELECT RAISE(ABORT,'Reviews are append-only'); END;
CREATE TRIGGER reviews_no_delete BEFORE DELETE ON reviews
 BEGIN SELECT RAISE(ABORT,'Reviews are append-only'); END;
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit
 BEGIN SELECT RAISE(ABORT,'Audit is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit
 BEGIN SELECT RAISE(ABORT,'Audit is append-only'); END;
PRAGMA user_version=1;
"""

MIGRATION_2 = """
CREATE TABLE source_recordings (
 source_video_id TEXT PRIMARY KEY, assignment TEXT NOT NULL, cohort TEXT NOT NULL);
INSERT INTO source_recordings
 SELECT DISTINCT source_video_id,assignment,cohort FROM clips;
CREATE TABLE clip_sources (
 clip_id TEXT NOT NULL REFERENCES clips(clip_id),
 source_video_id TEXT NOT NULL REFERENCES source_recordings(source_video_id),
 PRIMARY KEY(clip_id,source_video_id));
INSERT INTO clip_sources SELECT clip_id,source_video_id FROM clips;
PRAGMA user_version=2;
"""

MIGRATION_3 = """
ALTER TABLE jobs ADD COLUMN config_sha256 TEXT;
PRAGMA user_version=3;
"""


class Catalog:
    def __init__(self, directory, *, max_bytes=512 * 1024 * 1024):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.media = self.directory / "media"
        self.temporary = self.directory / "temporary"
        self.media.mkdir(exist_ok=True)
        self.temporary.mkdir(exist_ok=True)
        self.path = self.directory / "catalog.sqlite3"
        self.max_bytes = max_bytes
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 3:
                raise ValueError("Catalog is newer than this application; upgrade the app")
            if version == 0:
                self.migrate(db, MIGRATION_1)
            if version < 2:
                self.migrate(db, MIGRATION_2)
            if version < 3:
                self.migrate(db, MIGRATION_3)

    @staticmethod
    def migrate(db, script):
        # execute(), rather than executescript(), preserves the surrounding transaction.
        statement = ""
        for line in script.splitlines():
            statement += line + "\n"
            if sqlite3.complete_statement(statement):
                db.execute(statement)
                statement = ""

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def audit(self, db, action, subject, payload):
        db.execute(
            "INSERT INTO audit(action,subject,payload,created_at) VALUES(?,?,?,?)",
            (action, subject, encode(payload), now()),
        )

    def prepare(self, record, allowed_root):
        record = dict(record)
        path = inside(record["file_path"], allowed_root)
        metadata = inspect_video(path, self.max_bytes)
        digest = sha256(path)
        if record.get("sha256") and record["sha256"] != digest:
            raise ValueError("Manifest SHA256 differs from video bytes")
        assignment, cohort = record.get("assignment"), record.get("cohort")
        if assignment not in ASSIGNMENTS or cohort not in COHORTS:
            raise ValueError("Explicit assignment and cohort are required")
        if cohort == "untouched_evaluation" and assignment != "test":
            raise ValueError("Untouched evaluation clips require test assignment")
        if cohort == "arbitrary_upload" and assignment != "upload":
            raise ValueError("Arbitrary footage requires upload assignment")
        if cohort == "curated_demo" and assignment != "demo":
            raise ValueError("Curated demonstration clips require demo assignment")
        for field in ("source_video_id", "source", "license", "selection_method"):
            if not bounded_text(record.get(field), field).strip():
                raise ValueError(f"Explicit {field} is required (use unknown where necessary)")
        for field in ("external_handover_permission", "training_permission"):
            if record.get(field) not in ("yes", "no", "unknown"):
                raise ValueError(f"Explicit yes/no/unknown required for {field}")
        geo = record.get("geography", {"location": None, "confidence": "unknown"})
        if not isinstance(geo, dict) or geo.get("confidence") not in (
            "verified",
            "reported",
            "inferred",
            "unknown",
        ):
            raise ValueError("Invalid geographic provenance")
        if geo["confidence"] in ("verified", "reported") and not geo.get("evidence"):
            raise ValueError("Geographic provenance requires evidence")
        if "reference" in record or "label" in record or "event_intervals" in record:
            raise ValueError("Import verified references separately with the audited reference API")
        clip_id = bounded_text(record.get("clip_id", uuid.uuid4().hex), "clip_id", 128)
        if not clip_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in clip_id
        ):
            raise ValueError("clip_id must contain only letters, numbers, underscore or hyphen")
        saved = {
            **record,
            **metadata,
            "clip_id": clip_id,
            "sha256": digest,
            "geography": geo,
            "annotation_status": "unlabeled",
            "created_at": bounded_text(record.get("created_at", now()), "created_at", 128),
            "notes": bounded_text(record.get("notes", ""), "notes"),
        }
        # Bound all untrusted manifest metadata before persisting or displaying it.
        if len(encode(saved)) > 32768:
            raise ValueError("Clip metadata exceeds 32 KiB")
        return path, saved

    def import_clip(self, record, allowed_root, *, dry_run=False):
        path, saved = self.prepare(record, allowed_root)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM clips WHERE sha256=?", (saved["sha256"],)
            ).fetchone()
            group = db.execute(
                "SELECT assignment,cohort FROM source_recordings WHERE source_video_id=?",
                (saved["source_video_id"],),
            ).fetchall()
            if any(
                r["assignment"] != saved["assignment"] or r["cohort"] != saved["cohort"]
                for r in group
            ):
                raise ValueError("Source recording cannot cross protected assignments or cohorts")
            if existing:
                if (existing["assignment"], existing["cohort"]) != (
                    saved["assignment"],
                    saved["cohort"],
                ):
                    raise ValueError("Duplicate bytes cannot cross assignments or cohorts")
                original = json.loads(existing["metadata"])
                if any(
                    original[field] != saved[field]
                    for field in ("license", "external_handover_permission", "training_permission")
                ):
                    raise ValueError("Duplicate bytes have conflicting rights or permissions")
                if not dry_run:
                    db.execute(
                        "INSERT OR IGNORE INTO source_recordings VALUES(?,?,?)",
                        (saved["source_video_id"], saved["assignment"], saved["cohort"]),
                    )
                    db.execute(
                        "INSERT OR IGNORE INTO clip_sources VALUES(?,?)",
                        (existing["clip_id"], saved["source_video_id"]),
                    )
                    self.audit(
                        db,
                        "duplicate_source_alias",
                        existing["clip_id"],
                        {"source_video_id": saved["source_video_id"]},
                    )
                return {"clip_id": existing["clip_id"], "deduplicated": True}
            if db.execute("SELECT 1 FROM clips WHERE clip_id=?", (saved["clip_id"],)).fetchone():
                raise ValueError("clip_id already exists with different bytes")
            if dry_run:
                return {"clip_id": saved["clip_id"], "dry_run": True, "metadata": saved}
            destination = self.media / (uuid.uuid4().hex + path.suffix.lower())
            try:
                copied = 0
                with path.open("rb") as source, destination.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > self.max_bytes:
                            raise ValueError("Video grew beyond the configured size limit")
                        target.write(chunk)
                if sha256(destination) != saved["sha256"]:
                    raise ValueError("Source changed during import")
                saved.update(inspect_video(destination, self.max_bytes))
                saved["file_path"] = str(destination)
                imported = now()
                db.execute(
                    "INSERT INTO clips VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        saved["clip_id"],
                        saved["source_video_id"],
                        saved["sha256"],
                        saved["assignment"],
                        saved["cohort"],
                        str(destination),
                        encode(saved),
                        imported,
                        "ready",
                    ),
                )
                db.execute(
                    "INSERT OR IGNORE INTO source_recordings VALUES(?,?,?)",
                    (saved["source_video_id"], saved["assignment"], saved["cohort"]),
                )
                db.execute(
                    "INSERT INTO clip_sources VALUES(?,?)",
                    (saved["clip_id"], saved["source_video_id"]),
                )
                self.audit(db, "clip_import", saved["clip_id"], {"sha256": saved["sha256"]})
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        return {"clip_id": saved["clip_id"], "deduplicated": False}

    def clips(self, search="", assignment=None):
        with self.connect() as db:
            rows = db.execute(
                "SELECT clips.*, (SELECT MAX(version) FROM reference_versions r "
                "WHERE r.clip_id=clips.clip_id) AS reference_version "
                "FROM clips ORDER BY imported_at,clip_id"
            ).fetchall()
        return [
            self.unpack(r)
            for r in rows
            if (not assignment or r["assignment"] == assignment)
            and search.casefold()
            in (r["clip_id"] + r["source_video_id"] + r["metadata"]).casefold()
        ]

    @staticmethod
    def unpack(row):
        value = {**json.loads(row["metadata"]), **dict(row)}
        value.pop("metadata")
        value["annotation_status"] = "verified" if value.get("reference_version") else "unlabeled"
        return value

    def clip(self, clip_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT clips.*, (SELECT MAX(version) FROM reference_versions r "
                "WHERE r.clip_id=clips.clip_id) AS reference_version FROM clips WHERE clip_id=?",
                (clip_id,),
            ).fetchone()
        if not row:
            raise ValueError("Unknown clip ID")
        return self.unpack(row)

    def media_path(self, clip_id):
        clip = self.clip(clip_id)
        path = inside(clip["file_path"], self.media)
        if not path.is_file() or sha256(path) != clip["sha256"]:
            raise ValueError("Catalog video is missing or its SHA256 changed")
        return path

    def reference(self, clip_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM reference_versions WHERE clip_id=? ORDER BY version DESC LIMIT 1",
                (clip_id,),
            ).fetchone()
        return {**dict(row), "payload": json.loads(row["payload"])} if row else None

    def add_reference(self, clip_id, payload, author, reason, expected_version):
        from surveillance.datasets.dcsass_protocol import CLASSES

        clip = self.clip(clip_id)
        if not author.strip() or not reason.strip():
            raise ValueError("Reference changes require author and change reason")
        bounded_text(author, "author", 200)
        bounded_text(reason, "reason")
        if payload.get("verified") is not True or not payload.get("evidence"):
            raise ValueError("Verified reference requires supplied evidence")
        if payload.get("behavior") not in (*CLASSES, None):
            raise ValueError("Reference behavior is outside the frozen class vocabulary")
        if payload.get("violence") is not None and type(payload["violence"]) is not bool:
            raise ValueError("violence must be boolean or unknown/null")
        if "event_intervals" in payload:
            intervals(payload["event_intervals"], clip["duration"])
            if payload.get("temporal_complete") is not True:
                raise ValueError("Temporal metrics require complete event annotations")
        if len(encode(payload)) > 32768:
            raise ValueError("Reference is too large")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute(
                "SELECT COALESCE(MAX(version),0) FROM reference_versions WHERE clip_id=?",
                (clip_id,),
            ).fetchone()[0]
            if expected_version != version:
                raise ValueError("Reference version conflict; read current version first")
            db.execute(
                "INSERT INTO reference_versions VALUES(?,?,?,?,?,?)",
                (clip_id, version + 1, encode(payload), author, reason, now()),
            )
            self.audit(db, "reference_version", clip_id, {"version": version + 1, "reason": reason})
        return self.reference(clip_id)

    def review(self, item_id, reviewer, decision, notes="", event_intervals=None, *, blinded=True):
        if decision not in DECISIONS or not bounded_text(reviewer, "reviewer", 200).strip():
            raise ValueError("Reviewer and valid review decision are required")
        bounded_text(notes, "notes")
        with self.connect() as db:
            row = db.execute(
                "SELECT clip_id,state FROM items WHERE item_id=?", (item_id,)
            ).fetchone()
            if not row or row["state"] != "completed":
                raise ValueError("Review requires a completed prediction item")
            clip = self.clip(row["clip_id"])
            events = intervals(event_intervals or [], clip["duration"])
            review_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    review_id,
                    row["clip_id"],
                    item_id,
                    reviewer,
                    decision,
                    notes,
                    encode(events),
                    int(blinded),
                    now(),
                ),
            )
            self.audit(db, "review", review_id, {"item_id": item_id, "blinded": blinded})
        return review_id

    def backup(self, destination):
        destination = inside(destination, self.directory / "exports")
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise ValueError("Backup destination exists; choose a new filename")
        with self.connect() as db, sqlite3.connect(destination) as target:
            db.backup(target)
        return destination

    def purge_temporary(self, retention_hours=24):
        if not math.isfinite(retention_hours) or retention_hours < 0:
            raise ValueError("Retention must be nonnegative hours")
        cutoff = datetime.now(UTC).timestamp() - retention_hours * 3600
        removed = 0
        for directory, subdirs, filenames in os.walk(self.temporary, followlinks=False):
            subdirs[:] = [d for d in subdirs if not (Path(directory) / d).is_symlink()]
            for filename in filenames:
                candidate = Path(directory) / filename
                if (
                    not candidate.is_symlink()
                    and candidate.is_file()
                    and candidate.stat().st_mtime < cutoff
                ):
                    inside(candidate, self.temporary).unlink()
                    removed += 1
        return removed
