"""Versioned local catalog, explicit provenance and media boundary validation."""

import hashlib
import json
import math
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict, Field

from surveillance.datasets.dcsass_protocol import CLASSES

EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
ASSIGNMENTS = {"train", "validation", "test", "demo", "unassigned"}
POPULATIONS = {"evaluation", "curated_demo", "upload", "synthetic"}


def now():
    return datetime.now(UTC).isoformat()


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def scoped_path(root, relative):
    root = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in str(relative):
        raise ValueError("Use a relative video path inside the configured import directory")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Video path escapes its configured directory")
    return resolved


def inspect_video(path, max_bytes=500 * 1024 * 1024):
    path = Path(path)
    if path.suffix.lower() not in EXTENSIONS or not path.is_file():
        raise ValueError("Missing video or unsupported extension")
    if not 0 < path.stat().st_size <= max_bytes:
        raise ValueError("Video must be nonempty and within the configured size limit")
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or expected <= 0:
            raise ValueError("Video metadata is invalid or the codec is unavailable")
        count, width, height = 0, 0, 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            count += 1
        if count != expected or not width or not height:
            raise ValueError("Video decoding is incomplete or corrupt")
        return dict(
            sha256=digest(path),
            duration=count / fps,
            fps=fps,
            resolution=[width, height],
            frames=count,
            playback_note="Browser codec support varies; original video is preserved",
        )
    finally:
        capture.release()


class ClipSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    clip_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    source_video_id: str = Field(min_length=1, max_length=200)
    file_path: str = Field(min_length=1, max_length=1000)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source: str = Field(min_length=1, max_length=2000)
    location: str | None = Field(default=None, max_length=500)
    location_confidence: str = "unknown"
    geographic_evidence: str = Field(default="", max_length=2000)
    license: str = Field(min_length=1, max_length=2000)
    training_permission: bool = False
    municipal_redistribution_permission: bool = False
    assignment: str = "unassigned"
    population: str = "curated_demo"
    selection_method: str = Field(min_length=1, max_length=2000)
    training_overlap: str = "unknown"
    evaluation_eligibility: str = Field(default="", max_length=2000)
    annotation_status: str = "unlabeled"
    notes: str = Field(default="", max_length=5000)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: int = 1
    clips: list[ClipSpec] = Field(max_length=10000)


SCHEMA = """
CREATE TABLE clips (
 clip_id TEXT PRIMARY KEY, source_video_id TEXT NOT NULL, file_path TEXT NOT NULL,
 sha256 TEXT NOT NULL UNIQUE, metadata TEXT NOT NULL, spec TEXT NOT NULL,
 assignment TEXT NOT NULL, population TEXT NOT NULL, imported_at TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'available', owned_upload INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX clips_source ON clips(source_video_id);
CREATE TABLE reference_versions (
 clip_id TEXT NOT NULL REFERENCES clips(clip_id), version INTEGER NOT NULL,
 payload TEXT NOT NULL, author TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
 PRIMARY KEY(clip_id, version)
);
CREATE TABLE jobs (
 job_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, request TEXT NOT NULL,
 profile TEXT NOT NULL, status TEXT NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE items (
 item_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id),
 clip_id TEXT NOT NULL REFERENCES clips(clip_id), status TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0, started_at TEXT, finished_at TEXT,
 error TEXT, result TEXT, UNIQUE(job_id, clip_id)
);
CREATE TABLE reviews (
 review_id TEXT PRIMARY KEY, item_id TEXT NOT NULL REFERENCES items(item_id),
 decision TEXT NOT NULL, notes TEXT NOT NULL, reviewer TEXT NOT NULL,
 intervals TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE audit (
 audit_id INTEGER PRIMARY KEY, action TEXT NOT NULL, subject TEXT NOT NULL,
 actor TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TRIGGER reference_no_update BEFORE UPDATE ON reference_versions
 BEGIN SELECT RAISE(ABORT, 'reference versions are immutable'); END;
CREATE TRIGGER reference_no_delete BEFORE DELETE ON reference_versions
 BEGIN SELECT RAISE(ABORT, 'reference versions are immutable'); END;
CREATE TRIGGER review_no_update BEFORE UPDATE ON reviews
 BEGIN SELECT RAISE(ABORT, 'reviews are append only'); END;
CREATE TRIGGER review_no_delete BEFORE DELETE ON reviews
 BEGIN SELECT RAISE(ABORT, 'reviews are append only'); END;
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit
 BEGIN SELECT RAISE(ABORT, 'audit is append only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit
 BEGIN SELECT RAISE(ABORT, 'audit is append only'); END;
PRAGMA user_version=1;
"""


MIGRATION_2 = """
CREATE TRIGGER completed_prediction_no_update BEFORE UPDATE OF result ON items
 WHEN OLD.status='completed'
 BEGIN SELECT RAISE(ABORT, 'completed prediction is immutable'); END;
CREATE TRIGGER completed_prediction_no_delete BEFORE DELETE ON items
 WHEN OLD.status='completed'
 BEGIN SELECT RAISE(ABORT, 'completed prediction is immutable'); END;
CREATE TRIGGER clip_identity_no_update
 BEFORE UPDATE OF clip_id,source_video_id,file_path,sha256,metadata,spec,
 assignment,population ON clips
 BEGIN SELECT RAISE(ABORT, 'catalog identity and membership are immutable'); END;
CREATE TRIGGER source_membership_insert BEFORE INSERT ON clips
 WHEN EXISTS(SELECT 1 FROM clips WHERE source_video_id=NEW.source_video_id
 AND (assignment!=NEW.assignment OR ((population='evaluation')!=(NEW.population='evaluation'))))
 BEGIN SELECT RAISE(ABORT, 'source assignment conflict'); END;
PRAGMA user_version=2;
"""


class Catalog:
    def __init__(self, directory, import_root=None, max_bytes=500 * 1024 * 1024):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.import_root = Path(import_root or self.directory / "imports").resolve()
        self.import_root.mkdir(parents=True, exist_ok=True)
        self.uploads = self.directory / "uploads"
        self.uploads.mkdir(exist_ok=True)
        self.max_bytes = max_bytes
        self.legacy_guard = threading.RLock()
        self.legacy_active = set()
        self.path = self.directory / "catalog.sqlite3"
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                db.executescript("BEGIN IMMEDIATE;" + SCHEMA + "COMMIT;")
                version = 1
            if version == 1:
                db.executescript("BEGIN IMMEDIATE;" + MIGRATION_2 + "COMMIT;")
            elif version != 2:
                raise ValueError(
                    "Unsupported catalog schema; preserve database and use matching app"
                )

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

    def audit(self, db, action, subject, actor, detail):
        db.execute(
            "INSERT INTO audit(action,subject,actor,detail,created_at) VALUES(?,?,?,?,?)",
            (action, subject, actor, encode(detail), now()),
        )

    def import_manifest(self, payload, *, dry_run=False):
        manifest = Manifest.model_validate(payload)
        if manifest.schema_version != 1:
            raise ValueError("Unsupported manifest version")
        entries = []
        for spec in manifest.clips:
            if spec.assignment not in ASSIGNMENTS or spec.population not in POPULATIONS:
                raise ValueError("Invalid assignment or population")
            if spec.annotation_status not in {"unlabeled", "pending", "verified"}:
                raise ValueError("Invalid annotation status")
            if spec.annotation_status == "verified":
                raise ValueError(
                    "Import references separately through audited reference versioning"
                )
            if spec.location_confidence not in {"unknown", "inferred_appearance", "documented"}:
                raise ValueError("Invalid geographic confidence")
            if spec.location_confidence == "documented" and not spec.geographic_evidence.strip():
                raise ValueError("Documented geography requires evidence")
            if spec.population == "evaluation" and spec.assignment != "test":
                raise ValueError("Protected evaluation clips must be assigned test")
            if spec.training_overlap not in {"unknown", "excluded", "known_overlap"}:
                raise ValueError("Invalid training overlap declaration")
            if spec.population == "evaluation" and (
                spec.training_overlap != "excluded" or not spec.evaluation_eligibility.strip()
            ):
                raise ValueError(
                    "Evaluation requires source-safe exclusion evidence before inference"
                )
            path = scoped_path(self.import_root, spec.file_path)
            metadata = inspect_video(path, self.max_bytes)
            if metadata["sha256"] != spec.sha256:
                raise ValueError("Manifest video SHA256 mismatch")
            entries.append((spec, path, metadata))
        imported, duplicates = [], []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for spec, path, metadata in entries:
                # A duplicate cannot silently alter provenance, population or split membership.
                previous = db.execute(
                    "SELECT * FROM clips WHERE sha256=? OR clip_id=?",
                    (spec.sha256, spec.clip_id),
                ).fetchone()
                if previous:
                    old = json.loads(previous["spec"])
                    if any(
                        old.get(k, ClipSpec.model_fields[k].default) != spec.model_dump()[k]
                        for k in (
                            "sha256",
                            "source_video_id",
                            "assignment",
                            "population",
                            "source",
                            "license",
                            "training_permission",
                            "municipal_redistribution_permission",
                            "training_overlap",
                            "evaluation_eligibility",
                        )
                    ):
                        raise ValueError("Duplicate conflicts with protected identity/membership")
                    duplicates.append(previous["clip_id"])
                    continue
                source = db.execute(
                    "SELECT assignment,population FROM clips WHERE source_video_id=?",
                    (spec.source_video_id,),
                ).fetchall()
                if any(
                    r["assignment"] != spec.assignment
                    or (r["population"] == "evaluation") != (spec.population == "evaluation")
                    for r in source
                ):
                    raise ValueError("Source recording crosses protected assignment/population")
                db.execute(
                    "INSERT INTO clips VALUES(?,?,?,?,?,?,?,?,?,?,0)",
                    (
                        spec.clip_id,
                        spec.source_video_id,
                        str(path),
                        spec.sha256,
                        encode(metadata),
                        encode(spec.model_dump()),
                        spec.assignment,
                        spec.population,
                        now(),
                        "available",
                    ),
                )
                imported.append(spec.clip_id)
            if dry_run:
                db.rollback()
            else:
                self.audit(db, "manifest_import", "catalog", "local_importer", imported)
        return dict(imported=imported, duplicates=duplicates, dry_run=dry_run)

    def add_upload(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.uploads.resolve()):
            raise ValueError("Upload outside managed storage")
        metadata = inspect_video(path, self.max_bytes)
        token = uuid.uuid4().hex
        spec = ClipSpec(
            clip_id=token,
            source_video_id="upload-" + metadata["sha256"],
            file_path=path.name,
            sha256=metadata["sha256"],
            source="Operator upload; unverified",
            license="Unknown; local assessment only",
            population="upload",
            selection_method="Operator selected arbitrary footage",
        )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT clip_id,status FROM clips WHERE sha256=?", (metadata["sha256"],)
            ).fetchone()
            if existing:
                path.unlink()
                if existing["status"] != "available":
                    raise ValueError(
                        "Duplicate refers to a deleted clip; preserve its audit identity"
                    )
                return existing["clip_id"]
            db.execute(
                "INSERT INTO clips VALUES(?,?,?,?,?,?,?,?,?,?,1)",
                (
                    token,
                    spec.source_video_id,
                    str(path),
                    spec.sha256,
                    encode(metadata),
                    encode(spec.model_dump()),
                    spec.assignment,
                    spec.population,
                    now(),
                    "available",
                ),
            )
            self.audit(db, "upload", token, "local_operator", {})
        return token

    def clips(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM clips ORDER BY imported_at,clip_id").fetchall()
        return [self.clip_dict(row) for row in rows]

    @staticmethod
    def clip_dict(row):
        return {
            **dict(row),
            "metadata": json.loads(row["metadata"]),
            "spec": json.loads(row["spec"]),
        }

    def clip(self, clip_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
        if row is None:
            raise ValueError("Clip not found")
        return self.clip_dict(row)

    def video_path(self, clip):
        path = Path(clip["file_path"]).resolve()
        root = self.uploads.resolve() if clip["owned_upload"] else self.import_root
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Catalog video unavailable or outside authorized storage")
        return path

    def reference(self, clip_id, payload, author, reason, expected_version):
        clip = self.clip(clip_id)
        if not isinstance(payload, dict):
            raise ValueError("Reference payload must be an object")
        if not author.strip() or not reason.strip():
            raise ValueError("Reference version requires author and change reason")
        if set(payload) - {"class_label", "violence", "events", "temporal_complete"}:
            raise ValueError("Unknown reference field")
        if payload.get("class_label") is not None and payload["class_label"] not in CLASSES:
            raise ValueError("Unknown reference class")
        if payload.get("violence") is not None and type(payload["violence"]) is not bool:
            raise ValueError("Violence reference must be explicit boolean or unknown")
        if type(payload.get("temporal_complete", False)) is not bool:
            raise ValueError("Temporal completeness must be explicit boolean")
        validate_intervals(payload.get("events", []), clip["metadata"]["duration"])
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT COALESCE(MAX(version),0) FROM reference_versions WHERE clip_id=?",
                (clip_id,),
            ).fetchone()[0]
            if current != expected_version:
                raise ValueError(
                    "Reference version changed; inspect current version before updating"
                )
            db.execute(
                "INSERT INTO reference_versions VALUES(?,?,?,?,?,?)",
                (
                    clip_id,
                    current + 1,
                    encode(payload),
                    author,
                    reason,
                    now(),
                ),
            )
            self.audit(
                db,
                "reference_version",
                clip_id,
                author,
                {
                    "version": current + 1,
                    "reason": reason,
                },
            )
        return current + 1

    def review(self, item_id, decision, notes, reviewer, intervals):
        if decision not in {"correct", "incorrect", "uncertain", "not_assessable"}:
            raise ValueError("Invalid review decision")
        if not reviewer.strip() or len(notes) > 5000 or len(reviewer) > 200:
            raise ValueError("Reviewer required; review text exceeds limit")
        with self.connect() as db:
            row = db.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
            if row is None or row["status"] != "completed":
                raise ValueError("Only completed predictions can be reviewed")
            validate_intervals(intervals, self.clip(row["clip_id"])["metadata"]["duration"])
            review_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO reviews VALUES(?,?,?,?,?,?,?)",
                (
                    review_id,
                    item_id,
                    decision,
                    notes,
                    reviewer,
                    encode(intervals),
                    now(),
                ),
            )
            self.audit(db, "review", item_id, reviewer, {"review_id": review_id})
        return review_id

    def backup(self):
        destination = self.directory / ("backup-" + uuid.uuid4().hex + ".sqlite3")
        with self.connect() as source, sqlite3.connect(destination) as target:
            source.backup(target)
        return destination

    def delete_upload(self, clip_id):
        with self.legacy_guard:
            if clip_id in self.legacy_active:
                raise ValueError("Finish active professor inference before deleting upload")
            self._delete_upload(clip_id)

    def _delete_upload(self, clip_id):
        clip = self.clip(clip_id)
        if not clip["owned_upload"]:
            raise ValueError("Only managed uploads may be deleted; source footage is never deleted")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM items WHERE clip_id=? AND status IN ('queued','running')",
                (clip_id,),
            ).fetchone():
                raise ValueError("Cancel and finish active jobs before deleting upload")
            path = Path(clip["file_path"]).resolve()
            if not path.is_relative_to(self.uploads.resolve()):
                raise ValueError("Unsafe managed upload path")
            path.unlink(missing_ok=True)
            db.execute("UPDATE clips SET status='deleted' WHERE clip_id=?", (clip_id,))
            self.audit(db, "delete_upload", clip_id, "local_operator", {})


def validate_intervals(intervals, duration):
    if not isinstance(intervals, list) or len(intervals) > 1000:
        raise ValueError("Intervals must be a bounded list of [start,end] seconds")
    for pair in intervals:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)
            or not 0 <= pair[0] < pair[1] <= duration
        ):
            raise ValueError("Intervals must lie within the clip duration")
