"""SQLite durable clip jobs; serialized frozen inference and restart recovery."""

import json
import threading
import uuid

from surveillance.demo_profile import profile_hash
from surveillance.inference.provenance import pipeline_identity
from surveillance.municipal.catalog import digest, encode, now


class JobQueue:
    def __init__(self, catalog, service, model_lock=None, concurrency=1, max_retries=1):
        if not 1 <= concurrency <= 4 or not 0 <= max_retries <= 3:
            raise ValueError("Concurrency must be 1–4 and retries 0–3")
        self.catalog, self.service = catalog, service
        self.profile = {
            "profile_sha256": profile_hash(service.config),
            "provenance": pipeline_identity(service.config),
        }
        self.model_lock = model_lock or threading.Lock()
        self.concurrency, self.max_retries = concurrency, max_retries
        self.stop_event, self.wake = threading.Event(), threading.Event()
        self.threads = []
        self.owner = None

    def start(self):
        # One process owns recovery and workers, including on Windows.
        self.owner = (self.catalog.directory / "worker.lock").open("a+b")
        try:
            import os

            if os.name == "nt":
                import msvcrt

                self.owner.seek(0)
                self.owner.write(b"0")
                self.owner.flush()
                self.owner.seek(0)
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            self.owner = None
            raise RuntimeError("Another municipal application owns this catalog") from None
        self.recover()
        for index in range(self.concurrency):
            thread = threading.Thread(target=self.run, name=f"municipal-{index}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.stop_event.set()
        self.wake.set()
        for thread in self.threads:
            thread.join()
        if self.owner:
            self.owner.close()
            self.owner = None

    def recover(self):
        with self.catalog.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE items SET status='queued',error='Interrupted; safely rerun after restart' "
                "WHERE status='running' AND attempts<=?",
                (self.max_retries,),
            )
            db.execute(
                "UPDATE items SET status='failed',finished_at=?,"
                "error='Restart retry budget exhausted' "
                "WHERE status='running'",
                (now(),),
            )
            db.execute(
                "UPDATE items SET status='cancelled',finished_at=? WHERE status='queued' "
                "AND job_id IN (SELECT job_id FROM jobs WHERE cancel_requested=1)",
                (now(),),
            )
            for row in db.execute("SELECT job_id FROM jobs").fetchall():
                self.aggregate(db, row[0])
            self.catalog.audit(db, "queue_recovery", "queue", "worker", {})

    def submit(self, clip_ids, idempotency_key):
        if not clip_ids or len(clip_ids) > 1000 or len(set(clip_ids)) != len(clip_ids):
            raise ValueError("Select 1–1000 distinct clips")
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200:
            raise ValueError("A bounded idempotency key is required")
        request = encode(sorted(clip_ids))
        profile = encode(self.profile)
        with self.catalog.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing:
                if existing["request"] != request or existing["profile"] != profile:
                    raise ValueError("Idempotency key was already used for a different request")
                return existing["job_id"]
            for clip_id in clip_ids:
                row = db.execute("SELECT status FROM clips WHERE clip_id=?", (clip_id,)).fetchone()
                if row is None or row[0] != "available":
                    raise ValueError("Selected clip is unavailable")
            job_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,NULL)",
                (
                    job_id,
                    idempotency_key,
                    request,
                    profile,
                    "queued",
                    0,
                    now(),
                ),
            )
            db.executemany(
                "INSERT INTO items(item_id,job_id,clip_id,status) VALUES(?,?,?,'queued')",
                ((uuid.uuid4().hex, job_id, clip_id) for clip_id in clip_ids),
            )
            self.catalog.audit(db, "submit", job_id, "local_operator", clip_ids)
        self.wake.set()
        return job_id

    def cancel(self, job_id):
        with self.catalog.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone():
                raise ValueError("Job not found")
            db.execute("UPDATE jobs SET cancel_requested=1 WHERE job_id=?", (job_id,))
            db.execute(
                "UPDATE items SET status='cancelled',finished_at=? "
                "WHERE job_id=? AND status='queued'",
                (now(), job_id),
            )
            self.aggregate(db, job_id)
            self.catalog.audit(db, "cancel", job_id, "local_operator", {})

    def retry(self, job_id, key):
        job = self.job(job_id)
        clips = [i["clip_id"] for i in job["items"] if i["status"] in {"failed", "cancelled"}]
        if not clips:
            raise ValueError("No failed/cancelled clips to retry")
        return self.submit(clips, key)

    def job(self, job_id):
        with self.catalog.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("Job not found")
            items = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM items WHERE job_id=? ORDER BY rowid",
                    (job_id,),
                )
            ]
        for item in items:
            item["result"] = json.loads(item["result"]) if item["result"] else None
        return {
            **dict(row),
            "profile": json.loads(row["profile"]),
            "items": items,
            "progress": sum(i["status"] in {"completed", "failed", "cancelled"} for i in items)
            / len(items),
        }

    def jobs(self):
        with self.catalog.connect() as db:
            ids = [row[0] for row in db.execute("SELECT job_id FROM jobs ORDER BY created_at DESC")]
        return [self.job(job_id) for job_id in ids]

    @staticmethod
    def aggregate(db, job_id):
        states = [r[0] for r in db.execute("SELECT status FROM items WHERE job_id=?", (job_id,))]
        if not states:
            return
        terminal = not any(s in {"queued", "running"} for s in states)
        state = (
            ("running" if "running" in states else "queued")
            if not terminal
            else (
                "failed"
                if "failed" in states
                else "cancelled"
                if "cancelled" in states
                else "completed"
            )
        )
        db.execute(
            "UPDATE jobs SET status=?,finished_at=? WHERE job_id=?",
            (
                state,
                now() if terminal else None,
                job_id,
            ),
        )

    def claim(self):
        with self.catalog.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT items.*,jobs.profile FROM items JOIN jobs USING(job_id) "
                "WHERE items.status='queued' AND jobs.cancel_requested=0 "
                "ORDER BY jobs.created_at,items.rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE items SET status='running',attempts=attempts+1,started_at=? "
                "WHERE item_id=?",
                (now(), row["item_id"]),
            )
            self.aggregate(db, row["job_id"])
            return dict(row)

    def process_one(self):
        # Claim inside shared lock: no prefetched GPU work escapes cancellation.
        with self.model_lock:
            if self.stop_event.is_set():
                return False
            item = self.claim()
            if item is None:
                return False
            try:
                if json.loads(item["profile"]) != self.profile:
                    raise ValueError(
                        "Frozen profile changed since submission; explicit resubmit required"
                    )
                clip = self.catalog.clip(item["clip_id"])
                path = self.catalog.video_path(clip)
                if digest(path) != clip["sha256"]:
                    raise ValueError("Catalog video changed since import")
                result = self.service.analyze(str(path), "Upload a video", False)
                if (
                    result.get("execution_mode") != "on_demand"
                    or result.get("video_sha256") != clip["sha256"]
                    or result.get("provenance") != self.profile["provenance"]
                    or result.get("profile_sha256") != self.profile["profile_sha256"]
                    or len(result.get("anomaly", {}).get("segment_scores", [])) != 32
                ):
                    raise ValueError(
                        "Inference result violates frozen identity/32-segment contract"
                    )
                result["catalog_video_metadata"] = clip["metadata"]
                state, error = "completed", None
            except Exception as exc:
                # Keep filenames and untrusted metadata out of operator error summaries.
                transient = isinstance(exc, TimeoutError)
                state = "queued" if transient and item["attempts"] < self.max_retries else "failed"
                if transient:
                    error = "Transient inference timeout; bounded retry"
                elif isinstance(exc, FileNotFoundError):
                    error = "Required video/checkpoint missing; inspect local assets with preflight"
                elif str(exc) in {
                    "Catalog video changed since import",
                    "Frozen profile changed since submission; explicit resubmit required",
                    "Inference result violates frozen identity/32-segment contract",
                    "Catalog video unavailable or outside authorized storage",
                }:
                    error = str(exc)
                elif "out of memory" in str(exc).lower():
                    error = "Model memory exhausted; restart and use an explicit suitable device"
                else:
                    error = "Inference unavailable: check decoding and frozen assets with preflight"
                result = None
            with self.catalog.connect() as db:
                db.execute(
                    "UPDATE items SET status=?,error=?,result=?,finished_at=? WHERE item_id=?",
                    (
                        state,
                        error,
                        encode(result) if result else None,
                        None if state == "queued" else now(),
                        item["item_id"],
                    ),
                )
                if state == "queued":
                    db.execute(
                        "UPDATE items SET status='cancelled',finished_at=? WHERE item_id=? "
                        "AND job_id IN (SELECT job_id FROM jobs WHERE cancel_requested=1)",
                        (now(), item["item_id"]),
                    )
                self.aggregate(db, item["job_id"])
            return True

    def run(self):
        while not self.stop_event.is_set():
            if not self.process_one():
                self.wake.wait(0.5)
                self.wake.clear()
