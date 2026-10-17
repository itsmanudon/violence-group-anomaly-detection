"""Durable per-clip jobs. A process lease prevents competing GPU owners."""

import copy
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from surveillance.municipal.catalog import encode, fingerprint, now


class ProcessLease:
    def __init__(self, directory):
        self.file = open(directory / "worker.lock", "a+b")
        self.file.seek(0)
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("Another municipal worker owns this catalog") from None

    def close(self):
        if not self.file.closed:
            self.file.close()


def validate_result(result, identity, mode):
    if result.get("schema_version") != 1 or result.get("provenance") != identity:
        raise ValueError("Inference result does not match the frozen submitted model profile")
    scores = result.get("anomaly", {}).get("segment_scores", [])
    if len(scores) != 32 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError("Inference requires 32 finite anomaly scores")
    if result.get("execution_mode") != mode:
        raise ValueError("Execution mode differs from submitted job")
    for field in ("video_metadata", "behavior", "alerts", "final_alert", "timings", "policy"):
        if field not in result:
            raise ValueError(f"Incomplete inference result: {field}")
    encode(result)


class JobQueue:
    def __init__(
        self, catalog, identity, runner, *, concurrency=1, max_retries=1, config_sha256=None
    ):
        if type(concurrency) is not int or not 1 <= concurrency <= 4:
            raise ValueError("Concurrency must be between one and four; use one for a GPU")
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ValueError("Retries must be between zero and three")
        self.catalog, self.identity = catalog, copy.deepcopy(identity)
        self.config_sha256 = config_sha256 or fingerprint(identity)
        self.profile_hash = fingerprint([identity, self.config_sha256])
        self.runner = runner
        self.concurrency, self.max_retries = concurrency, max_retries
        self.stop_event = threading.Event()
        self.executor = None
        self.lease = None

    def acquire(self):
        if self.lease:
            return
        self.lease = ProcessLease(self.catalog.directory)
        with self.catalog.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE items SET state=CASE WHEN job_id IN "
                "(SELECT job_id FROM jobs WHERE cancel_requested=1) "
                "THEN 'cancelled' ELSE 'queued' END,"
                "error='Application restart interrupted this attempt',started_at=NULL "
                "WHERE state='running'"
            )
            self.catalog.audit(db, "worker_recovery", "queue", {})

    def submit(self, clip_ids, submission_key, mode="on_demand"):
        if mode not in ("on_demand", "cached_example", "synthetic_fixture"):
            raise ValueError("Unknown execution mode")
        if not isinstance(submission_key, str) or not 1 <= len(submission_key) <= 128:
            raise ValueError("An idempotency key of 1–128 characters is required")
        if not isinstance(clip_ids, list) or not 1 <= len(clip_ids) <= 1000:
            raise ValueError("Select between one and 1000 clips")
        ids = sorted(set(clip_ids))
        for clip_id in ids:
            clip = self.catalog.clip(clip_id)
            if mode == "synthetic_fixture" and clip["cohort"] != "synthetic_fixture":
                raise ValueError("Fixture mode requires fixture clips")
        request_hash = fingerprint([ids, self.profile_hash, mode])
        with self.catalog.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT * FROM jobs WHERE submission_key=?", (submission_key,)
            ).fetchone()
            if previous:
                if previous["request_hash"] != request_hash:
                    raise ValueError("Idempotency key already identifies a different submission")
                return previous["job_id"]
            job_id = uuid.uuid4().hex
            created = now()
            db.execute(
                "INSERT INTO jobs(job_id,submission_key,request_hash,profile_hash,provenance,"
                "execution_mode,created_at,cancel_requested,config_sha256) "
                "VALUES(?,?,?,?,?,?,?,0,?)",
                (
                    job_id,
                    submission_key,
                    request_hash,
                    self.profile_hash,
                    encode(self.identity),
                    mode,
                    created,
                    self.config_sha256,
                ),
            )
            for clip_id in ids:
                db.execute(
                    "INSERT INTO items(item_id,job_id,clip_id,state,created_at) "
                    "VALUES(?,?,?,'queued',?)",
                    (uuid.uuid4().hex, job_id, clip_id, created),
                )
            self.catalog.audit(db, "job_submit", job_id, {"clip_ids": ids, "mode": mode})
        return job_id

    def jobs(self):
        with self.catalog.connect() as db:
            jobs = [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY created_at DESC")]
            for job in jobs:
                job["items"] = [
                    dict(r)
                    for r in db.execute(
                        "SELECT item_id,clip_id,state,attempts,started_at,finished_at,error "
                        "FROM items WHERE job_id=? ORDER BY created_at,item_id",
                        (job["job_id"],),
                    )
                ]
                states = [r["state"] for r in job["items"]]
                job["state"] = (
                    "running"
                    if "running" in states
                    else "queued"
                    if "queued" in states
                    else "failed"
                    if "failed" in states
                    else "cancelled"
                    if "cancelled" in states
                    else "completed"
                )
                job["progress"] = sum(
                    s in ("completed", "failed", "cancelled") for s in states
                ) / len(states)
        return jobs

    def cancel(self, job_id):
        with self.catalog.connect() as db:
            if not db.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone():
                raise ValueError("Unknown job ID")
            db.execute("UPDATE jobs SET cancel_requested=1 WHERE job_id=?", (job_id,))
            db.execute(
                "UPDATE items SET state='cancelled',finished_at=? "
                "WHERE job_id=? AND state='queued'",
                (now(), job_id),
            )
            self.catalog.audit(db, "job_cancel", job_id, {})

    def retry(self, job_id, key):
        with self.catalog.connect() as db:
            job = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            ids = [
                r[0]
                for r in db.execute(
                    "SELECT clip_id FROM items WHERE job_id=? AND state IN ('failed','cancelled')",
                    (job_id,),
                )
            ]
        if not job or not ids:
            raise ValueError("No failed or cancelled items to retry")
        if job["profile_hash"] != self.profile_hash:
            raise ValueError("Retry profile changed; submit an explicitly new job")
        return self.submit(ids, key, job["execution_mode"])

    def process_one(self):
        self.acquire()
        with self.catalog.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT i.*,j.profile_hash,j.execution_mode FROM items i JOIN jobs j USING(job_id) "
                "WHERE i.state='queued' AND j.cancel_requested=0 "
                "ORDER BY i.created_at,i.item_id LIMIT 1"
            ).fetchone()
            if not row:
                return False
            db.execute(
                "UPDATE items SET state='running',attempts=attempts+1,started_at=?,error=NULL "
                "WHERE item_id=?",
                (now(), row["item_id"]),
            )
        try:
            if row["profile_hash"] != self.profile_hash:
                raise ValueError(
                    "Frozen profile changed after submission; restore the submitted configuration"
                )
            path = self.catalog.media_path(row["clip_id"])
            result = self.runner(path, row["execution_mode"])
            validate_result(result, self.identity, row["execution_mode"])
            result = {
                **result,
                "profile_hash": self.profile_hash,
                "config_sha256": self.config_sha256,
                "video_sha256": self.catalog.clip(row["clip_id"])["sha256"],
            }
            with self.catalog.connect() as db:
                db.execute(
                    "UPDATE items SET state='completed',result=?,finished_at=? WHERE item_id=?",
                    (encode(result), now(), row["item_id"]),
                )
        except Exception as error:
            # Persist no raw path, video bytes, frame imagery or arbitrary backend traceback.
            transient = isinstance(error, (TimeoutError, ConnectionError))
            category = (
                "transient_failure"
                if transient
                else "missing_asset_or_media"
                if isinstance(error, FileNotFoundError)
                else "validation_failure"
                if isinstance(error, ValueError)
                else "inference_failure"
            )
            message = category + ": " + type(error).__name__
            guidance = {
                "missing_asset_or_media": "Run asset preflight; "
                "verify the catalog video is installed.",
                "validation_failure": "Check video decoding, frozen hashes and submitted profile; "
                "no substitute model was used.",
                "inference_failure": "Check device memory and local runtime; "
                "retry only after correction.",
                "transient_failure": "Temporary execution failure; automatic retries are bounded.",
            }
            message += ". " + guidance[category]
            with self.catalog.connect() as db:
                cancelled = db.execute(
                    "SELECT cancel_requested FROM jobs WHERE job_id=?", (row["job_id"],)
                ).fetchone()[0]
                retry = transient and row["attempts"] < self.max_retries and not cancelled
                state = "cancelled" if cancelled else "queued" if retry else "failed"
                db.execute(
                    "UPDATE items SET state=?,error=?,finished_at=? WHERE item_id=?",
                    (state, message, None if retry else now(), row["item_id"]),
                )
        return True

    def start(self):
        self.acquire()
        if self.executor:
            return
        self.executor = ThreadPoolExecutor(
            max_workers=self.concurrency, thread_name_prefix="municipal"
        )
        for _ in range(self.concurrency):
            self.executor.submit(self._loop)

    def _loop(self):
        while not self.stop_event.is_set():
            if not self.process_one():
                self.stop_event.wait(0.25)

    def close(self):
        self.stop_event.set()
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        if self.lease:
            self.lease.close()
            self.lease = None


class FrozenRunner:
    """Reuse DemoService under a lock: even multiple workers serialize GPU calls."""

    def __init__(self, config, root):
        from surveillance.demo_service import DemoService

        self.service = DemoService(copy.deepcopy(config), root)
        self.lock = threading.Lock()

    def __call__(self, path, mode):
        from surveillance.datasets.dcsass_audit import sha256

        with self.lock:
            if mode == "cached_example":
                digest = sha256(path)
                example = next(
                    (
                        name
                        for name, entry in self.service.examples.items()
                        if entry.get("video_sha256") == digest
                    ),
                    None,
                )
                if example is None:
                    raise ValueError(
                        "Only explicitly installed byte-pinned examples have cache mode"
                    )
                entry = self.service.examples[example]
                if (
                    not entry.get("result_cache")
                    or not (self.service.root / entry["result_cache"]).is_file()
                ):
                    raise FileNotFoundError("Requested example cache is not installed")
                result = self.service.analyze(None, example, use_cache=True)
                result["video_path"] = str(path)
                return result
            if mode != "on_demand":
                raise ValueError(
                    "Production runner only supports genuine inference or explicit example cache"
                )
            before = time.perf_counter()
            result = self.service.analyze(str(path), "", use_cache=False)
            result["timings"]["worker_wall_seconds"] = time.perf_counter() - before
            return result
