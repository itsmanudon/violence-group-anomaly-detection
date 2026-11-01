"""Municipal routes mounted on the existing loopback application."""

import json
import uuid
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from surveillance.demo_rendering import reference_gallery
from surveillance.municipal.catalog import Manifest, now
from surveillance.municipal.reporting import csv_export, evaluation, export_rows, records


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_ids: list[str] = Field(min_length=1, max_length=1000)
    idempotency_key: str = Field(min_length=1, max_length=200)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: str
    reviewer: str = Field(min_length=1, max_length=200)
    notes: str = Field(default="", max_length=5000)
    intervals: list[list[float]] = Field(default_factory=list, max_length=1000)


class Retry(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=200)


def municipal_router(catalog, queue):
    router = APIRouter(prefix="/api/platform")

    def safe(action):
        try:
            return action()
        except (ValueError, OSError) as error:
            detail = str(error) if isinstance(error, ValueError) else "Local storage unavailable"
            raise HTTPException(400, detail) from None

    @router.get("/health")
    def health():
        with catalog.connect() as db:
            schema = db.execute("PRAGMA user_version").fetchone()[0]
        alive = bool(queue.threads) and all(t.is_alive() for t in queue.threads)
        return dict(
            status="ok" if alive else "worker_unavailable",
            schema_version=schema,
            storage="local_sqlite",
            worker_alive=alive,
            concurrency=queue.concurrency,
            gpu_inference_concurrency=1,
            profile=queue.profile,
            timestamp=now(),
        )

    @router.get("/dashboard")
    def dashboard():
        clips, jobs = catalog.clips(), queue.jobs()
        all_records = records(catalog)
        return dict(
            library_count=len(clips),
            jobs=dict(Counter(j["status"] for j in jobs)),
            reviews_pending=sum(
                r["status"] == "completed" and not r["reviews"] for r in all_records
            ),
            evaluation=evaluation(catalog, queue.profile["profile_sha256"]),
        )

    @router.get("/clips")
    def clips(
        q: str = "",
        population: str = "",
        assignment: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ):
        rows = [
            c
            for c in catalog.clips()
            if (not q or q.lower() in json.dumps(c["spec"]).lower())
            and (not population or c["population"] == population)
            and (not assignment or c["assignment"] == assignment)
        ]
        for row in rows:
            row["video_url"] = f"/api/platform/clips/{row['clip_id']}/video"
            row.pop("file_path")
            row["spec"].pop("file_path")
        return dict(total=len(rows), clips=rows[offset : offset + limit])

    @router.post("/import")
    def import_manifest(payload: Manifest, dry_run: bool = True):
        return safe(lambda: catalog.import_manifest(payload.model_dump(), dry_run=dry_run))

    @router.post("/uploads", status_code=201)
    async def upload(file: UploadFile):
        from surveillance.municipal.catalog import EXTENSIONS

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in EXTENSIONS:
            await file.close()
            raise HTTPException(400, "Unsupported video extension")
        path = catalog.uploads / (uuid.uuid4().hex + suffix)
        size = 0
        try:
            with path.open("xb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > catalog.max_bytes:
                        raise HTTPException(413, "Video exceeds configured upload size limit")
                    handle.write(chunk)
            clip_id = await run_in_threadpool(lambda: safe(lambda: catalog.add_upload(path)))
            return dict(clip_id=clip_id, video_url=f"/api/platform/clips/{clip_id}/video")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @router.get("/clips/{clip_id}/video")
    def video(clip_id: str):
        clip = safe(lambda: catalog.clip(clip_id))
        if clip["status"] == "deleted":
            raise HTTPException(404, "Upload deleted under local retention policy")
        path = safe(lambda: catalog.video_path(clip))
        return FileResponse(path)

    @router.delete("/clips/{clip_id}/upload")
    def delete_upload(clip_id: str):
        safe(lambda: catalog.delete_upload(clip_id))
        return dict(status="deleted", clip_id=clip_id)

    @router.post("/jobs", status_code=202)
    def submit(payload: Submission):
        return dict(job_id=safe(lambda: queue.submit(payload.clip_ids, payload.idempotency_key)))

    @router.get("/jobs")
    def jobs(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        rows = queue.jobs()
        for job in rows:
            for item in job["items"]:
                item.pop("result")
        return dict(total=len(rows), jobs=rows[offset : offset + limit])

    @router.get("/jobs/{job_id}")
    def job(job_id: str):
        return safe(lambda: queue.job(job_id))

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        safe(lambda: queue.cancel(job_id))
        return dict(status="cancellation_requested")

    @router.post("/jobs/{job_id}/retry", status_code=202)
    def retry(job_id: str, payload: Retry):
        return dict(job_id=safe(lambda: queue.retry(job_id, payload.idempotency_key)))

    def record(item_id):
        row = next((r for r in records(catalog) if r["item_id"] == item_id), None)
        if row is None:
            raise HTTPException(404, "Prediction not found")
        return row

    @router.get("/items/{item_id}")
    def item(item_id: str):
        row = record(item_id)
        row["clip"].pop("file_path")
        row["video_url"] = f"/api/platform/clips/{row['clip_id']}/video"
        row["overlays"] = [
            dict(
                url=f"/api/platform/items/{item_id}/overlay/{i}",
                caption=f"Actor reference frame {i + 1}",
            )
            for i, window in enumerate((row["result"] or {}).get("behavior", {}).get("windows", []))
            if window.get("boxes")
        ]
        return row

    @router.get("/items/{item_id}/overlay/{index}")
    def overlay(item_id: str, index: int):
        import cv2

        row = record(item_id)
        if row["clip"]["status"] == "deleted":
            raise HTTPException(404, "Upload deleted")
        path = safe(lambda: catalog.video_path(row["clip"]))
        # Render on demand; no persistent facial imagery or frame logs.
        result = {**(row["result"] or {}), "video_path": str(path)}
        windows = result.get("behavior", {}).get("windows", [])
        if not 0 <= index < len(windows):
            raise HTTPException(404, "Reference frame not found")
        result["behavior"] = {"windows": [windows[index]]}
        gallery = safe(lambda: reference_gallery(result))
        if not gallery:
            raise HTTPException(404, "Reference frame unavailable")
        ok, png = cv2.imencode(".png", cv2.cvtColor(gallery[0][0], cv2.COLOR_RGB2BGR))
        if not ok:
            raise HTTPException(400, "Reference frame unavailable")
        return Response(
            png.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"}
        )

    @router.post("/items/{item_id}/reviews", status_code=201)
    def review(item_id: str, payload: Review):
        return dict(
            review_id=safe(
                lambda: catalog.review(
                    item_id,
                    payload.decision,
                    payload.notes,
                    payload.reviewer,
                    payload.intervals,
                )
            )
        )

    @router.get("/items/{item_id}/reference")
    def reveal(item_id: str):
        row = record(item_id)
        if not row["reviews"]:
            raise HTTPException(
                409, "Submit an independent review before revealing reference labels"
            )
        with catalog.connect() as db:
            catalog.audit(db, "reference_reveal", item_id, "local_operator", {})
        return next(
            r.get("reference") for r in records(catalog, references=True) if r["item_id"] == item_id
        )

    @router.get("/results")
    def results(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
        payload = export_rows(catalog)
        rows = payload["records"]
        for row in rows:
            row.pop("result")
        return dict(total=len(rows), records=rows[offset : offset + limit])

    @router.get("/evaluation")
    def metrics():
        return evaluation(catalog, queue.profile["profile_sha256"])

    @router.get("/export")
    def export(
        format: str = "json",
        references: bool = False,
        external: bool = False,
        item_id: str | None = None,
    ):
        if format not in {"json", "csv"}:
            raise HTTPException(400, "Choose JSON or CSV")
        payload = export_rows(catalog, references=references, external=external)
        if item_id:
            payload["records"] = [r for r in payload["records"] if r["item_id"] == item_id]
        payload["evaluation"] = evaluation(
            catalog,
            queue.profile["profile_sha256"],
            {r["clip_id"] for r in payload["records"]} if external or item_id else None,
        )
        with catalog.connect() as db:
            catalog.audit(
                db,
                "export",
                "results",
                "local_operator",
                dict(
                    format=format,
                    references=references,
                    external=external,
                    item_id=item_id,
                ),
            )
        content = (
            json.dumps(payload, indent=2, allow_nan=False)
            if format == "json"
            else csv_export(payload)
        )
        return Response(
            content,
            media_type="application/json" if format == "json" else "text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="results.{format}"',
                "X-Excluded-Rights": str(payload["excluded_rights"]),
            },
        )

    @router.post("/backup")
    def backup():
        path = catalog.backup()
        return FileResponse(path, filename="municipal-backup.sqlite3")

    @router.get("/audit")
    def audit(limit: int = Query(100, ge=1, le=1000)):
        with catalog.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM audit ORDER BY audit_id DESC LIMIT ?",
                    (limit,),
                )
            ]

    return router
