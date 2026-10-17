"""Loopback HTTP API on the existing Gradio/FastAPI stack; never serves arbitrary paths."""

import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from surveillance.municipal.catalog import EXTENSIONS


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Submission(StrictBody):
    clip_ids: list[str] = Field(min_length=1, max_length=1000)
    idempotency_key: str = Field(min_length=1, max_length=128)
    execution_mode: str = "on_demand"


class Review(StrictBody):
    item_id: str
    reviewer: str
    decision: str
    notes: str = ""
    event_intervals: list[list[float]] = Field(default_factory=list)


class Reference(StrictBody):
    payload: dict
    author: str
    reason: str
    expected_version: int = Field(ge=0)


def create_api(service, *, start_worker=True, allowed_host="127.0.0.1"):
    @asynccontextmanager
    async def lifespan(app):
        service.catalog.purge_temporary(service.retention_hours)
        if start_worker:
            service.queue.start()
        try:
            yield
        finally:
            service.queue.close()

    app = FastAPI(title="Municipal research evaluation", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def local_boundary(request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        if host not in {"127.0.0.1", "localhost", allowed_host}:
            return JSONResponse(
                {"detail": "Host is not configured for this local application"}, status_code=403
            )
        origin = request.headers.get("origin")
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site requests are disabled"}, status_code=403)
        if origin and (
            urlparse(origin).netloc != request.headers.get("host")
            or urlparse(origin).scheme not in ("http", "https")
        ):
            return JSONResponse({"detail": "Cross-origin access is disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "connect-src 'self'; font-src 'self' data:; frame-ancestors 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse(
            {"detail": "Required local media or asset is missing; run preflight"}, status_code=404
        )

    @app.get("/api/health")
    def health():
        return service.health()

    @app.post("/api/preflight")
    def preflight():
        return service.check_assets()

    @app.get("/api/clips")
    def clips(search: str = "", assignment: str | None = None):
        # No reference labels, free-form notes or filesystem paths in blinded catalog responses.
        return [
            {k: v for k, v in c.items() if k not in ("file_path", "notes")}
            for c in service.catalog.clips(search, assignment)
        ]

    @app.get("/api/clips/{clip_id}/video")
    def media(clip_id: str):
        return FileResponse(service.catalog.media_path(clip_id))

    @app.post("/api/uploads")
    async def upload(request: Request, filename: str):
        if request.headers.get("x-municipal-request") != "local-upload":
            raise HTTPException(400, "Upload requires X-Municipal-Request: local-upload")
        if (
            not filename
            or any(c in filename for c in ("/", "\\", ":", "\x00"))
            or len(filename) > 200
        ):
            raise ValueError("Unsafe upload filename")
        from pathlib import Path

        extension = Path(filename).suffix.lower()
        if extension not in EXTENSIONS:
            raise ValueError("Unsupported video extension")
        path = service.catalog.temporary / (uuid.uuid4().hex + extension)
        size = 0
        try:
            with path.open("xb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > service.catalog.max_bytes:
                        raise HTTPException(413, "Upload exceeds the configured size limit")
                    stream.write(chunk)
            # Decoding/hash and SQLite operations run off the event loop.
            from starlette.concurrency import run_in_threadpool

            return await run_in_threadpool(service.import_upload, path)
        finally:
            path.unlink(missing_ok=True)

    @app.get("/api/jobs")
    def jobs():
        return service.queue.jobs()

    @app.post("/api/jobs")
    def submit(body: Submission):
        if body.execution_mode == "synthetic_fixture":
            raise ValueError("Synthetic execution is available only in software tests")
        return {
            "job_id": service.queue.submit(body.clip_ids, body.idempotency_key, body.execution_mode)
        }

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        service.queue.cancel(job_id)
        return {"cancel_requested": True, "boundary": "after the current clip"}

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: str, idempotency_key: str = Body(embed=True)):
        return {"job_id": service.queue.retry(job_id, idempotency_key)}

    @app.get("/api/items/{item_id}")
    def item(item_id: str):
        data = service.item(item_id)
        if data["result"]:
            data["result"] = {k: v for k, v in data["result"].items() if k != "video_path"}
        return data

    @app.post("/api/items/{item_id}/reveal-reference")
    def reveal(item_id: str):
        return service.reveal(item_id)

    @app.post("/api/reviews")
    def review(body: Review):
        return {"review_id": service.review(**body.model_dump())}

    @app.post("/api/references/{clip_id}")
    def reference(clip_id: str, body: Reference):
        return service.catalog.add_reference(clip_id, **body.model_dump())

    @app.get("/api/evaluation")
    def evaluation(cohort: str = "untouched_evaluation", mode: str = "on_demand"):
        return service.evaluation(cohort, mode)

    @app.get("/api/exports/{format}")
    def export(format: str, include_references: bool = False, item_id: str | None = None):
        path = service.export(
            format, include_references=include_references, item_ids=[item_id] if item_id else None
        )
        return FileResponse(path, filename="municipal-evaluation." + format)

    @app.post("/api/backup")
    def backup():
        path = service.catalog.backup(
            service.catalog.directory / "exports" / (uuid.uuid4().hex + ".sqlite3")
        )
        return FileResponse(path, filename="municipal-catalog-backup.sqlite3")

    return app
