"""Loopback React demo transport over the existing frozen inference service."""

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from surveillance.demo_profile import profile_hash
from surveillance.demo_rendering import reference_gallery
from surveillance.demo_service import DemoService
from surveillance.inference.loading import asset_status


class AnalyzeRequest(BaseModel):
    example: str | None = None
    upload_id: str | None = None
    use_cache: bool = True


def create_api(config: dict, root: Path, *, frontend: Path | None = None) -> FastAPI:
    root = Path(root)
    service = DemoService(config, root)
    output = root / config["outputs"] / "console"
    output.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frozen-inference")

    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Frozen surveillance research demo", docs_url=None, redoc_url=None, lifespan=lifespan
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    jobs, media = {}, {}
    lock = threading.Lock()
    app.state.service = service

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        allowed = {
            f"http://{request.headers.get('host')}",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        }
        if request.method == "POST" and origin and origin not in allowed:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                {"detail": "Use the local demo page to submit videos"}, status_code=403
            )
        return await call_next(request)

    def register(path):
        token = uuid.uuid4().hex
        media[token] = Path(path)
        return "/api/media/" + token

    @app.get("/api/status")
    def status():
        assets = asset_status(config, root)
        examples = []
        for name, entry in service.examples.items():
            video = root / entry["video"]
            cache = root / entry["result_cache"]
            examples.append(
                {
                    "name": name,
                    "expected_behavior": entry.get("expected_behavior"),
                    "kind": "limitation" if name.startswith("Limitation:") else "verified",
                    "video_available": video.is_file(),
                    "cache_available": cache.is_file(),
                    "video_url": register(video) if video.is_file() else None,
                    "selection_policy": entry.get("selection_policy"),
                }
            )
        return {
            "assets": assets,
            "examples": examples,
            "profile_sha256": profile_hash(config),
            "device_policy": "CPU safe mode on macOS; MPS components unvalidated",
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            raise HTTPException(400, "Choose an MP4, MOV, AVI, MKV, WebM or M4V video")
        token = uuid.uuid4().hex
        path = output / (token + suffix)
        size = 0
        try:
            with path.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 500 * 1024 * 1024:
                        raise HTTPException(413, "Use a short clip smaller than 500 MB")
                    handle.write(chunk)
            if not size:
                raise HTTPException(400, "The uploaded video is empty")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        media[token] = path
        return {"upload_id": token, "video_url": "/api/media/" + token}

    def infer(job_id, payload):
        try:
            jobs[job_id] = {"status": "running"}
            if payload.example:
                result = service.analyze(None, payload.example, payload.use_cache)
            else:
                result = service.analyze(str(media[payload.upload_id]), "Upload a video", False)
            overlays = []
            for index, (rgb, caption) in enumerate(reference_gallery(result)):
                image = output / f"{job_id}-actors-{index}.png"
                if not cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError("Could not save actor reference frame")
                overlays.append({"url": register(image), "caption": caption})
            artifact = output / (job_id + ".json")
            artifact.write_text(json.dumps(result, indent=2, allow_nan=False))
            jobs[job_id] = {
                "status": "complete",
                "result": result,
                "video_url": register(result["video_path"]),
                "overlays": overlays,
                "download_url": register(artifact),
            }
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            jobs[job_id] = {"status": "failed", "error": str(error)}
        finally:
            lock.release()

    @app.post("/api/analyze", status_code=202)
    def analyze(payload: AnalyzeRequest):
        if bool(payload.example) == bool(payload.upload_id):
            raise HTTPException(400, "Choose exactly one example or uploaded video")
        if payload.example and payload.example not in service.examples:
            raise HTTPException(404, "Example is not in the frozen index")
        if payload.upload_id and payload.upload_id not in media:
            raise HTTPException(404, "Upload not found; select the video again")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "An inference is already running; wait for it to finish")
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "queued"}
        pool.submit(infer, job_id, payload)
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Inference request not found")
        return jobs[job_id]

    @app.get("/api/media/{token}")
    def playback(token: str):
        path = media.get(token)
        if path is None or not path.is_file():
            raise HTTPException(404, "Local media unavailable")
        return FileResponse(path, filename=path.name if path.suffix == ".json" else None)

    frontend = frontend or root / "frontend/dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="console")
    return app
