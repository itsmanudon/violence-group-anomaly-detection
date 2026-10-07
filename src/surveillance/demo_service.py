"""Offline demo service: genuine uploaded inference or explicit example-result caches."""

import json
import time
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.inference.loading import load_surveillance_pipeline
from surveillance.inference.provenance import pipeline_identity
from surveillance.video.decode import probe_video


class DemoService:
    def __init__(self, config: dict, root: Path):
        self.config, self.root = config, Path(root)
        path = self.root / config["examples"]
        entries = json.loads(path.read_text()) if path.is_file() else []
        self.examples = {entry["name"]: entry for entry in entries}
        self.pipeline = None

    def analyze(self, video: str | None, example: str, use_cache: bool = True) -> dict:
        started = time.perf_counter()
        entry = self.examples.get(example)
        if entry:
            path = self.root / entry["video"]
            video_hash = sha256(path)
            if entry.get("video_sha256") != video_hash:
                raise ValueError("Frozen example video changed or its identity is missing")
            cached = entry.get("result_cache")
            if use_cache and cached and (self.root / cached).is_file():
                payload = json.loads((self.root / cached).read_text())
                if payload["video_sha256"] != video_hash:
                    raise ValueError("Cached example video changed; recompute its result")
                result = dict(payload["result"])
                if result.get("schema_version") != 1:
                    raise ValueError("Unsupported demo result cache")
                if result.get("provenance") != pipeline_identity(self.config):
                    raise ValueError(
                        "Cached example model or policy provenance differs from this demo"
                    )
                result.update(execution_mode="cached_example", video_path=str(path))
                return result
        else:
            if not video:
                raise ValueError("Upload a video or select an installed example")
            path = Path(video[0] if isinstance(video, tuple) else video)
        probe_video(path)
        loading = 0.0
        if self.pipeline is None:
            before = time.perf_counter()
            self.pipeline = load_surveillance_pipeline(self.config, self.root)
            loading = time.perf_counter() - before
        result = self.pipeline.predict_video(path)
        result["timings"]["model_loading_seconds"] = loading
        result["timings"]["request_total_seconds"] = time.perf_counter() - started
        result["execution_mode"] = "on_demand"
        return result
