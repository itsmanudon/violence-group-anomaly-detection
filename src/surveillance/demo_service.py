"""Offline demo service: genuine uploaded inference or explicit example-result caches."""

import json
import logging
import time
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.demo_profile import profile_hash
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
        self.device_fallback = None
        self.pilot = None
        if config.get("indian_pilot"):
            from surveillance.inference.indian_pilot import IndianPilotHead

            self.pilot = IndianPilotHead(config["indian_pilot"], self.root)

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
                result.update(
                    execution_mode="cached_example",
                    video_path=str(path),
                    profile_sha256=profile_hash(self.config),
                    request_seconds=time.perf_counter() - started,
                )
                return result
        else:
            if not video:
                raise ValueError("Upload a video or select an installed example")
            path = Path(video[0] if isinstance(video, tuple) else video)
        probe_video(path)
        loading = 0.0
        try:
            if self.pipeline is None:
                before = time.perf_counter()
                self.pipeline = load_surveillance_pipeline(self.config, self.root)
                loading = time.perf_counter() - before
            result = self.pipeline.predict_video(path)
        except (RuntimeError, NotImplementedError) as error:
            devices = getattr(self.pipeline, "component_devices", {})
            mps_requested = self.config.get("device") == "mps" or "mps" in devices.values()
            if not mps_requested or "mps" not in str(error).lower():
                raise
            self.device_fallback = {"from": "mps", "to": "cpu", "reason": str(error)}
            logging.getLogger(__name__).warning(
                "MPS operator failed; retrying frozen inference on CPU: %s", error
            )
            before = time.perf_counter()
            self.pipeline = load_surveillance_pipeline({**self.config, "device": "cpu"}, self.root)
            loading += time.perf_counter() - before
            result = self.pipeline.predict_video(path)
        if self.device_fallback:
            result["device_fallback"] = self.device_fallback
        if self.pilot is not None:
            result["indian_pilot"] = self.pilot.score(self.pipeline.anomaly.last_features)
        result["timings"]["model_loading_seconds"] = loading
        result["timings"]["request_total_seconds"] = time.perf_counter() - started
        result["execution_mode"] = "on_demand"
        result["video_sha256"] = sha256(path)
        result["profile_sha256"] = profile_hash(self.config)
        result["component_devices"] = getattr(self.pipeline, "component_devices", {})
        return result
