"""Generate offline example receipts from genuine frozen cascade inference."""

import json
import time
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.experiments.dcsass_cache import write_json
from surveillance.inference.loading import load_surveillance_pipeline
from surveillance.inference.provenance import pipeline_identity


def cache_examples(config: dict, root: Path) -> dict:
    root = Path(root)
    expected = pipeline_identity(config)
    entries = json.loads((root / config["examples"]).read_text(encoding="utf-8"))
    pipeline, loading = None, 0.0
    generated, reused, records = 0, 0, []
    for entry in entries:
        video = root / entry["video"]
        video_hash = sha256(video)
        if video_hash != entry["video_sha256"]:
            raise ValueError("Demo example video bytes differ from the frozen selection")
        destination = root / entry["result_cache"]
        if destination.is_file():
            payload = json.loads(destination.read_text(encoding="utf-8"))
            result = payload["result"]
            if (
                payload["video_sha256"] != video_hash
                or result.get("schema_version") != 1
                or result.get("provenance") != expected
            ):
                raise ValueError("Existing example cache identity changed; preserve it for review")
            reused += 1
            was_reused = True
        else:
            if pipeline is None:
                before = time.perf_counter()
                pipeline = load_surveillance_pipeline(config, root)
                loading = time.perf_counter() - before
            result = pipeline.predict_video(video)
            if result.get("schema_version") != 1 or result.get("provenance") != expected:
                raise ValueError("Cascade result does not match the frozen trained deployment")
            result["execution_mode"] = "precomputed_on_demand"
            write_json(
                destination,
                {
                    "schema_version": 1,
                    "video_sha256": video_hash,
                    "result": result,
                    "generation": "genuine SurveillancePipeline on-demand inference",
                },
            )
            generated += 1
            was_reused = False
        records.append(
            {
                "name": entry["name"],
                "video_sha256": video_hash,
                "cache_reused": was_reused,
                "dataset_clip_label": entry.get("expected_behavior"),
                "source_video_id": entry.get("source_video_id"),
                "final_alert": result["final_alert"],
                "timings": result["timings"],
            }
        )
    return {
        "schema_version": 1,
        "provenance": expected,
        "generated": generated,
        "reused": reused,
        "model_loading_seconds_this_execution": loading,
        "population_scope": "curated held-out demonstration cases; not a new accuracy benchmark",
        "records": records,
    }
