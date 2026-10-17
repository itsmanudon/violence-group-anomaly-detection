"""Explicit local frozen-model queue smoke; never interpreted as benchmark scoring."""

import argparse
import json
from pathlib import Path

import yaml
from _common import run_cli

from surveillance.municipal.service import MunicipalService

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surveillance_demo.yaml")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    service = MunicipalService(args.output, config, ROOT)
    try:
        assets = service.check_assets()
        if not assets["ready"]:
            raise ValueError(assets["guidance"])
        path = args.video.resolve()
        clip = service.catalog.import_clip(
            {
                "file_path": str(path),
                "source_video_id": "explicit-smoke-" + path.stem,
                "source": "operator-selected local smoke footage; no geographic assertion",
                "license": "retain original acquisition terms; not redistributed",
                "external_handover_permission": "unknown",
                "training_permission": "no",
                "assignment": "demo",
                "cohort": "curated_demo",
                "selection_method": "explicit engineering smoke; not benchmark sampling",
            },
            path.parent,
        )["clip_id"]
        job = service.queue.submit([clip], "explicit-frozen-model-smoke")
        service.queue.process_one()
        status = next(j for j in service.queue.jobs() if j["job_id"] == job)
        if status["state"] != "completed":
            raise ValueError("Frozen-model smoke failed: " + json.dumps(status))
        item = service.item(status["items"][0]["item_id"])
        output = {
            "verified": True,
            "benchmark": False,
            "job": status,
            "config_sha256": item["result"]["config_sha256"],
            "model_provenance": item["result"]["provenance"],
            "execution_mode": item["result"]["execution_mode"],
            "anomaly_segments": len(item["result"]["anomaly"]["segment_scores"]),
            "timings": item["result"]["timings"],
            "export": str(service.export("json")),
        }
        (args.output / "receipt.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output, indent=2))
    finally:
        service.queue.close()


if __name__ == "__main__":
    run_cli(main)
