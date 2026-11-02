"""Indian pilot baseline: the frozen five-stage cascade, zero-shot, on one split.

A clip the cascade refuses (e.g. strict decode on an overstated frame count) counts
as "no alert" with score 0, as in deployment, and is listed. Test requires the
committed evaluation protocol.
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.datasets.indian_split import verify_split
from surveillance.inference.loading import load_surveillance_pipeline
from surveillance.training.indian_binary import committed_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("manifests/v2-pilot/split_v1"))
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/surveillance_demo_ucf_v1.yaml")
    )
    parser.add_argument("--test-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refuse to overwrite {args.output}")
    protocol = None
    if args.split == "test":
        if args.test_protocol is None:
            raise SystemExit("Test requires --test-protocol (committed EVAL_PROTOCOL.md)")
        protocol = committed_protocol(args.test_protocol)
    rows, receipt = verify_split(args.split_dir)
    chosen = sorted((r for r in rows if r.split == args.split), key=lambda r: r.video_id)
    config = yaml.safe_load(args.config.read_text())
    config["device"] = "cpu"
    started = time.perf_counter()
    pipeline = load_surveillance_pipeline(config, Path("."))
    loading = time.perf_counter() - started
    args.output.mkdir(parents=True)
    clips = {}
    for record in chosen:
        try:
            result = pipeline.predict_video(Path(record.path))
            alert = result["final_alert"]
            clips[record.video_id] = {
                "status": "scored",
                "label": record.label,
                "anomaly_score": result["anomaly"]["overall_score"],
                "alert_state": alert["state"],
                "alert": int(alert["state"] != "no_anomaly"),
                "behavior": alert.get("behavior"),
                "total_seconds": result["timings"]["total_seconds"],
            }
        except ValueError as error:
            clips[record.video_id] = {
                "status": "refused",
                "label": record.label,
                "anomaly_score": 0.0,
                "alert_state": "refused",
                "alert": 0,
                "error": str(error)[:300],
            }
        if sha256(Path(record.path)) != receipt["video_sha256"][record.video_id]:
            raise ValueError("Media changed during baseline inference")
        print(record.video_id, clips[record.video_id]["alert_state"], flush=True)
    write_new_json(
        args.output / "cascade.json",
        {
            "scope": f"zero_shot_cascade_{args.split}",
            "split_freeze_hash": receipt["freeze_hash"],
            "config_sha256": sha256(args.config),
            "device": "cpu",
            "model_loading_seconds": loading,
            "test_protocol": protocol,
            "refused": sorted(k for k, v in clips.items() if v["status"] == "refused"),
            "clips": clips,
        },
    )
    print(
        json.dumps(
            {
                "scored": sum(v["status"] == "scored" for v in clips.values()),
                "refused": sum(v["status"] == "refused" for v in clips.values()),
            }
        )
    )


if __name__ == "__main__":
    run_cli(main)
