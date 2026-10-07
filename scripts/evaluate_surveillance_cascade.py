"""One registered full cascade pass on frozen human-centric held-out clip labels."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.evaluation.cascade import held_out_cascade_rows, summarize_cascade
from surveillance.experiments.cascade_registration import EXPERIMENT, register_cascade_experiment
from surveillance.experiments.dcsass_cache import write_json
from surveillance.inference.loading import load_surveillance_pipeline
from surveillance.inference.provenance import pipeline_identity, verify_selected_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / EXPERIMENT)
    parser.add_argument("--resume", action="store_true", help="Score pending clips only")
    args = parser.parse_args()
    if args.output.resolve() != (ROOT / EXPERIMENT).resolve():
        raise ValueError("Use the canonical cascade output; changing folders cannot repeat test")
    config = yaml.safe_load(args.config.read_text())
    manifest = ROOT / "data/manifests/dcsass_human_centric_v1_final.jsonl"
    split = ROOT / "runs/dcsass/human_centric_v1/final_split_receipt.json"
    split_receipt = json.loads(split.read_text())
    if sha256(manifest) != split_receipt["manifest_sha256"]:
        raise ValueError("Frozen human-centric manifest changed")
    selected = verify_selected_checkpoint(ROOT / config["sultani_checkpoint"])
    protocol_path = (ROOT / config["sultani_checkpoint"]).parent.parent / "protocol.json"
    if selected.get("protocol_sha256") != sha256(protocol_path):
        raise ValueError("Selected anomaly checkpoint is not bound to its prospective protocol")
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_id"] != "ucf_sultani_shared_safe_v1":
        raise ValueError("This registered cascade population requires the source-safe UCF baseline")
    rows = held_out_cascade_rows(
        [json.loads(line) for line in manifest.read_text().splitlines()], protocol
    )
    expected = pipeline_identity(config)
    behavior_choice = json.loads(
        (ROOT / "runs/dcsass/human_rgb_detected_v1/initialization_selection.json").read_text()
    )
    behavior_archive = torch.load(
        ROOT / config["behavior_checkpoint"], map_location="cpu", weights_only=True
    )
    if behavior_archive.get("manifest_sha256") != sha256(manifest):
        raise ValueError("Behavior checkpoint did not use the frozen human source protocol")
    register_cascade_experiment(
        ROOT,
        config,
        selected,
        behavior_choice,
        sha256(manifest),
        sha256(protocol_path),
        args.output,
        resume=args.resume,
    )
    pipeline = None
    progress_path = args.output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    records = []
    for index, row in enumerate(rows, 1):
        video = Path(row["path"])
        if sha256(video) != row["sha256"]:
            raise ValueError("Frozen held-out clip bytes changed")
        destination = args.output / "clips" / (row["clip_id"] + ".json")
        if row["clip_id"] in progress:
            if sha256(destination) != progress[row["clip_id"]]:
                raise ValueError("Saved cascade output bytes changed")
            record = json.loads(destination.read_text())
            if (
                record["video_sha256"] != row["sha256"]
                or record["result"]["provenance"] != expected
            ):
                raise ValueError("Saved cascade output identity changed")
        else:
            if destination.exists():
                raise ValueError("Unregistered partial result exists; preserve it for diagnosis")
            if pipeline is None:
                pipeline = load_surveillance_pipeline(config, ROOT)
            result = pipeline.predict_video(video)
            if result["provenance"] != expected:
                raise ValueError("Live cascade model identities changed")
            record = {
                k: row[k] for k in ("clip_id", "source_video_id", "binary_label", "group_label")
            }
            record.update(video_sha256=row["sha256"], result=result)
            write_json(destination, record)
            progress[row["clip_id"]] = sha256(destination)
            write_json(progress_path, progress)
        records.append(record)
        if index % 10 == 0 or index == len(rows):
            print(f"Cascade held-out {index}/{len(rows)}", flush=True)
    metrics = summarize_cascade(records)
    write_json(args.output / "metrics.json", metrics)
    write_json(
        args.output / "completion.json",
        {
            "complete": True,
            "clips": len(records),
            "metrics_sha256": sha256(args.output / "metrics.json"),
        },
    )
    print(json.dumps(metrics["coverage"], indent=2), flush=True)


if __name__ == "__main__":
    run_cli(main)
