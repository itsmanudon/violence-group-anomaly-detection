"""Infer fixed historical common validation once, then diagnose identical cached outputs."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.evaluation.cascade import summarize_cascade
from surveillance.evaluation.cascade_review import compare_review_modes
from surveillance.evaluation.controlled_review import diagnose, eligible_validation
from surveillance.experiments.environment import environment_metadata
from surveillance.inference.loading import load_surveillance_pipeline
from surveillance.inference.provenance import pipeline_identity, verify_selected_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def paths():
    return {
        "config": ROOT / "configs/surveillance_demo_ucf_v1.yaml",
        "manifest": ROOT / "data/manifests/dcsass_human_centric_v1_final.jsonl",
        "human_receipt": ROOT / "runs/dcsass/human_centric_v1/final_split_receipt.json",
        "anomaly_protocol": ROOT / "runs/ucf-crime/sultani_shared_safe_v1/protocol.json",
        "behavior_registration": ROOT
        / "runs/dcsass/human_rgb_detected_v1/random_init_seed_0/training_registration.json",
        "study": ROOT / "configs/indian-domain/controlled-review-20261026.json",
        "runner": Path(__file__).resolve(),
        "diagnostic_code": ROOT / "src/surveillance/evaluation/controlled_review.py",
        "comparison_code": ROOT / "src/surveillance/evaluation/cascade_review.py",
        "assessment_code": ROOT / "src/surveillance/inference/cascade_review.py",
        "pipeline_code": ROOT / "src/surveillance/inference/surveillance_pipeline.py",
    }


def infer(output):
    if output.exists():
        raise FileExistsError("Refuse to overwrite inference artifacts")
    inputs = paths()
    hashes = {key: sha256(p) for key, p in inputs.items()}
    config = yaml.safe_load(inputs["config"].read_text())
    if config["anomaly_threshold"] != 0.5 or config["max_behavior_windows"] != 3:
        raise ValueError("Preserve the registered legacy threshold and window budget")
    human = json.loads(inputs["human_receipt"].read_text())
    protocol = json.loads(inputs["anomaly_protocol"].read_text())
    behavior = json.loads(inputs["behavior_registration"].read_text())
    if (
        human["manifest_sha256"] != hashes["manifest"]
        or behavior["manifest_sha256"] != hashes["manifest"]
        or behavior["split_receipt_sha256"] != hashes["human_receipt"]
    ):
        raise ValueError("Historical human manifest/training receipt changed")
    selected = verify_selected_checkpoint(ROOT / config["sultani_checkpoint"])
    if selected["protocol_sha256"] != hashes["anomaly_protocol"]:
        raise ValueError("Anomaly protocol differs from selected checkpoint")
    manifest = [json.loads(s) for s in inputs["manifest"].read_text().splitlines() if s.strip()]
    rows = eligible_validation(manifest, human, protocol)
    for row in rows:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Validation media changed: {row['clip_id']}")
    if not torch.cuda.is_available() or "4070" not in torch.cuda.get_device_name(0):
        raise ValueError("The local RTX 4070 is required for this bounded pass")
    torch.cuda.init()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    started = time.perf_counter()
    pipeline = load_surveillance_pipeline(config, ROOT)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    archive = pipeline.behavior.saved
    if archive["manifest_sha256"] != hashes["manifest"]:
        raise ValueError("Behavior checkpoint used a different source manifest")
    output.mkdir(parents=True, exist_ok=False)
    registration = {
        "scope": "existing_common_model_validation_only_diagnostic_no_optimization",
        "inputs": {k: {"path": str(p), "sha256": hashes[k]} for k, p in inputs.items()},
        "provenance": pipeline_identity(config),
        "count": len(rows),
        "source_ids": sorted({r["source_video_id"] for r in rows}),
        "validation_ids": [r["clip_id"] for r in rows],
        "environment": environment_metadata(hashes["config"], 0, "cuda:0"),
        "model_loading_seconds": load_seconds,
        "gpu_total_bytes": total,
        "gpu_free_before_loading_bytes": free,
        "validation_previously_used_for_checkpoint_selection": True,
        "threshold_tuned_in_this_experiment": False,
        "protected_test_inference": False,
        "indian_footage_accessed": False,
        "training_performed": False,
        "grouping": "original source recordings; shared incidents across recordings not annotated",
    }
    write_new_json(output / "registration.json", registration)
    # Pure schema adapter for the existing validation comparison command.
    with (output / "comparison-manifest.jsonl").open("x", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps({**r, "video_sha256": r["sha256"]}) + "\n")
    (output / "clips").mkdir()
    records, result_hashes, wall = [], {}, []
    torch.cuda.reset_peak_memory_stats()
    before_loop = time.perf_counter()
    for i, row in enumerate(rows):
        torch.cuda.synchronize()
        before = time.perf_counter()
        result = pipeline.predict_video(Path(row["path"]))
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - before)
        if (
            result["provenance"] != registration["provenance"]
            or sha256(Path(row["path"])) != row["sha256"]
        ):
            raise ValueError("Inference identity/media changed")
        record = {
            key: row[key]
            for key in ("clip_id", "source_video_id", "binary_label", "group_label", "split")
        }
        record.update(video_sha256=row["sha256"], result=result)
        destination = output / "clips" / f"{i:06d}.json"
        write_new_json(destination, record)
        result_hashes[str(destination.relative_to(output))] = sha256(destination)
        records.append(record)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"Historical validation inference {i + 1}/{len(rows)}", flush=True)
    loop_seconds = time.perf_counter() - before_loop
    if any(sha256(p) != hashes[k] for k, p in inputs.items()):
        raise ValueError("Registered input/code changed during inference")
    for key, digest in registration["provenance"]["models"].items():
        if sha256(ROOT / config[key]) != digest:
            raise ValueError("Checkpoint changed during inference")
    with (output / "predictions.json").open("x", encoding="utf-8") as f:
        json.dump(records, f, allow_nan=False)
    write_new_json(output / "metrics.json", summarize_cascade(records))
    write_new_json(
        output / "runtime.json",
        {
            "scope": (
                "one new inference per existing historical validation clip; "
                "not a paired speedup benchmark"
            ),
            "count": len(rows),
            "first_clip_seconds": wall[0],
            "all_clip_median_seconds": float(np.median(wall)),
            "all_clip_p95_seconds": float(np.quantile(wall, 0.95)),
            "inference_wall_sum_seconds": sum(wall),
            "loop_seconds_including_serialization": loop_seconds,
            "clips_per_second_loop": len(rows) / loop_seconds,
            "model_loading_seconds": load_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_free_after_bytes": torch.cuda.mem_get_info()[0],
            "warmup": (
                "no extra inference; first required validation clip includes cold-kernel effects"
            ),
            "latencies_per_clip_seconds": wall,
            "camera_hour_false_alarms": None,
            "live_cctv_fps": None,
        },
    )
    write_new_json(
        output / "completion.json",
        {
            "complete": True,
            "count": len(rows),
            "result_sha256": result_hashes,
            "output_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        },
    )


def analyze(predictions_dir, output):
    if output.exists():
        raise FileExistsError("Refuse to overwrite comparison")
    complete = json.loads((predictions_dir / "completion.json").read_text())
    if not complete["complete"]:
        raise ValueError("Incomplete validation pass")
    for name, digest in complete["output_sha256"].items():
        if sha256(predictions_dir / name) != digest:
            raise ValueError("Saved comparison input changed")
    for name, digest in complete["result_sha256"].items():
        if sha256(predictions_dir / name) != digest:
            raise ValueError("Saved individual prediction changed")
    reg = json.loads((predictions_dir / "registration.json").read_text())
    if reg["scope"] != "existing_common_model_validation_only_diagnostic_no_optimization":
        raise ValueError("Not an authorized validation diagnostic pass")
    rows = json.loads((predictions_dir / "predictions.json").read_text())
    manifest = [
        json.loads(s)
        for s in (predictions_dir / "comparison-manifest.jsonl").read_text().splitlines()
        if s.strip()
    ]
    compared = compare_review_modes(rows, manifest, reg["provenance"])
    study = json.loads(paths()["study"].read_text())
    before = time.perf_counter()
    diagnostic, details = diagnose(
        rows,
        bootstrap_repetitions=study["source_bootstrap_repetitions"],
        seed=study["source_bootstrap_seed"],
    )
    elapsed = time.perf_counter() - before
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "matched-comparison.json", compared)
    write_new_json(output / "diagnostic.json", diagnostic)
    with (output / "prediction-level-comparison.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    write_new_json(
        output / "registration.json",
        {
            "source_completion_sha256": sha256(predictions_dir / "completion.json"),
            "code_sha256": {
                k: sha256(paths()[k])
                for k in (
                    "runner",
                    "diagnostic_code",
                    "comparison_code",
                    "assessment_code",
                    "study",
                )
            },
            "cpu_diagnostic_seconds_including_bootstrap": elapsed,
            "gpu_inference_in_analysis": False,
            "outputs": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print(
        json.dumps(
            {
                m: {k: v for k, v in s.items() if k not in ("binary_metrics_identical_to_legacy",)}
                for m, s in diagnostic["modes"].items()
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("infer").add_argument("--output", type=Path, required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--predictions-dir", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "infer":
        infer(args.output)
    else:
        analyze(args.predictions_dir, args.output)


if __name__ == "__main__":
    run_cli(main)
