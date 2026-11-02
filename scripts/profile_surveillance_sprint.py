"""Bounded inference-only RTX timing on an existing pinned demonstration asset."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.inference.cascade_review import ReviewPolicy, assess_review
from surveillance.inference.loading import load_surveillance_pipeline

ROOT = Path(__file__).resolve().parents[1]


def distribution(values):
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 10:
        raise ValueError("Use 2..10 bounded engineering repetitions")
    if args.output.exists():
        raise ValueError("Output exists; choose a new version")
    if not torch.cuda.is_available() or "4070" not in torch.cuda.get_device_name(0):
        raise ValueError("This diagnostic requires the local RTX 4070")
    config_path = ROOT / "configs/surveillance_demo_ucf_v1.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["device"] = "cuda:0"
    # Fixed existing curated asset, never chosen using new predictions/labels.
    examples = json.loads((ROOT / config["examples"]).read_text())
    example = next(e for e in examples if e["clip_id"] == "Robbery020_x264_10")
    video = ROOT / example["video"]
    if sha256(video) != example["video_sha256"]:
        raise ValueError("Demonstration media hash mismatch")
    torch.cuda.init()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    before = time.perf_counter()
    pipeline = load_surveillance_pipeline(config, ROOT)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - before
    load_peak = torch.cuda.max_memory_allocated()
    captured = {}

    def capture(name):
        def hook(module, inputs):
            captured[name] = inputs

        return hook

    handles = [
        pipeline.anomaly.model.register_forward_pre_hook(capture("sultani")),
        pipeline.behavior.model.register_forward_pre_hook(capture("actor_transformer")),
    ]
    with torch.inference_mode():
        before = time.perf_counter()
        cold = pipeline.predict_video(video)
        torch.cuda.synchronize()
        first_inference = time.perf_counter() - before
    for handle in handles:
        handle.remove()
    stage_values, totals, policy_times = {}, [], []
    reference = None
    deterministic = True
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            before = time.perf_counter()
            result = pipeline.predict_video(video)
            torch.cuda.synchronize()
            totals.append(time.perf_counter() - before)
            for key, value in result["timings"].items():
                stage_values.setdefault(key, []).append(value)
            signature = [
                result["anomaly"]["segment_scores"],
                [(w["boxes"], w["probabilities"]) for w in result["behavior"]["windows"]],
                result["final_alert"],
            ]
            if reference is None:
                reference = signature
            deterministic &= signature == reference
            before = time.perf_counter()
            assessed = assess_review(result, ReviewPolicy("coverage_consistency"))
            policy_times.append(time.perf_counter() - before)
            assert assessed["final_alert"] == result["final_alert"]
    warm_memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "free_device_bytes_after": torch.cuda.mem_get_info()[0],
    }
    cached = {}
    with torch.inference_mode():
        for name, model in (
            ("sultani", pipeline.anomaly.model),
            ("actor_transformer", pipeline.behavior.model),
        ):
            if name not in captured:
                cached[name] = {"status": "NOT MEASURED: no covered actor window"}
                continue
            inputs = captured[name]
            samples = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                before = time.perf_counter()
                model(*inputs)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - before)
            tensors = list(inputs[0].values()) if isinstance(inputs[0], dict) else [inputs[0]]
            cached[name] = {
                "seconds": distribution(samples),
                "input_shapes": [list(t.shape) for t in tensors],
                "input_sha256": [
                    hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in tensors
                ],
                "scope": "resident actual last-window tensors, excludes extraction and transfers",
            }
    report = {
        "scope": "existing_curated_held_out_demo_runtime_only_no_accuracy_or_selection",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "vram": {
            "total_bytes": total,
            "free_before_loading_bytes": free,
            "loading_peak_allocated_bytes": load_peak,
            **warm_memory,
        },
        "model_loading_seconds": load_seconds,
        "first_inference_seconds": first_inference,
        "warmup": "one complete uncached first inference, five models remain resident",
        "batch_size_c3d": config["c3d_batch_size"],
        "video": {
            "clip_id": example["clip_id"],
            "sha256": sha256(video),
            "metadata": cold["video_metadata"],
            "routed_image_sizes_hw": [w["image_size"] for w in cold["behavior"]["windows"]],
            "actor_sampled_frame_indices": [
                w["sampled_frame_indices"] for w in cold["behavior"]["windows"]
            ],
        },
        "provenance": pipeline.provenance,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)),
        "review_code_sha256": sha256(ROOT / "src/surveillance/inference/cascade_review.py"),
        "uncached_end_to_end_seconds": distribution(totals),
        "clips_per_second_from_median": 1 / float(np.median(totals)),
        "legacy_stage_wall_timer_seconds": {k: distribution(v) for k, v in stage_values.items()},
        "cached_component_forward": cached,
        "review_assessment_cpu_seconds": distribution(policy_times),
        "repeated_scores_boxes_probabilities_alert_exactly_equal": deterministic,
        "indian_metrics": "NOT MEASURED",
        "live_detection_latency": "NOT MEASURED",
        "camera_hour_false_alarms": "NOT MEASURED",
        "notes": (
            "Stage wall timers use existing boundaries; quantiles must not be added. "
            "OS page cache not flushed. No universal video FPS inference."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "model_loading_seconds",
                    "uncached_end_to_end_seconds",
                    "vram",
                    "repeated_scores_boxes_probabilities_alert_exactly_equal",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_cli(main)
