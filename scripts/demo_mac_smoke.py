"""Frozen component repeat/finite checks and genuine live/cached render smoke."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from _common import run_cli

from surveillance.demo_profile import load_professor_profile
from surveillance.demo_rendering import reference_gallery, timeline_figure
from surveillance.demo_service import DemoService
from surveillance.inference.loading import load_surveillance_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--cached-example")
    parser.add_argument(
        "--probe-mps",
        action="store_true",
        help="Try each frozen component on MPS, restore CPU and record failures",
    )
    parser.add_argument("--local-assets", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/demo_mac/smoke.json")
    args = parser.parse_args()
    config = load_professor_profile(ROOT, args.local_assets)
    pipeline = load_surveillance_pipeline(config, ROOT)
    device = pipeline.behavior.device
    boxes = torch.tensor([[[0.1, 0.1, 0.8, 0.9]]], device=device)
    valid = torch.ones(1, 1, dtype=torch.bool, device=device)
    frames = torch.zeros(1, 10, 3, 480, 720, device=device)
    components = {
        "sultani": lambda: pipeline.anomaly.model(torch.zeros(32, 4096, device=device)),
        "c3d": lambda: pipeline.anomaly.extractor.model(
            torch.zeros(1, 3, 16, 112, 112, device=device)
        ),
        "faster_rcnn": lambda: (
            pipeline.behavior.detector.detect(np.zeros((480, 720, 3), dtype=np.uint8)).scores
        ),
        "i3d": lambda: pipeline.behavior.extractor(
            frames.to(device), boxes.to(device), valid.to(device)
        ),
        "actor_transformer": lambda: pipeline.behavior.model(
            {
                "rgb_features": torch.zeros(1, 1, 20800, device=device),
                "actor_boxes": boxes.to(device),
                "actor_valid_mask": valid.to(device),
            }
        )["group_logits"],
    }
    report = {"components": [], "device_policy": "CPU on Mac; MPS unvalidated"}
    with torch.inference_mode():
        for name, forward in components.items():
            times, values = [], []
            for _ in range(2):
                before = time.perf_counter()
                value = forward().cpu()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite output: {name}")
                times.append(time.perf_counter() - before)
                values.append(value)
            report["components"].append(
                {
                    "component": name,
                    "status": "PASS",
                    "device": pipeline.component_devices[name],
                    "latency_seconds": times,
                    "repeat_equal": torch.equal(*values),
                    "fallback_reason": "MPS not validated",
                }
            )
    if args.probe_mps:
        report["mps_probes"] = []
        models = {
            "sultani": pipeline.anomaly.model,
            "c3d": pipeline.anomaly.extractor.model,
            "faster_rcnn": pipeline.behavior.detector.model,
            "i3d": pipeline.behavior.extractor,
            "actor_transformer": pipeline.behavior.model,
        }
        for name, model in models.items():
            probe = {"component": name, "chosen_device": "cpu"}
            try:
                if not torch.backends.mps.is_available():
                    raise RuntimeError(
                        "MPS unavailable in this process; run outside restricted sandbox"
                    )
                device = torch.device("mps")
                model.to(device)
                if name == "faster_rcnn":
                    pipeline.behavior.detector.device = device
                times = []
                with torch.inference_mode():
                    for _ in range(2):
                        before = time.perf_counter()
                        value = components[name]().cpu()
                        if not torch.isfinite(value).all():
                            raise ValueError("Nonfinite MPS output")
                        times.append(time.perf_counter() - before)
                probe.update(
                    status="PASS",
                    latency_seconds=times,
                    note="Synthetic probe only; live pipeline stays on validated CPU policy",
                )
            except (RuntimeError, ValueError, NotImplementedError) as error:
                probe.update(status="WARN", fallback_reason=str(error))
            finally:
                device = torch.device("cpu")
                model.to(device)
                pipeline.behavior.detector.device = device
            report["mps_probes"].append(probe)
    service = DemoService(config, ROOT)
    service.pipeline = pipeline
    result = service.analyze(str(args.clip), "Upload a video", False)
    figure = timeline_figure(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"))
    gallery = reference_gallery(result)
    report["live"] = {"result": result, "rendered_reference_frames": len(gallery)}
    if args.cached_example:
        report["cached"] = service.analyze(None, args.cached_example, True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(args.output)


if __name__ == "__main__":
    run_cli(main)
