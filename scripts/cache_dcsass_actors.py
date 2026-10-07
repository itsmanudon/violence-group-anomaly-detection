"""Precompute fixed Faster R-CNN detections or I3D actor features on DCSASS."""

import argparse
import json
from pathlib import Path

from _common import run_cli

from surveillance.experiments.dcsass_cache import run_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["detect", "extract"], required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/dcsass_human_centric_v1_final.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path("checkpoints/external/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"),
    )
    parser.add_argument(
        "--i3d-checkpoint", type=Path, default=Path("checkpoints/i3d_mixed4f_collective_rgb_v1.pt")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args()
    report = run_cache(
        args.manifest,
        args.output,
        args.stage,
        args.detector_checkpoint,
        args.i3d_checkpoint,
        args.device,
        args.max_clips,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "by_split"}, indent=2))


if __name__ == "__main__":
    run_cli(main)
