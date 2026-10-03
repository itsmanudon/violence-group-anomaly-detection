"""Compare the same Actor-Transformer checkpoint with GT and detected actor boxes."""

import argparse
from pathlib import Path

from _common import run_cli, write_json

from surveillance.detection.config import load_detection_config
from surveillance.evaluation.box_robustness import evaluate_box_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="GT manifest with GT features")
    parser.add_argument("--detections", type=Path, required=True, help="Detected-feature JSONL")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--config", type=Path, default=Path("configs/person_detector.yaml"))
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/box_robustness.json"))
    args = parser.parse_args()
    settings = load_detection_config(args.config)
    threshold = (
        settings["matching"]["iou_threshold"] if args.iou_threshold is None else args.iou_threshold
    )
    result = evaluate_box_robustness(
        args.checkpoint,
        args.manifest,
        args.detections,
        args.split,
        threshold,
        args.device,
        args.attention,
    )
    write_json(result, args.output)
    print(f"Compared GT paper baseline and detected-box adaptation; output={args.output}")


if __name__ == "__main__":
    run_cli(main)
