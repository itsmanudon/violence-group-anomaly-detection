"""Evaluate cached person detections against annotated actor boxes."""

import argparse
from pathlib import Path

from _common import run_cli, write_json

from surveillance.detection.config import load_detection_config
from surveillance.evaluation.detection_metrics import evaluate_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/person_detector.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_detection_config(args.config)
    threshold = (
        config["matching"]["iou_threshold"] if args.iou_threshold is None else args.iou_threshold
    )
    result = evaluate_detections(
        args.manifest,
        args.detections,
        None if args.split == "all" else args.split,
        threshold,
    )
    write_json(result, args.output)
    print(f"Evaluated {result['frame_count']} scenes; saved {args.output}")


if __name__ == "__main__":
    run_cli(main)
