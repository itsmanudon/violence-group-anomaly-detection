"""Match cached absolute detector boxes to annotated GT actors offline."""

import argparse
from pathlib import Path

from _common import run_cli, write_json

from surveillance.detection.config import load_detection_config
from surveillance.detection.matching import match_manifest


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
    scenes = match_manifest(
        args.manifest,
        args.detections,
        None if args.split == "all" else args.split,
        threshold,
    )
    write_json({"scenes": scenes, "ignore_index": -100}, args.output)
    print(f"Matched {len(scenes)} scenes; saved {args.output}")


if __name__ == "__main__":
    run_cli(main)
