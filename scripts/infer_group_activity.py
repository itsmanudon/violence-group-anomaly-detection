"""Predict group and actor classes from local scene manifest inputs."""

import argparse
from pathlib import Path

from _common import run_cli, write_json

from surveillance.detection.config import load_detection_config
from surveillance.inference.detected_group_activity import DetectedGroupActivityPipeline
from surveillance.inference.group_activity_pipeline import GroupActivityPipeline
from surveillance.training.actor_transformer_trainer import load_checkpoint, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--box-source", choices=["ground_truth", "detections"], default="ground_truth"
    )
    parser.add_argument("--detections", type=Path)
    parser.add_argument(
        "--detector-config", type=Path, default=Path("configs/person_detector.yaml")
    )
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/group_activity.json"))
    args = parser.parse_args()
    if (args.box_source == "detections") != (args.detections is not None):
        raise ValueError("--detections is required exactly when --box-source detections")
    system, _ = load_checkpoint(args.checkpoint, select_device(args.device))
    split = None if args.split == "all" else args.split
    if args.box_source == "detections":
        settings = load_detection_config(args.detector_config)
        threshold = (
            settings["matching"]["iou_threshold"]
            if args.iou_threshold is None
            else args.iou_threshold
        )
        scenes = DetectedGroupActivityPipeline(system).predict_manifest(
            args.manifest, args.detections, split, threshold, args.attention
        )
    else:
        scenes = GroupActivityPipeline(system).predict_manifest(
            args.manifest, split, return_attention=args.attention
        )
    write_json(
        {"scenes": scenes, "box_coordinates": "normalized_xyxy", "box_source": args.box_source},
        args.output,
    )
    print(f"Processed {len(scenes)} scenes; box_source={args.box_source}; output={args.output}")


if __name__ == "__main__":
    run_cli(main)
