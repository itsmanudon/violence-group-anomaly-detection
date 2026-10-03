"""Predict group and actor classes from local scene manifest inputs."""

import argparse
from pathlib import Path

from _common import run_cli, write_json

from surveillance.inference.group_activity_pipeline import GroupActivityPipeline
from surveillance.training.actor_transformer_trainer import load_checkpoint, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/group_activity.json"))
    args = parser.parse_args()
    system, _ = load_checkpoint(args.checkpoint, select_device(args.device))
    scenes = GroupActivityPipeline(system).predict_manifest(
        args.manifest, None if args.split == "all" else args.split, return_attention=args.attention
    )
    write_json({"scenes": scenes, "box_coordinates": "normalized_xyxy"}, args.output)
    print(f"Predicted {len(scenes)} scenes; output={args.output}")


if __name__ == "__main__":
    run_cli(main)
