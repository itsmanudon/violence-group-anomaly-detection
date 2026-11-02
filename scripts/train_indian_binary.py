"""Cache pinned frozen C3D features or train an isolated Indian binary baseline."""

import argparse
from pathlib import Path

from _common import run_cli

from surveillance.training.indian_binary import (
    cache_c3d,
    calibrate_binary_checkpoint,
    profile_binary_device,
    train_binary,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile", help="Bounded synthetic-head RTX engineering profile")
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    profile.add_argument("--steps", type=int, default=10)
    profile.add_argument("--precision", choices=["fp32", "amp_fp16"], default="fp32")
    calibration = commands.add_parser(
        "calibrate", help="Validation-only selected-checkpoint calibration"
    )
    calibration.add_argument("--split-dir", type=Path, required=True)
    calibration.add_argument("--checkpoint", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--min-recall", type=float, default=0.9)
    calibration.add_argument("--thresholds", type=float, nargs="+", default=None)
    cache = commands.add_parser("cache")
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--splits", nargs="+", default=["train", "val"])
    cache.add_argument(
        "--non-strict-frame-count",
        action="store_true",
        help="use only frames that decode when a container header overstates its count",
    )
    cache.add_argument("--test-protocol", type=Path, help="committed EVAL_PROTOCOL.md (test only)")
    train = commands.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--resume", type=Path)
    train.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    train.add_argument("--sampling", choices=["uniform", "balanced"], default="uniform")
    train.add_argument("--patience", type=int, default=0)
    train.add_argument("--precision", choices=["fp32", "amp_fp16"], default="fp32")
    train.add_argument("--accumulation-steps", type=int, default=1)
    for sub in (cache, train):
        sub.add_argument("--split-dir", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--batch-size", type=int, default=2 if sub is cache else 8)
    args = parser.parse_args()
    if args.command == "profile":
        report = profile_binary_device(
            args.output, batch_sizes=args.batch_sizes, steps=args.steps, precision=args.precision
        )
        print(report)
    elif args.command == "calibrate":
        report = calibrate_binary_checkpoint(
            args.split_dir,
            args.checkpoint,
            args.output,
            min_recall=args.min_recall,
            candidates=args.thresholds,
        )
        print(report)
    elif args.command == "cache":
        cache_c3d(
            args.split_dir,
            args.checkpoint,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            splits=tuple(args.splits),
            strict_frame_count=not args.non_strict_frame_count,
            test_protocol=args.test_protocol,
        )
    else:
        print(
            train_binary(
                args.split_dir,
                args.cache,
                args.output,
                steps=args.steps,
                seed=args.seed,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                device=args.device,
                resume=args.resume,
                class_weighting=args.class_weighting,
                sampling=args.sampling,
                patience=args.patience,
                precision=args.precision,
                accumulation_steps=args.accumulation_steps,
            )
        )


if __name__ == "__main__":
    run_cli(main)
