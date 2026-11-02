"""Future approved-validation baseline or exact paired review replay; never test tuning."""

import argparse
from pathlib import Path

from _common import run_cli

from surveillance.evaluation.indian_frozen import replay_frozen_review, run_frozen_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/surveillance_demo_ucf_v1.yaml"))
    replay = commands.add_parser("review")
    replay.add_argument("--baseline", type=Path, required=True)
    for sub in (run, replay):
        sub.add_argument("--split-dir", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_frozen_validation(
            args.split_dir, args.config, args.output, root=Path(__file__).resolve().parents[1]
        )
    else:
        result = replay_frozen_review(args.baseline, args.split_dir, args.output)
    print(result)


if __name__ == "__main__":
    run_cli(main)
