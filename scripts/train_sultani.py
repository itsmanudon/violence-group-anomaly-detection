"""Train Sultani MIL on explicit precomputed feature bags."""

import argparse
from pathlib import Path

from _common import run_cli

from surveillance.config import load_config
from surveillance.training.sultani_trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sultani_mil.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/sultani"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    checkpoint = train(load_config(args.config), args.manifest, args.output, args.resume)
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    run_cli(main)
