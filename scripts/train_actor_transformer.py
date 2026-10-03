"""Train the local Actor Transformer group activity baseline."""

import argparse
from pathlib import Path

from _common import run_cli

from surveillance.actor_config import load_actor_config
from surveillance.training.actor_transformer_trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/actor_transformer.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/actor_transformer"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    checkpoint = train(load_actor_config(args.config), args.manifest, args.output, args.resume)
    print(f"Saved Actor Transformer checkpoint: {checkpoint}")


if __name__ == "__main__":
    run_cli(main)
