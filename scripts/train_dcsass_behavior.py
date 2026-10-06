"""Train the transferred six-class RGB Actor-Transformer using group labels only."""

import argparse
from pathlib import Path

import yaml
from _common import run_cli

from surveillance.training.dcsass_behavior import train_behavior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/dcsass_human_rgb_detected_v1.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--preflight-iterations", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    print(
        train_behavior(
            config, args.output, cache=args.cache, preflight_iterations=args.preflight_iterations
        )
    )


if __name__ == "__main__":
    run_cli(main)
