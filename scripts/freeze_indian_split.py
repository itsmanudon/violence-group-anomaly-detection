"""Inventory historical sources, or freeze a reviewed source-safe Indian split."""

import argparse
import json
from pathlib import Path

from _common import run_cli

from surveillance.datasets.indian_domain import benchmark_inventory, write_new_json
from surveillance.datasets.indian_split import freeze_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--manifests", type=Path, nargs="+", required=True)
    inventory.add_argument("--evidence", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--import-path", type=Path, required=True)
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--review", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=7)
    freeze.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    args = parser.parse_args()
    if args.command == "inventory":
        result = benchmark_inventory(args.manifests, args.evidence)
        write_new_json(args.output, result)
    else:
        result = freeze_split(
            args.import_path,
            args.inventory,
            args.review,
            args.output,
            seed=args.seed,
            ratios=args.ratios,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_cli(main)
