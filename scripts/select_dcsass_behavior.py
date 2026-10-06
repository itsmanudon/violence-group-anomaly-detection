"""Freeze the predeclared initialization comparison using validation only."""

import argparse
from pathlib import Path

from _common import run_cli

from surveillance.experiments.behavior_selection import select_initialization_control
from surveillance.experiments.dcsass_cache import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Initialization comparison selection is already frozen")
    result = select_initialization_control(args.transfer, args.random)
    write_json(args.output, result)
    print(result["selected_run"], result["selection_metric"])


if __name__ == "__main__":
    run_cli(main)
