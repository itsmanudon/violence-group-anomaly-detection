"""Prepare or extract a separately registered real DCSASS Sultani clip baseline."""

import argparse
import json

from _common import run_cli

from surveillance.experiments.sultani_dcsass import extract_generic, prepare_generic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "extract"], required=True)
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_generic() if args.stage == "prepare" else extract_generic(args.max_clips),
            indent=2,
        )
    )


if __name__ == "__main__":
    run_cli(main)
