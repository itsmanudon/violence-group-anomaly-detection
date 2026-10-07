"""Verify orchestration using artificial features, never benchmark results."""

import argparse
from pathlib import Path

from surveillance.experiments.synthetic import smoke_workflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New ignored output directory")
    args = parser.parse_args()
    print(smoke_workflow(args.output))
