"""Propose provisional groups without creating a training-approved dataset."""

import argparse
import json
from pathlib import Path

from _common import run_cli

from surveillance.datasets.indian_development import propose_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = propose_development(args.audit, args.csv, args.catalog, args.output, seed=args.seed)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "quality_candidates",
                    "quality_holds_unchanged",
                    "proposed_records",
                    "counts",
                    "training_approved",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_cli(main)
