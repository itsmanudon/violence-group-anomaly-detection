"""Create deduplicated review queues using verified existing media receipts."""

import argparse
import json
from pathlib import Path

from _common import run_cli

from surveillance.datasets.indian_review import screen_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = screen_review(args.audit, args.csv, args.catalog, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "deduplicated_queue_records",
                    "first_priority_records",
                    "flag_counts_not_disjoint",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_cli(main)
