"""Validate an installed Collective tree and optional manifest; write JSON on failure."""

import argparse
import json
from pathlib import Path

from surveillance.datasets.collective_validation import DatasetValidationError, validate_collective


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--val-sequences", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--actor-count-warning", type=int, default=30)
    args = parser.parse_args()
    try:
        report = validate_collective(
            args.root,
            args.val_sequences,
            args.manifest,
            not args.allow_subset,
            args.actor_count_warning,
        )
        status = 0
    except DatasetValidationError as error:
        report, status = error.report, 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validation {'passed' if report['valid'] else 'failed'}; report: {args.report}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
