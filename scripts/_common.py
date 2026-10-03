"""CLI utilities; run scripts after installing the project in editable mode."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from surveillance.datasets.common import check_leakage, split_records, write_manifest
from surveillance.datasets.preparation import read_source_map


def run_cli(function) -> None:
    """Render actionable errors without traceback noise for expected input failures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        function()
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"Error: {error}") from None


def write_json(data: dict, path: Path) -> None:
    """Save a standards-compliant JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preparation_parser(description: str) -> argparse.ArgumentParser:
    """Common explicit local dataset and source split options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-map", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST")
    )
    return parser


def finish_preparation(records, args, official: bool = False) -> None:
    """Write only after validating source isolation and report actual class counts."""
    if not official:
        records = split_records(records, args.seed, tuple(args.ratios))
    check_leakage(records)
    write_manifest(records, args.output)
    counts = Counter((r.split, r.label) for r in records)
    print(f"Wrote {len(records)} records to {args.output}; (split,label) counts: {dict(counts)}")


__all__ = ["read_source_map", "run_cli", "write_json", "preparation_parser", "finish_preparation"]
