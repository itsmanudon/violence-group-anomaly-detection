"""Jointly resplit related datasets by canonical source identity."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

from _common import run_cli

from surveillance.datasets.common import (
    check_leakage,
    read_manifest,
    resolve_path,
    split_records,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    args = parser.parse_args()
    rows = []
    for manifest in args.manifests:
        for row in read_manifest(manifest):

            def relocated(value: str) -> str:
                return Path(
                    os.path.relpath(resolve_path(value, manifest), args.output.resolve().parent)
                ).as_posix()

            rows.append(
                replace(
                    row,
                    path=relocated(row.path),
                    feature_path=relocated(row.feature_path) if row.feature_path else None,
                )
            )
    keys = [(r.dataset, r.video_id) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate dataset/video_id across input manifests")
    rows = split_records(rows, args.seed, tuple(args.ratios))
    check_leakage(rows)
    write_manifest(rows, args.output)
    print(f"Wrote joint source-aware split for {len(rows)} videos to {args.output}")


if __name__ == "__main__":
    run_cli(main)
