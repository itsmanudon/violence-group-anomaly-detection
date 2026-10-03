"""Prepare local Collective frames/annotations into an actor-scene JSONL manifest."""

import argparse
from dataclasses import replace
from pathlib import Path

from _common import run_cli

from surveillance.datasets.collective import (
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    prepare_collective,
    write_actor_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Contains seqNN directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sequences", type=int, nargs="+", default=list(TRAIN_SEQUENCES))
    parser.add_argument("--test-sequences", type=int, nargs="+", default=list(TEST_SEQUENCES))
    parser.add_argument(
        "--val-sequences",
        type=int,
        nargs="*",
        default=[],
        help="Explicit training-source holdout; never taken from test sources",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Only for local smoke data; disables the 32/12 source-count check",
    )
    args = parser.parse_args()
    if len(set(args.val_sequences)) != len(args.val_sequences) or not set(
        args.val_sequences
    ) <= set(args.train_sequences):
        raise ValueError("Validation IDs must be unique IDs from the training source list")
    if set(args.val_sequences) == set(args.train_sequences):
        raise ValueError("Validation holdout must leave at least one training source")
    rows = prepare_collective(
        args.root,
        args.train_sequences,
        args.test_sequences,
        require_full_split=not args.allow_subset,
    )
    validation = {f"seq{s:02d}" for s in args.val_sequences}
    rows = [replace(row, split="val") if row.video_id in validation else row for row in rows]
    write_actor_manifest(rows, args.output)
    counts = {split: sum(r.split == split for r in rows) for split in ("train", "val", "test")}
    print(f"Wrote {len(rows)} actor scenes to {args.output}: {counts}")


if __name__ == "__main__":
    run_cli(main)
