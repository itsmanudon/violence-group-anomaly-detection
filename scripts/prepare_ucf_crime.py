"""Prepare installed UCF-Crime with optional official split and annotations."""

from pathlib import Path

from _common import finish_preparation, preparation_parser, read_source_map, run_cli

from surveillance.datasets.common import split_records
from surveillance.datasets.ucf_crime import (
    apply_frame_annotations,
    apply_official_splits,
    prepare_ucf_crime,
)


def main() -> None:
    parser = preparation_parser(__doc__)
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Source fraction held out from official train list",
    )
    args = parser.parse_args()
    if bool(args.train_list) != bool(args.test_list):
        raise ValueError("Supply both --train-list and --test-list")
    rows = prepare_ucf_crime(args.root, args.output, read_source_map(args.source_map))
    official = args.train_list is not None
    if official:
        if not 0 <= args.val_fraction < 1:
            raise ValueError("val-fraction must be in [0,1)")
        rows = apply_official_splits(rows, args.train_list, args.test_list)
        training = split_records(
            [r for r in rows if r.split == "train"],
            args.seed,
            (1 - args.val_fraction, args.val_fraction, 0),
        )
        rows = training + [r for r in rows if r.split == "test"]
    if args.annotations:
        rows = apply_frame_annotations(rows, args.annotations)
    finish_preparation(rows, args, official)


if __name__ == "__main__":
    run_cli(main)
