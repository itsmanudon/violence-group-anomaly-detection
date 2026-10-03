"""Prepare installed DCSASS; no data download."""

from pathlib import Path

from _common import finish_preparation, preparation_parser, read_source_map, run_cli

from surveillance.datasets.dcsass import prepare_dcsass


def main() -> None:
    parser = preparation_parser(__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Normalized CSV: path,label[,source_video_id,anomaly_type]",
    )
    args = parser.parse_args()
    rows = prepare_dcsass(args.root, args.labels, args.output, read_source_map(args.source_map))
    finish_preparation(rows, args)


if __name__ == "__main__":
    run_cli(main)
