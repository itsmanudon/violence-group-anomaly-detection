"""Audit exported Collective predictions without reading/copying source images."""

import argparse
from pathlib import Path

from surveillance.datasets.collective import CLASSES
from surveillance.experiments.reporting import analyze_predictions, read_json, write_error_report


def _predictions(path, kind):
    value = read_json(path)
    if isinstance(value, dict):
        value = value.get(f"{kind}_predictions", value.get("predictions"))
    if not isinstance(value, list) or not all(isinstance(scene, dict) for scene in value):
        raise ValueError(f"{path}: expected a prediction list or an exported prediction object")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", "--ground-truth", dest="gt", type=Path, required=True)
    parser.add_argument("--detected", type=Path)
    parser.add_argument("--classes", nargs="+", default=list(CLASSES))
    parser.add_argument("--high-confidence", type=float, default=0.8)
    parser.add_argument("--low-coverage", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze_predictions(
            _predictions(args.gt, "ground_truth"),
            _predictions(args.detected, "detected") if args.detected else None,
            args.classes,
            args.high_confidence,
            args.low_coverage,
        )
        paths = write_error_report(report, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(paths["json"])


if __name__ == "__main__":
    main()
