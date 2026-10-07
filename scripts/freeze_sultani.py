"""Verify completed real Sultani training and freeze validation selection."""

import argparse
from pathlib import Path

from _common import run_cli

from surveillance.experiments.sultani_selection import freeze_sultani_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--population", required=True)
    args = parser.parse_args()
    result = freeze_sultani_selection(args.run, args.manifest, args.population)
    print(
        "Frozen selected epoch",
        result["selected_epoch"],
        "val bag AUC",
        result["validation"]["roc_auc"],
    )


if __name__ == "__main__":
    run_cli(main)
