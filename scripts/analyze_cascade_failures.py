"""Describe immutable historical cascade errors; never tune on held-out labels."""

import argparse
import json
from pathlib import Path

from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.evaluation.cascade_failures import attribute_failures, load_registered_cascade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, _ = load_registered_cascade(args.registered_run)
    report = attribute_failures(rows)
    report["input_sha256"] = {
        name: sha256(args.registered_run / name)
        for name in ("registration.json", "progress.json", "completion.json", "metrics.json")
    }
    report["analyzer_sha256"] = sha256(
        Path(__file__).resolve().parents[1] / "src/surveillance/evaluation/cascade_failures.py"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "failures.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    run_cli(main)
