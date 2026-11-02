"""Compare review assessments using complete matched validation predictions only."""

import argparse
import json
from pathlib import Path

import yaml
from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.evaluation.cascade_review import compare_review_modes
from surveillance.inference.provenance import pipeline_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON list, split=val")
    parser.add_argument("--manifest", type=Path, required=True, help="All-split JSONL provenance")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_review_modes(
        json.loads(args.predictions.read_text()),
        [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()],
        pipeline_identity(yaml.safe_load(args.config.read_text())),
    )
    report["input_sha256"] = {
        key: sha256(getattr(args, key)) for key in ("predictions", "manifest", "config")
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run_cli(main)
