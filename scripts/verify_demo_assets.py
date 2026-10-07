"""Verify frozen local identities and load every deployed model without scoring test data."""

import argparse
import json
from pathlib import Path

import yaml
from _common import run_cli

from surveillance.inference.loading import load_surveillance_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surveillance_demo.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pipeline = load_surveillance_pipeline(config, ROOT)
    print(
        json.dumps(
            {
                "verified": True,
                "device": str(pipeline.behavior.device),
                "provenance": pipeline.provenance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_cli(main)
