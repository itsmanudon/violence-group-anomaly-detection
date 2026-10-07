"""Create provenance-bound classroom caches using actual local model inference."""

import argparse
from pathlib import Path

import yaml
from _common import run_cli

from surveillance.experiments.dcsass_cache import write_json
from surveillance.experiments.demo_cache import cache_examples

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surveillance_demo.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Demo generation receipt exists; use it or choose a new receipt path")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = cache_examples(config, ROOT)
    write_json(args.output, result)
    print("Generated", result["generated"], "reused", result["reused"], "example caches")


if __name__ == "__main__":
    run_cli(main)
