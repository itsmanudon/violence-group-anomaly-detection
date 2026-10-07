"""Evaluate scene group and valid actor classification on a held-out split."""

import argparse
from pathlib import Path

from _common import run_cli, write_json

from surveillance.evaluation.group_activity_metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/group_activity_metrics.json"))
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.manifest, args.split, args.device)
    write_json(result, args.output)
    print(
        f"Group accuracy={result['group']['accuracy']:.4f}; "
        f"actor accuracy={result['actor']['accuracy']:.4f}; output={args.output}"
    )


if __name__ == "__main__":
    run_cli(main)
