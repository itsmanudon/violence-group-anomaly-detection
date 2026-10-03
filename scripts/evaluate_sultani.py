"""Evaluate held-out bags or temporally annotated video frames."""

import argparse
from pathlib import Path

from _common import run_cli, write_json

from surveillance.evaluation.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--mode", choices=["bag", "frame"], default="bag")
    parser.add_argument("--projection", choices=["repeat", "interpolate"], default="repeat")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics.json"))
    args = parser.parse_args()
    metrics = evaluate(
        args.checkpoint, args.manifest, args.split, args.mode, args.threshold, args.projection
    )
    write_json(metrics, args.output)
    print(
        f"{metrics['explanation']}\nROC-AUC={metrics['roc_auc']}; "
        f"F1={metrics['f1']:.4f}; output={args.output}"
    )


if __name__ == "__main__":
    run_cli(main)
