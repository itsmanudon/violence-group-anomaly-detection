"""Single registered held-out pass from a completed frozen Sultani selection."""

import argparse
from pathlib import Path

from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.evaluation.evaluate import evaluate
from surveillance.experiments.dcsass_cache import write_json
from surveillance.experiments.sultani_selection import validate_evaluation_policy
from surveillance.inference.provenance import verify_selected_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["bag", "frame"], default="bag")
    parser.add_argument(
        "--projection", choices=["repeat", "interpolate", "c3d_units"], default=None
    )
    args = parser.parse_args()
    checkpoint = args.run / "best.pt"
    selected = verify_selected_checkpoint(checkpoint)
    projection = validate_evaluation_policy(selected, args.mode, args.projection)
    if (
        selected.get("protocol_sha256")
        and sha256(args.run.parent / "protocol.json") != selected["protocol_sha256"]
    ):
        raise ValueError("Frozen scientific protocol changed after validation selection")
    if sha256(args.manifest) != selected["manifest_sha256"]:
        raise ValueError("Sultani manifest changed after validation selection")
    destination = args.run / "held_out"
    if destination.exists():
        raise FileExistsError("Held-out pass already started; analyze saved predictions")
    destination.mkdir()
    write_json(
        destination / "evaluation_registration.json",
        {
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "manifest_sha256": selected["manifest_sha256"],
            "population": selected["population"],
            "selected_epoch": selected["selected_epoch"],
            "mode": args.mode,
            "projection": projection if args.mode == "frame" else None,
            "threshold": 0.5,
            "frame_ground_truth_claim": args.mode == "frame",
        },
    )
    metrics = evaluate(
        checkpoint,
        args.manifest,
        split="test",
        mode=args.mode,
        threshold=0.5,
        projection=projection,
        include_predictions=True,
    )
    predictions = metrics.pop("predictions")
    write_json(destination / "predictions.json", {"records": predictions})
    write_json(destination / "metrics.json", metrics)
    print("Held-out", args.mode, "ROC-AUC", metrics["roc_auc"], "F1", metrics["f1"])


if __name__ == "__main__":
    run_cli(main)
