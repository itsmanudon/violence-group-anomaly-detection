"""One held-out behavior pass after validation checkpoint selection is frozen."""

import argparse
import json
from pathlib import Path

from _common import run_cli

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.dcsass_behavior import BehaviorFeatureDataset
from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.experiments.dcsass_cache import write_json
from surveillance.training.dcsass_behavior import (
    behavior_metrics,
    load_behavior_checkpoint,
    predict_dataset,
)
from surveillance.training.sultani_trainer import seed_everything, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run = args.run
    selection = json.loads((run / "selection.json").read_text())
    checkpoint = run / "best.pt"
    if selection["dry_run"] or not selection["selection_frozen"]:
        raise ValueError(
            "Held-out inference requires a completed non-preflight checkpoint selection"
        )
    if selection["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Selected checkpoint changed")
    destination = run / "held_out"
    if destination.exists():
        raise FileExistsError(
            "Held-out pass already started; analyze its saved predictions instead"
        )
    device = select_device(args.device)
    model, saved = load_behavior_checkpoint(checkpoint, str(device))
    config = saved["config"]
    if saved["manifest_sha256"] != sha256(Path(config["manifest"])):
        raise ValueError("Held-out manifest changed since training")
    seed_everything(config["seed"])
    dataset = BehaviorFeatureDataset(Path(config["manifest"]), Path(config["cache"]), "test")
    destination.mkdir()
    write_json(
        destination / "evaluation_registration.json",
        {
            "checkpoint_sha256": sha256(checkpoint),
            "selected_iteration": saved["iteration"],
            "manifest_sha256": saved["manifest_sha256"],
            "binary_threshold": 0.5,
            "conditional_population": "covered test clips",
            "selection": "validation_macro_f1",
            "total_test_clips": len(dataset.population),
            "covered_clips": len(dataset),
            "no_actor_clips": len(dataset.uncovered),
        },
    )
    records = predict_dataset(model, dataset, config["training"]["batch_size"], device)
    write_json(destination / "predictions.json", {"classes": list(CLASSES), "records": records})
    metrics = behavior_metrics(records)
    correct = sum(r["target"] == r["prediction"] for r in records)
    metrics["population"] = {
        "total_test_clips": len(dataset.population),
        "covered_clips": len(dataset),
        "no_actor_clips": len(dataset.uncovered),
        "coverage": len(dataset) / len(dataset.population),
        "covered_correct": correct,
        "covered_incorrect": len(dataset) - correct,
        "abstained": len(dataset.uncovered),
        "classification_scope": "covered clips only",
    }
    write_json(destination / "metrics.json", metrics)
    errors = [r for r in records if r["target"] != r["prediction"]]
    pairs = {
        "normal_to_anomaly": [r for r in errors if r["target"] == 0],
        "anomaly_to_normal": [r for r in errors if r["prediction"] == 0],
        "fighting_assault": [r for r in errors if {r["target"], r["prediction"]} == {2, 3}],
        "abuse_assault": [r for r in errors if {r["target"], r["prediction"]} == {1, 2}],
        "robbery_errors": [r for r in errors if r["target"] == 4 or r["prediction"] == 4],
        "vandalism_errors": [r for r in errors if r["target"] == 5 or r["prediction"] == 5],
        "high_confidence_errors": sorted(errors, key=lambda r: -r["confidence"])[:25],
    }
    strata = {}
    for name, subset in {
        "one_actor": [r for r in records if r["actor_count"] == 1],
        "two_to_five": [r for r in records if 2 <= r["actor_count"] <= 5],
        "six_or_more": [r for r in records if r["actor_count"] >= 6],
    }.items():
        strata[name] = behavior_metrics(subset) if subset else None
    write_json(
        destination / "error_analysis.json",
        {"examples": pairs, "actor_count_strata": strata, "uncovered_clips": dataset.uncovered},
    )
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in ["binary_anomaly", "per_class"]}, indent=2
        )
    )


if __name__ == "__main__":
    run_cli(main)
