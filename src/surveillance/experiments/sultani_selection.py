"""Freeze completed Sultani validation selection without inspecting held-out scores."""

import math
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.evaluation.evaluate import evaluate
from surveillance.experiments.dcsass_cache import write_json
from surveillance.training.sultani_trainer import load_checkpoint


def freeze_sultani_selection(run: Path, manifest: Path, population: str) -> dict:
    run, manifest = Path(run), Path(manifest)
    destination = run / "selection.json"
    if destination.exists():
        raise FileExistsError("Sultani checkpoint selection is already frozen")
    _, best = load_checkpoint(run / "best.pt")
    _, last = load_checkpoint(run / "last.pt")
    if (
        best.get("dry_run")
        or last.get("dry_run")
        or best.get("selection_metric") != "validation_bag_roc_auc"
        or last.get("selection_metric") != "validation_bag_roc_auc"
        or best["config"] != last["config"]
        or last["epoch"] != last["config"]["training"]["epochs"]
        or not 1 <= best["epoch"] <= last["epoch"]
        or best["best_value"] != last["best_value"]
    ):
        raise ValueError("Sultani training or validation selection is incomplete/preflight")
    metrics = evaluate(run / "best.pt", manifest, split="val", mode="bag", threshold=0.5)
    if metrics["roc_auc"] is None or not math.isclose(
        metrics["roc_auc"], best["best_value"], rel_tol=0, abs_tol=1e-5
    ):
        raise ValueError("Reloaded Sultani validation does not reproduce selected bag ROC-AUC")
    receipt = {
        "schema_version": 1,
        "selection_frozen": True,
        "dry_run": False,
        "checkpoint_sha256": sha256(run / "best.pt"),
        "last_checkpoint_sha256": sha256(run / "last.pt"),
        "manifest_sha256": sha256(manifest),
        "population": population,
        "selected_epoch": best["epoch"],
        "completed_epochs": last["epoch"],
        "selection_metric": "validation_bag_roc_auc",
        "validation": metrics,
        "anomaly_threshold": 0.5,
        "threshold_selection": "fixed prospective default; not tuned on test or demo",
        "held_out_used_for_selection": False,
    }
    write_json(destination, receipt)
    return receipt
