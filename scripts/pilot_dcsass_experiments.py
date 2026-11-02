"""Indian pilot v2 (DCSASS_PROTOCOL.md): learning curve over added DCSASS train sources.

Settings are fixed by the committed protocol. Validation only; test features do not exist
when this runs.
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from _common import run_cli

from surveillance.datasets.indian_split import verify_split
from surveillance.evaluation.indian_binary import evaluate_binary
from surveillance.training.indian_binary import (
    calibrate_binary_checkpoint,
    load_binary_checkpoint,
    train_binary,
)

SPLIT = Path("manifests/v2-pilot/split_combined_v1")
CACHE = Path("runs/indian_pilot/features/c3d_combined_trainval_v1")
OUT = Path("runs/indian_pilot/experiments_v2")
TABLE = Path("manifests/v2-pilot/split_v1.csv")
TRAIN = dict(
    steps=600,
    learning_rate=1e-3,
    batch_size=16,
    class_weighting="balanced",
    sampling="uniform",
    patience=100,
    precision="fp32",
)
SEEDS = (7, 8, 9)
DEPLOYED_SEED = 7
FRACTIONS = (0, 10, 25, 50, 100)
SUBSET_SEED = 7
MIN_RECALL = 0.8
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def train_groups(rows):
    groups = sorted({r.source_video_id for r in rows if r.split == "train"})
    indian = [g for g in groups if g.startswith("indian:")]
    dcsass = [g for g in groups if g.startswith("dcsass:")]
    strata: dict[str, list] = {}
    for g in dcsass:
        strata.setdefault(re.match(r"dcsass:([a-z]+)", g).group(1), []).append(g)
    rng = random.Random(SUBSET_SEED)
    for members in strata.values():
        rng.shuffle(members)
    subsets = {
        fraction: sorted(
            g
            for members in strata.values()
            for g in members[: math.ceil(fraction / 100 * len(members))]
        )
        for fraction in FRACTIONS
    }
    return indian, dcsass, subsets


def cmd_train(args):
    rows, _ = verify_split(SPLIT)
    indian, dcsass, subsets = train_groups(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    plan = OUT / "subsets.json"
    if not plan.exists():
        plan.write_text(
            json.dumps({"indian": indian, **{str(k): v for k, v in subsets.items()}}, indent=1)
        )
    for seed in args.seeds:
        for fraction in FRACTIONS:
            name = f"curve_d{fraction:03d}_s{seed}"
            if not (OUT / name / "best.pt").exists():
                train_binary(
                    SPLIT,
                    CACHE,
                    OUT / name,
                    seed=seed,
                    train_groups=indian + subsets[fraction],
                    **TRAIN,
                )
            print("done", name, flush=True)
        name = f"dcsass_only_s{seed}"
        if not (OUT / name / "best.pt").exists():
            train_binary(SPLIT, CACHE, OUT / name, seed=seed, train_groups=dcsass, **TRAIN)
        print("done", name, flush=True)


def cmd_calibrate(args):
    output = OUT / f"curve_d100_s{DEPLOYED_SEED}" / "calibration"
    if not output.exists():
        result = calibrate_binary_checkpoint(
            SPLIT,
            OUT / f"curve_d100_s{DEPLOYED_SEED}" / "best.pt",
            output,
            min_recall=MIN_RECALL,
            candidates=THRESHOLDS,
        )
        print("threshold", result["selected_threshold"])


def cmd_summarize(args):
    import csv

    rows, receipt = verify_split(SPLIT)
    with TABLE.open(encoding="utf-8") as stream:
        geography = {r["clip"]: r["geography"] for r in csv.DictReader(stream)}
    dataset = {r.video_id: r.dataset for r in rows}
    calibration = json.loads(
        (OUT / f"curve_d100_s{DEPLOYED_SEED}/calibration/calibration.json").read_text()
    )
    threshold = calibration["selected_threshold"]
    runs = {}
    for run in sorted(p for p in OUT.iterdir() if (p / "best.pt").exists()):
        _, saved = load_binary_checkpoint(run / "best.pt")
        preds = saved["validation_predictions"]

        def metrics(keep, preds=preds):
            chosen = [p for p in preds if keep(p["clip_id"])]
            report = evaluate_binary(
                [p["label"] for p in chosen], [p["violence_score"] for p in chosen]
            )
            return {k: report[k] for k in ("count", "pr_auc", "roc_auc")}

        indian_nv = [p for p in preds if geography.get(p["clip_id"]) == "documented_indian"]
        runs[run.name] = {
            "best_step": max(
                saved["history"], key=lambda h: (h["validation"]["pr_auc"], h["validation"]["f1"])
            )["step"],
            "combined_val": metrics(lambda k: True),
            "dcsass_val": metrics(lambda k: dataset[k] == "dcsass"),
            "indian_val": metrics(lambda k: dataset[k] == "indian"),
            "documented_indian_nv_fpr_at_threshold": float(
                np.mean([p["violence_score"] >= threshold for p in indian_nv])
            ),
        }
    curve = {}
    for fraction in FRACTIONS:
        values = [runs[f"curve_d{fraction:03d}_s{s}"] for s in SEEDS]
        curve[str(fraction)] = {
            f"{part}_{m}": [
                float(np.mean([v[part][m] for v in values])),
                float(np.std([v[part][m] for v in values])),
            ]
            for part in ("dcsass_val", "indian_val")
            for m in ("pr_auc", "roc_auc")
        }
    result = {
        "scope": "validation_only",
        "split_freeze_hash": receipt["freeze_hash"],
        "deployed": f"curve_d100_s{DEPLOYED_SEED}",
        "threshold": threshold,
        "runs": runs,
        "learning_curve_val": curve,
    }
    (OUT.parent / "validation_results_v2.json").write_text(json.dumps(result, indent=2) + "\n")
    for k, v in curve.items():
        print(k, {m: [round(x, 3) for x in val] for m, val in v.items()})
    print("threshold", threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    commands.add_parser("calibrate")
    commands.add_parser("summarize")
    args = parser.parse_args()
    {"train": cmd_train, "calibrate": cmd_calibrate, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    torch.set_num_threads(3)
    run_cli(main)
