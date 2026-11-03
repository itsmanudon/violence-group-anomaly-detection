"""Indian pilot v3 (V3_PROTOCOL.md): train both arms, pick the deployed arm, calibrate.

Settings are fixed by the committed protocol. Validation only; IITB test features do not
exist when this runs.
"""

import argparse
import csv
import json
import math
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

SPLIT = Path("manifests/v3-pilot/split_v3")
CACHE = Path("runs/indian_pilot/features/c3d_v3_trainval")
OUT = Path("runs/indian_pilot/experiments_v3")
TABLE = Path("manifests/v2-pilot/split_v1.csv")
BATCH = 16
TRAIN = dict(
    learning_rate=1e-3,
    batch_size=BATCH,
    class_weighting="balanced",
    sampling="uniform",
    precision="fp32",
)
SEEDS = (7, 8, 9)
DEPLOYED_SEED = 7
ARMS = ("v3_cctv", "v3_all")
MIN_RECALL = 0.8
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def arm_groups(rows):
    groups = sorted({r.source_video_id for r in rows if r.split == "train"})
    return {"v3_cctv": [g for g in groups if not g.startswith("xd:")], "v3_all": groups}


def schedule(rows, groups):
    """Two passes over the arm's training clips (protocol formula)."""
    chosen = set(groups)
    n_train = sum(r.split == "train" and r.source_video_id in chosen for r in rows)
    steps = 100 * math.ceil(2 * n_train / BATCH / 100)
    return n_train, steps, int(round(steps / 6, -1))


def cmd_train(args):
    rows, _ = verify_split(SPLIT)
    groups = arm_groups(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        arm: dict(zip(("n_train", "steps", "patience"), schedule(rows, g), strict=True))
        for arm, g in groups.items()
    }
    (OUT / "schedule.json").write_text(json.dumps(plan, indent=1) + "\n")
    for arm in args.arms:
        for seed in args.seeds:
            name = f"{arm}_s{seed}"
            if not (OUT / name / "best.pt").exists():
                train_binary(
                    SPLIT,
                    CACHE,
                    OUT / name,
                    seed=seed,
                    train_groups=groups[arm],
                    steps=plan[arm]["steps"],
                    patience=plan[arm]["patience"],
                    **TRAIN,
                )
            print("done", name, plan[arm], flush=True)


def validation_metrics(run, dataset, keep):
    _, saved = load_binary_checkpoint(run / "best.pt")
    chosen = [p for p in saved["validation_predictions"] if keep(dataset[p["clip_id"]])]
    report = evaluate_binary([p["label"] for p in chosen], [p["violence_score"] for p in chosen])
    return {k: report[k] for k in ("count", "pr_auc", "roc_auc")}


def cmd_select(args):
    """Deployed arm: higher seed-mean PR-AUC on non-XD validation clips."""
    target = OUT / "deployed.json"
    if target.exists():
        raise FileExistsError(f"Refuse to overwrite {target}")
    rows, receipt = verify_split(SPLIT)
    dataset = {r.video_id: r.dataset for r in rows}
    means = {
        arm: float(
            np.mean(
                [
                    validation_metrics(OUT / f"{arm}_s{s}", dataset, lambda d: d != "xd")["pr_auc"]
                    for s in SEEDS
                ]
            )
        )
        for arm in ARMS
    }
    deployed = max(ARMS, key=lambda arm: (means[arm], arm == "v3_cctv"))
    result = {
        "rule": "higher seed-mean PR-AUC on non-XD validation clips (V3_PROTOCOL.md)",
        "non_xd_val_pr_auc_seed_mean": means,
        "deployed_arm": deployed,
        "deployed_run": f"{deployed}_s{DEPLOYED_SEED}",
        "split_freeze_hash": receipt["freeze_hash"],
    }
    target.write_text(json.dumps(result, indent=1) + "\n")
    print(result)


def cmd_calibrate(args):
    run = json.loads((OUT / "deployed.json").read_text())["deployed_run"]
    output = OUT / run / "calibration"
    if not output.exists():
        result = calibrate_binary_checkpoint(
            SPLIT, OUT / run / "best.pt", output, min_recall=MIN_RECALL, candidates=THRESHOLDS
        )
        print("threshold", result["selected_threshold"])


def cmd_summarize(args):
    rows, receipt = verify_split(SPLIT)
    with TABLE.open(encoding="utf-8") as stream:
        geography = {r["clip"]: r["geography"] for r in csv.DictReader(stream)}
    dataset = {r.video_id: r.dataset for r in rows}
    deployed = json.loads((OUT / "deployed.json").read_text())
    calibration = json.loads(
        (OUT / deployed["deployed_run"] / "calibration/calibration.json").read_text()
    )
    threshold = calibration["selected_threshold"]
    runs = {}
    for arm in ARMS:
        for seed in SEEDS:
            run = OUT / f"{arm}_s{seed}"
            _, saved = load_binary_checkpoint(run / "best.pt")
            indian_nv = [
                p
                for p in saved["validation_predictions"]
                if geography.get(p["clip_id"]) == "documented_indian"
            ]
            runs[run.name] = {
                "best_step": max(
                    saved["history"],
                    key=lambda h: (h["validation"]["pr_auc"], h["validation"]["f1"]),
                )["step"],
                "all_val": validation_metrics(run, dataset, lambda d: True),
                **{
                    f"{name}_val": validation_metrics(run, dataset, lambda d, n=name: d == n)
                    for name in ("indian", "dcsass", "ubi", "xd")
                },
                "documented_indian_nv_fpr_at_threshold": float(
                    np.mean([p["violence_score"] >= threshold for p in indian_nv])
                ),
            }
    arms = {
        arm: {
            f"{part}_{m}": [
                float(np.mean([runs[f"{arm}_s{s}"][part][m] for s in SEEDS])),
                float(np.std([runs[f"{arm}_s{s}"][part][m] for s in SEEDS])),
            ]
            for part in ("all_val", "indian_val", "dcsass_val", "ubi_val", "xd_val")
            for m in ("pr_auc", "roc_auc")
        }
        for arm in ARMS
    }
    result = {
        "scope": "validation_only",
        "split_freeze_hash": receipt["freeze_hash"],
        "deployed": deployed,
        "threshold": threshold,
        "sanity_documented_indian_nv_fpr": runs[deployed["deployed_run"]][
            "documented_indian_nv_fpr_at_threshold"
        ],
        "runs": runs,
        "arms_val": arms,
    }
    (OUT.parent / "validation_results_v3.json").write_text(json.dumps(result, indent=2) + "\n")
    for arm, values in arms.items():
        print(arm, {k: [round(x, 3) for x in v] for k, v in values.items()})
    print("deployed", deployed["deployed_run"], "threshold", threshold)
    print("documented-Indian NV FPR", result["sanity_documented_indian_nv_fpr"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    train.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    for name in ("select", "calibrate", "summarize"):
        commands.add_parser(name)
    args = parser.parse_args()
    {
        "train": cmd_train,
        "select": cmd_select,
        "calibrate": cmd_calibrate,
        "summarize": cmd_summarize,
    }[args.command](args)


if __name__ == "__main__":
    torch.set_num_threads(3)
    run_cli(main)
