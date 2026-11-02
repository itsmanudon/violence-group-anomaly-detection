"""Indian pilot v2: the one-time DCSASS test evaluation defined by DCSASS_PROTOCOL.md.

Refuses to rerun and refuses test features not produced under the committed protocol.
--dry-run-val runs the identical code on the DCSASS validation clips instead.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from _common import run_cli
from scipy.stats import rankdata
from sklearn.metrics import auc, precision_recall_curve

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.datasets.indian_split import verify_split
from surveillance.inference.anomaly_pipeline import AnomalyPipeline
from surveillance.training.indian_binary import committed_protocol, load_binary_checkpoint
from surveillance.training.sultani_trainer import load_checkpoint

SPLIT = Path("manifests/v2-pilot/split_combined_v1")
PROTOCOL = Path("runs/indian_pilot/DCSASS_PROTOCOL.md")
TEST_CACHE = Path("runs/indian_pilot/features/c3d_combined_test_v1")
VAL_CACHE = Path("runs/indian_pilot/features/c3d_combined_trainval_v1")
V2 = Path("runs/indian_pilot/experiments_v2")
V1_HEAD = Path("runs/indian_pilot/experiments/violence_f100_s7/best.pt")
SULTANI = Path("runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt")
OUTPUT = Path("runs/indian_pilot/test_results_v2.json")
SEEDS, FRACTIONS = (7, 8, 9), (0, 10, 25, 50, 100)
RESAMPLES, BOOT_SEED = 2000, 20261103


def pr_auc(y, s):
    precision, recall, _ = precision_recall_curve(y, s)
    return float(auc(recall, precision))


def roc(y, s):
    """Exact ROC-AUC via the Mann-Whitney rank statistic (ties averaged), as sklearn."""
    ranks = rankdata(s)
    positives = y == 1
    n1, n0 = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_source_roc(y, s, sources):
    values = []
    for source in np.unique(sources):
        mask = sources == source
        if 0 < y[mask].sum() < mask.sum():
            values.append(roc(y[mask], s[mask]))
    return float(np.mean(values)) if values else float("nan")


def metrics(y, s, sources):
    return {
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc(y, s),
        "within_source_roc_auc": within_source_roc(y, s, sources),
    }


def at_threshold(y, s, threshold):
    flagged = s >= threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn, tn = int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def head(path, bags):
    model, _ = load_binary_checkpoint(path)
    with torch.inference_mode():
        return model(torch.from_numpy(bags)).softmax(-1)[:, 1].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-val", action="store_true")
    args = parser.parse_args()
    split, cache, output = "test", TEST_CACHE, OUTPUT
    if args.dry_run_val:
        split, cache = "val", VAL_CACHE
        output = Path("runs/indian_pilot/features/test_eval_v2_dry_run_on_val.json")
        output.unlink(missing_ok=True)
    if output.exists():
        raise SystemExit("v2 test evaluation already ran once; refusing to rerun")
    protocol = None if args.dry_run_val else committed_protocol(PROTOCOL)
    rows, receipt = verify_split(SPLIT)
    chosen = sorted(
        (r for r in rows if r.split == split and r.dataset == "dcsass"), key=lambda r: r.video_id
    )
    registry = json.loads((cache / "registration.json").read_text())
    if registry["split_freeze_hash"] != receipt["freeze_hash"]:
        raise ValueError("Feature cache belongs to a different split")
    if not args.dry_run_val and (
        registry["test_protocol"] != protocol
        or set(registry["features"]) != {r.video_id for r in rows if r.split == "test"}
    ):
        raise ValueError("Test features were not produced under the committed protocol")
    bags = []
    for r in chosen:
        entry = registry["features"][r.video_id]
        if sha256(cache / entry["path"]) != entry["sha256"]:
            raise ValueError("Feature changed")
        bags.append(np.load(cache / entry["path"], allow_pickle=False))
    bags = np.stack(bags).astype(np.float32)
    y = np.array([r.label for r in chosen])
    sources = np.array([r.source_video_id for r in chosen])

    scorer, _ = load_checkpoint(SULTANI, "cpu")
    pipeline = AnomalyPipeline(scorer, None, 32)
    scores = {
        "zero_shot_sultani": np.array(
            [pipeline.predict_features(b, 1.0).overall_score for b in bags]
        ),
        "v1_indian_only_head": head(V1_HEAD, bags),
    }
    for fraction in FRACTIONS:
        for seed in SEEDS:
            scores[f"curve_d{fraction:03d}_s{seed}"] = head(
                V2 / f"curve_d{fraction:03d}_s{seed}/best.pt", bags
            )
    for seed in SEEDS:
        scores[f"dcsass_only_s{seed}"] = head(V2 / f"dcsass_only_s{seed}/best.pt", bags)
    families = {f"curve_d{f:03d}": [f"curve_d{f:03d}_s{s}" for s in SEEDS] for f in FRACTIONS}
    families["dcsass_only"] = [f"dcsass_only_s{s}" for s in SEEDS]
    families["zero_shot_sultani"] = ["zero_shot_sultani"]
    families["v1_indian_only_head"] = ["v1_indian_only_head"]

    def family_metrics(idx):
        out = {}
        for name, members in families.items():
            per_seed = [metrics(y[idx], scores[m][idx], sources[idx]) for m in members]
            out[name] = {k: float(np.nanmean([p[k] for p in per_seed])) for k in per_seed[0]}
        return out

    everything = np.arange(len(y))
    point = family_metrics(everything)
    spread = {
        name: {
            k: float(np.std([metrics(y, scores[m], sources)[k] for m in members]))
            for k in ("pr_auc", "roc_auc", "within_source_roc_auc")
        }
        for name, members in families.items()
    }
    unique_sources = np.unique(sources)
    by_source = {s: np.flatnonzero(sources == s) for s in unique_sources}
    rng = np.random.default_rng(BOOT_SEED)
    draws = {name: {k: [] for k in point[name]} for name in families}
    paired = {"curve_d100_vs_curve_d000": [], "curve_d100_vs_zero_shot_sultani": []}
    paired_ws = {"curve_d100_vs_curve_d000": [], "curve_d100_vs_zero_shot_sultani": []}
    for _ in range(RESAMPLES):
        picked = rng.choice(unique_sources, len(unique_sources), replace=True)
        # Duplicate sources get distinct labels so within-source AUC counts each draw once.
        idx = np.concatenate([by_source[s] for s in picked])
        tags = np.concatenate([np.full(len(by_source[s]), i) for i, s in enumerate(picked)])
        if len(np.unique(y[idx])) < 2:
            continue
        sample = {}
        for name, members in families.items():
            per_seed = [metrics(y[idx], scores[m][idx], tags) for m in members]
            sample[name] = {k: float(np.nanmean([p[k] for p in per_seed])) for k in per_seed[0]}
            for k, v in sample[name].items():
                draws[name][k].append(v)
        for key, other in (
            ("curve_d100_vs_curve_d000", "curve_d000"),
            ("curve_d100_vs_zero_shot_sultani", "zero_shot_sultani"),
        ):
            paired[key].append(sample["curve_d100"]["pr_auc"] - sample[other]["pr_auc"])
            paired_ws[key].append(
                sample["curve_d100"]["within_source_roc_auc"]
                - sample[other]["within_source_roc_auc"]
            )

    def ci(values):
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    differences = {
        key: {
            "pr_auc_difference": float(np.mean(paired[key])),
            "pr_auc_ci95": ci(paired[key]),
            "within_source_roc_auc_difference": float(np.nanmean(paired_ws[key])),
            "within_source_roc_auc_ci95": ci(paired_ws[key]),
        }
        for key in paired
    }
    criterion = differences["curve_d100_vs_curve_d000"]
    improved = criterion["pr_auc_ci95"][0] > 0 and criterion["within_source_roc_auc_ci95"][0] > 0

    deployed = json.loads((V2 / "curve_d100_s7/calibration/calibration.json").read_text())
    result = {
        "scope": "dry_run_on_dcsass_validation"
        if args.dry_run_val
        else "one_time_dcsass_test_evaluation",
        "protocol": protocol,
        "split_freeze_hash": receipt["freeze_hash"],
        "clips": int(len(y)),
        "positives": int(y.sum()),
        "sources": int(len(unique_sources)),
        "sources_with_both_labels": int(sum(len(np.unique(y[i])) == 2 for i in by_source.values())),
        "metrics_seed_mean": point,
        "metrics_seed_std": spread,
        "source_bootstrap_ci95": {
            name: {k: ci(v) for k, v in d.items()} for name, d in draws.items()
        },
        "paired_differences": differences,
        "improved_by_preregistered_criterion": bool(improved),
        "deployed_model": "curve_d100_s7",
        "deployed_at_threshold": at_threshold(
            y, scores["curve_d100_s7"], deployed["selected_threshold"]
        ),
        "v1_head_at_0.40": at_threshold(y, scores["v1_indian_only_head"], 0.40),
        "zero_shot_sultani_at_0.5": at_threshold(y, scores["zero_shot_sultani"], 0.5),
    }
    write_new_json(output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "metrics_seed_mean",
                    "paired_differences",
                    "improved_by_preregistered_criterion",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    run_cli(main)
