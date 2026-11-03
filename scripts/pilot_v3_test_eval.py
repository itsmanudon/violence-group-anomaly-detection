"""Indian pilot v3: the one-time IITB-Corridor test evaluation defined by V3_PROTOCOL.md.

Refuses to rerun and refuses test features not produced under the committed protocol.
--dry-run-val runs the identical code on the v3 validation clips instead.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from _common import run_cli
from pilot_dcsass_test_eval import at_threshold, head, metrics

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.datasets.indian_split import verify_split
from surveillance.inference.anomaly_pipeline import AnomalyPipeline
from surveillance.training.indian_binary import committed_protocol
from surveillance.training.sultani_trainer import load_checkpoint

SPLIT = Path("manifests/v3-pilot/split_v3")
PROTOCOL = Path("runs/indian_pilot/V3_PROTOCOL.md")
TEST_CACHE = Path("runs/indian_pilot/features/c3d_v3_test")
VAL_CACHE = Path("runs/indian_pilot/features/c3d_v3_trainval")
V3 = Path("runs/indian_pilot/experiments_v3")
V2 = Path("runs/indian_pilot/experiments_v2")
V1_HEAD = Path("runs/indian_pilot/experiments/violence_f100_s7/best.pt")
SULTANI = Path("runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt")
OUTPUT = Path("runs/indian_pilot/test_results_v3.json")
SEEDS = (7, 8, 9)
V2_THRESHOLD, V1_THRESHOLD = 0.45, 0.40
RESAMPLES, BOOT_SEED = 2000, 20261103


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-val", action="store_true")
    args = parser.parse_args()
    split, cache, output = "test", TEST_CACHE, OUTPUT
    if args.dry_run_val:
        split, cache = "val", VAL_CACHE
        output = Path("runs/indian_pilot/features/test_eval_v3_dry_run_on_val.json")
        output.unlink(missing_ok=True)
    if output.exists():
        raise SystemExit("v3 test evaluation already ran once; refusing to rerun")
    protocol = None if args.dry_run_val else committed_protocol(PROTOCOL)
    deployed = json.loads((V3 / "deployed.json").read_text())
    arm, run = deployed["deployed_arm"], deployed["deployed_run"]
    threshold = json.loads((V3 / run / "calibration/calibration.json").read_text())[
        "selected_threshold"
    ]
    rows, receipt = verify_split(SPLIT)
    if deployed["split_freeze_hash"] != receipt["freeze_hash"]:
        raise ValueError("Deployed arm was selected on a different split")
    chosen = sorted((r for r in rows if r.split == split), key=lambda r: r.video_id)
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
    families = {
        "zero_shot_sultani": ["zero_shot_sultani"],
        "v1_indian_only_head": ["v1_indian_only_head"],
    }
    for name in ("v3_cctv", "v3_all"):
        families[name] = [f"{name}_s{s}" for s in SEEDS]
        for s in SEEDS:
            scores[f"{name}_s{s}"] = head(V3 / f"{name}_s{s}/best.pt", bags)
    families["v2"] = [f"v2_s{s}" for s in SEEDS]
    for s in SEEDS:
        scores[f"v2_s{s}"] = head(V2 / f"curve_d100_s{s}/best.pt", bags)

    def family_metrics(index, groups):
        out = {}
        for name, members in families.items():
            per_seed = [metrics(y[index], scores[m][index], groups) for m in members]
            out[name] = {k: float(np.nanmean([p[k] for p in per_seed])) for k in per_seed[0]}
        return out

    point = family_metrics(np.arange(len(y)), sources)
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
    comparisons = {f"{arm}_vs_v2": "v2", f"{arm}_vs_zero_shot_sultani": "zero_shot_sultani"}
    draws = {name: {k: [] for k in point[name]} for name in families}
    paired = {key: {"pr_auc": [], "within_source_roc_auc": []} for key in comparisons}
    for _ in range(RESAMPLES):
        picked = rng.choice(unique_sources, len(unique_sources), replace=True)
        # Duplicate sources get distinct labels so within-source AUC counts each draw once.
        index = np.concatenate([by_source[s] for s in picked])
        tags = np.concatenate([np.full(len(by_source[s]), i) for i, s in enumerate(picked)])
        if len(np.unique(y[index])) < 2:
            continue
        sample = family_metrics(index, tags)
        for name, values in sample.items():
            for k, v in values.items():
                draws[name][k].append(v)
        for key, other in comparisons.items():
            for k in paired[key]:
                paired[key][k].append(sample[arm][k] - sample[other][k])

    def ci(values):
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    differences: dict[str, dict] = {}
    for key, values in paired.items():
        differences[key] = {}
        for k, v in values.items():
            differences[key][f"{k}_difference"] = float(np.nanmean(v))
            differences[key][f"{k}_ci95"] = ci(v)
    criterion = differences[f"{arm}_vs_v2"]
    improved = criterion["pr_auc_ci95"][0] > 0 and criterion["within_source_roc_auc_ci95"][0] > 0
    result = {
        "scope": "dry_run_on_v3_validation"
        if args.dry_run_val
        else "one_time_iitb_test_evaluation",
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
        "deployed_model": run,
        "deployed_at_threshold": at_threshold(y, scores[run], threshold),
        "v2_at_0.45": at_threshold(y, scores["v2_s7"], V2_THRESHOLD),
        "v1_head_at_0.40": at_threshold(y, scores["v1_indian_only_head"], V1_THRESHOLD),
        "zero_shot_sultani_at_0.5": at_threshold(y, scores["zero_shot_sultani"], 0.5),
        "clip_scores": {
            r.video_id: {"label": int(r.label), **{m: float(scores[m][i]) for m in scores}}
            for i, r in enumerate(chosen)
        }
        if not args.dry_run_val
        else None,
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
                    "deployed_at_threshold",
                    "v2_at_0.45",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    run_cli(main)
