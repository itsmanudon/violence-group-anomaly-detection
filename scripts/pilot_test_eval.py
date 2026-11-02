"""Indian pilot Phase D: the one-time test evaluation defined by EVAL_PROTOCOL.md.

Refuses to run twice (exclusive output), and refuses test inputs that were not produced
under the committed protocol.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from _common import run_cli
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.datasets.indian_split import verify_split
from surveillance.inference.anomaly_pipeline import AnomalyPipeline
from surveillance.training.indian_binary import committed_protocol, load_binary_checkpoint
from surveillance.training.sultani_trainer import load_checkpoint

SPLIT = Path("manifests/v2-pilot/split_v1")
TABLE = Path("manifests/v2-pilot/split_v1.csv")
PROTOCOL = Path("runs/indian_pilot/EVAL_PROTOCOL.md")
CACHE = Path("runs/indian_pilot/features/c3d_test_v1")
CASCADE = Path("runs/indian_pilot/baselines/cascade_test/cascade.json")
EXPERIMENTS = Path("runs/indian_pilot/experiments")
SULTANI = Path("runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt")
OUTPUT = Path("runs/indian_pilot/test_results.json")
HEAD, HEAD_THRESHOLD = "violence_f100_s7", 0.40
SEEDS, FRACTIONS = (7, 8, 9), (25, 50, 75, 100)
SOURCE_COLLECTIONS = {"HCI-User-Collection", "Snatch-1.0", "Unidentified-Refined"}
RESAMPLES, BOOT_SEED, MAX_FPR = 2000, 20261103, 0.05


def pr_auc(y, s):
    precision, recall, _ = precision_recall_curve(y, s)
    return float(auc(recall, precision))


def recall_at_fpr(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= MAX_FPR].max())


def point(y, s, threshold):
    y, flagged = np.asarray(y), np.asarray(s) >= threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn, tn = int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if recall and precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def stratified_indices(y, rng):
    y = np.asarray(y)
    return np.concatenate(
        [rng.choice(np.flatnonzero(y == c), (y == c).sum(), replace=True) for c in (0, 1)]
    )


def bootstrap(y, scores: dict, threshold: dict, paired=None):
    """Percentile CIs; the same resample indices for every model (paired differences)."""
    y = np.asarray(y)
    rng = np.random.default_rng(BOOT_SEED)
    draws = {
        name: {k: [] for k in ("pr_auc", "roc_auc", "recall", "precision", "f1")} for name in scores
    }
    differences = []
    for _ in range(RESAMPLES):
        idx = stratified_indices(y, rng)
        for name, s in scores.items():
            s = np.asarray(s)[idx]
            draws[name]["pr_auc"].append(pr_auc(y[idx], s))
            draws[name]["roc_auc"].append(roc_auc_score(y[idx], s))
            p = point(y[idx], s, threshold[name])
            for key in ("recall", "precision", "f1"):
                draws[name][key].append(p[key])
        if paired:
            a, b = paired
            differences.append(
                pr_auc(y[idx], np.asarray(scores[a])[idx])
                - pr_auc(y[idx], np.asarray(scores[b])[idx])
            )
    ci = {
        name: {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()
        }
        for name, d in draws.items()
    }
    if paired:
        ci["paired_pr_auc_difference"] = {
            "models": list(paired),
            "mean": float(np.mean(differences)),
            "ci95": [
                float(np.percentile(differences, 2.5)),
                float(np.percentile(differences, 97.5)),
            ],
        }
    return ci


def evaluate(y, s, threshold):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    return {
        "count": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": pr_auc(y, s),
        "average_precision": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        f"recall_at_fpr_le_{MAX_FPR}": recall_at_fpr(y, s),
        **point(y, s, threshold),
    }


def head_scores(name, bags):
    model, _ = load_binary_checkpoint(EXPERIMENTS / name / "best.pt")
    with torch.inference_mode():
        return model(torch.from_numpy(bags)).softmax(-1)[:, 1].tolist()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run-val",
        action="store_true",
        help="run the identical code on validation inputs to catch bugs; never touches test",
    )
    args = parser.parse_args()
    split, cache, cascade_path, output = "test", CACHE, CASCADE, OUTPUT
    if args.dry_run_val:
        split = "val"
        cache = Path("runs/indian_pilot/features/c3d_trainval_v1")
        cascade_path = Path("runs/indian_pilot/baselines/cascade_val/cascade.json")
        output = Path("runs/indian_pilot/features/test_eval_dry_run_on_val.json")
        output.unlink(missing_ok=True)
    if output.exists():
        raise SystemExit("Test evaluation already ran once; refusing to rerun")
    protocol = None if args.dry_run_val else committed_protocol(PROTOCOL)
    rows, receipt = verify_split(SPLIT)
    with TABLE.open(encoding="utf-8") as stream:
        attrs = {row["clip"]: row for row in csv.DictReader(stream)}
    test = sorted(r.video_id for r in rows if r.split == split)
    registry = json.loads((cache / "registration.json").read_text())
    cascade = json.loads(cascade_path.read_text())
    for name, item in (("features", registry), ("cascade", cascade)):
        if item["test_protocol"] != protocol or item["split_freeze_hash"] != receipt["freeze_hash"]:
            raise ValueError(f"Test {name} were not produced under the committed protocol")
    if (set(registry["features"]) != set(test) and not args.dry_run_val) or set(
        cascade["clips"]
    ) != set(test):
        raise ValueError("Test inputs must cover exactly the frozen test clips")
    bags = []
    for key in test:
        entry = registry["features"][key]
        path = cache / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Test feature changed")
        bags.append(np.load(path, allow_pickle=False))
    bags = np.stack(bags).astype(np.float32)
    y = [int(attrs[k]["label"]) for k in test]

    scorer, _ = load_checkpoint(SULTANI, "cpu")
    pipeline = AnomalyPipeline(scorer, None, 32)
    scores = {
        "HEAD": head_scores(HEAD, bags),
        "SULTANI": [pipeline.predict_features(b, 1.0).overall_score for b in bags],
        "CASCADE": [float(cascade["clips"][k]["alert"]) for k in test],
        "SOURCE": [float(attrs[k]["collection"] in SOURCE_COLLECTIONS) for k in test],
    }
    thresholds = {"HEAD": HEAD_THRESHOLD, "SULTANI": 0.5, "CASCADE": 0.5, "SOURCE": 0.5}

    unknown = [i for i, k in enumerate(test) if attrs[k]["geography"] == "unknown_geography"]
    indian = [i for i, k in enumerate(test) if attrs[k]["geography"] == "documented_indian"]
    populations = {}
    for name, s in scores.items():
        s = np.asarray(s)
        populations[name] = {
            "overall": evaluate(y, s, thresholds[name]),
            "unknown_geography": evaluate(np.asarray(y)[unknown], s[unknown], thresholds[name]),
            "documented_indian_nv_false_positive_rate": float(
                (s[indian] >= thresholds[name]).mean()
            ),
        }
    ci = {
        "overall": bootstrap(y, scores, thresholds, paired=("HEAD", "SOURCE")),
        "unknown_geography": bootstrap(
            np.asarray(y)[unknown],
            {k: np.asarray(v)[unknown] for k, v in scores.items()},
            thresholds,
            paired=("HEAD", "SOURCE"),
        ),
    }

    curve = {}
    for fraction in FRACTIONS:
        values = {
            "overall_pr_auc": [],
            "overall_roc_auc": [],
            "unknown_pr_auc": [],
            "unknown_roc_auc": [],
        }
        for seed in SEEDS:
            s = np.asarray(head_scores(f"violence_f{fraction:03d}_s{seed}", bags))
            values["overall_pr_auc"].append(pr_auc(y, s))
            values["overall_roc_auc"].append(roc_auc_score(y, s))
            values["unknown_pr_auc"].append(pr_auc(np.asarray(y)[unknown], s[unknown]))
            values["unknown_roc_auc"].append(roc_auc_score(np.asarray(y)[unknown], s[unknown]))
        curve[str(fraction)] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in values.items()}

    diagnostics = {}
    geography = [int(attrs[k]["geography"] == "unknown_geography") for k in test]
    probe = head_scores("probe_geography_all_s7", bags)
    diagnostics["probe_geography_test"] = {
        "pr_auc": pr_auc(geography, probe),
        "roc_auc": float(roc_auc_score(geography, probe)),
    }
    for seed in SEEDS:
        s = np.asarray(head_scores(f"mitigation_unknown_only_s{seed}", bags))
        diagnostics[f"mitigation_unknown_only_s{seed}"] = {
            "overall_pr_auc": pr_auc(y, s),
            "unknown_pr_auc": pr_auc(np.asarray(y)[unknown], s[unknown]),
            "documented_indian_nv_fpr_at_0.40": float((s[indian] >= HEAD_THRESHOLD).mean()),
        }

    head = np.asarray(scores["HEAD"])
    violent = sorted((i for i in range(len(test)) if y[i] == 1), key=lambda i: -head[i])
    calm = sorted((i for i in range(len(test)) if y[i] == 0), key=lambda i: head[i])
    correct = [i for i in violent if head[i] >= HEAD_THRESHOLD][:5]
    correct += calm[: 5 - len(correct)] if len(correct) < 5 else []
    failures = [i for i in violent if head[i] < HEAD_THRESHOLD]
    failures += [i for i in reversed(calm) if head[i] >= HEAD_THRESHOLD][
        : max(0, 5 - len(failures))
    ]
    if len(failures) < 5:
        failures += [i for i in reversed(calm) if i not in failures][: 5 - len(failures)]

    def describe(i):
        k = test[i]
        return {
            "clip": k,
            "label": y[i],
            "collection": attrs[k]["collection"],
            "relative_path": attrs[k]["relative_path"],
            "head_score": float(head[i]),
            "sultani_score": float(scores["SULTANI"][i]),
            "cascade_alert_state": cascade["clips"][k]["alert_state"],
        }

    result = {
        "scope": "dry_run_on_validation" if args.dry_run_val else "one_time_test_evaluation",
        "protocol": protocol,
        "split_freeze_hash": receipt["freeze_hash"],
        "test_clips": len(test),
        "test_positives": int(sum(y)),
        "head_checkpoint_sha256": sha256(EXPERIMENTS / HEAD / "best.pt"),
        "thresholds": thresholds,
        "populations": populations,
        "bootstrap_ci95": ci,
        "learning_curve_test": curve,
        "diagnostics": diagnostics,
        "examples": {
            "correct": [describe(i) for i in correct],
            "failures": [describe(i) for i in failures],
        },
        "per_clip_head_scores": {k: float(s) for k, s in zip(test, head, strict=True)},
        "cascade_refused": cascade["refused"],
    }
    write_new_json(output, result)
    print(
        json.dumps(
            {k: result[k] for k in ("populations", "learning_curve_test", "diagnostics")}, indent=1
        )
    )


if __name__ == "__main__":
    run_cli(main)
