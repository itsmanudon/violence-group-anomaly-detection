"""Indian pilot Phase C: train heads, learning curve, shortcut probes and baselines.

Every setting below was fixed before any validation result was seen (DECISIONS.md
D14-D16). Validation only; test features do not exist when this runs.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from _common import run_cli

from surveillance.datasets.indian_split import verify_split
from surveillance.evaluation.indian_binary import evaluate_binary
from surveillance.inference.anomaly_pipeline import AnomalyPipeline
from surveillance.training.indian_binary import (
    calibrate_binary_checkpoint,
    load_binary_checkpoint,
    train_binary,
)
from surveillance.training.sultani_trainer import load_checkpoint

SPLIT = Path("manifests/v2-pilot/split_v1")
TABLE = Path("manifests/v2-pilot/split_v1.csv")
CACHE = Path("runs/indian_pilot/features/c3d_trainval_v1")
OUT = Path("runs/indian_pilot/experiments")
SULTANI = Path("runs/ucf-crime/sultani_shared_safe_v1/seed_0/best.pt")
TRAIN = dict(
    steps=300,
    learning_rate=1e-3,
    batch_size=16,
    class_weighting="balanced",
    sampling="uniform",
    patience=60,
    precision="fp32",
)
SEEDS = (7, 8, 9)
DEPLOYED_SEED = 7
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SUBSET_SEED = 7
MIN_RECALL = 0.8
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def attributes() -> dict[str, dict]:
    with TABLE.open(encoding="utf-8") as stream:
        return {row["clip"]: row for row in csv.DictReader(stream)}


def nested_subsets(rows, attrs) -> dict[float, list[str]]:
    """Nested train-group subsets, stratified by (contains violence, geography)."""
    groups: dict[str, set] = {}
    for r in rows:
        if r.split == "train":
            groups.setdefault(r.source_video_id, set()).add(r.video_id)
    strata: dict[tuple, list] = {}
    for group, clips in sorted(groups.items()):
        violent = any(attrs[c]["label"] == "1" for c in clips)
        geography = attrs[next(iter(sorted(clips)))]["geography"]
        strata.setdefault((violent, geography), []).append(group)
    rng = random.Random(SUBSET_SEED)
    for members in strata.values():
        rng.shuffle(members)
    return {
        fraction: sorted(
            g for members in strata.values() for g in members[: math.ceil(fraction * len(members))]
        )
        for fraction in FRACTIONS
    }


def run_one(name, **options):
    output = OUT / name
    if (output / "best.pt").exists():
        return output
    train_binary(SPLIT, CACHE, output, device="cpu", **TRAIN, **options)
    return output


def cmd_train(args):
    rows, _ = verify_split(SPLIT)
    attrs = attributes()
    trainval = [r for r in rows if r.split in {"train", "val"}]
    subsets = nested_subsets(rows, attrs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subsets.json").write_text(json.dumps({str(k): v for k, v in subsets.items()}, indent=2))
    for fraction in FRACTIONS:
        for seed in SEEDS:
            groups = None if fraction == 1.0 else subsets[fraction]
            run_one(f"violence_f{int(fraction * 100):03d}_s{seed}", seed=seed, train_groups=groups)
            print("done", fraction, seed, flush=True)
    geography = {
        r.video_id: int(attrs[r.video_id]["geography"] == "unknown_geography") for r in trainval
    }
    nonviolent = sorted(
        {r.source_video_id for r in rows if r.split == "train"}
        - {r.source_video_id for r in rows if r.split == "train" and r.label == 1}
    )
    unknown = sorted(
        {
            r.source_video_id
            for r in rows
            if r.split == "train" and attrs[r.video_id]["geography"] == "unknown_geography"
        }
    )
    for seed in SEEDS:
        # Probe A: can the features tell the two geographies apart (all clips)?
        run_one(
            f"probe_geography_all_s{seed}", seed=seed, relabel=geography, target_name="geography"
        )
        # Probe B: the same, trained only on non-violence clips (a violence-free collection signal).
        run_one(
            f"probe_geography_nv_s{seed}",
            seed=seed,
            relabel=geography,
            target_name="geography_nonviolence_only",
            train_groups=nonviolent,
        )
        # Mitigation: violence trained only within unknown geography.
        run_one(f"mitigation_unknown_only_s{seed}", seed=seed, train_groups=unknown)
        print("done probes/mitigation", seed, flush=True)
    calibration = OUT / f"violence_f100_s{DEPLOYED_SEED}" / "calibration"
    if not calibration.exists():
        calibrate_binary_checkpoint(
            SPLIT,
            OUT / f"violence_f100_s{DEPLOYED_SEED}" / "best.pt",
            calibration,
            min_recall=MIN_RECALL,
            candidates=THRESHOLDS,
        )


def sultani_scores(cache: Path, clips) -> dict[str, float]:
    """Zero-shot UCF Sultani clip score: max over the 32 segments of the same C3D bag."""
    registry = json.loads((cache / "registration.json").read_text())
    scorer, _ = load_checkpoint(SULTANI, "cpu")
    pipeline = AnomalyPipeline(scorer, None, 32)
    scores = {}
    for key in clips:
        bag = np.load(cache / registry["features"][key]["path"], allow_pickle=False)
        scores[key] = pipeline.predict_features(bag, 1.0).overall_score
    return scores


def subset_metrics(predictions, attrs, threshold):
    def pick(condition):
        chosen = [p for p in predictions if condition(attrs[p["clip_id"]])]
        labels = [p["label"] for p in chosen]
        scores = [p["score"] for p in chosen]
        if not chosen:
            return None
        report = evaluate_binary(labels, scores, threshold)
        keep = ("count", "pr_auc", "average_precision", "roc_auc", "precision", "recall", "f1")
        out = {k: report.get(k) for k in keep}
        out["positives"] = sum(labels)
        out["false_positive_rate"] = report["confusion_matrix"][0][1] / max(
            1, report["confusion_matrix"][0][0] + report["confusion_matrix"][0][1]
        )
        out["confusion_matrix"] = report["confusion_matrix"]
        return out

    return {
        "overall": pick(lambda a: True),
        "unknown_geography": pick(lambda a: a["geography"] == "unknown_geography"),
        "documented_indian_nv_only": pick(lambda a: a["geography"] == "documented_indian"),
        "snatch_only": pick(lambda a: a["collection"] == "Snatch-1.0"),
    }


def val_predictions(run: Path):
    _, saved = load_binary_checkpoint(run / "best.pt")
    return [
        {"clip_id": p["clip_id"], "label": p["label"], "score": p["violence_score"]}
        for p in saved["validation_predictions"]
    ], saved


def cmd_summarize(args):
    rows, receipt = verify_split(SPLIT)
    attrs = attributes()
    calibration = json.loads(
        (OUT / f"violence_f100_s{DEPLOYED_SEED}" / "calibration" / "calibration.json").read_text()
    )
    threshold = calibration["selected_threshold"]
    result = {
        "scope": "validation_only",
        "split_freeze_hash": receipt["freeze_hash"],
        "deployed_seed": DEPLOYED_SEED,
        "validation_threshold": threshold,
        "threshold_rule": calibration["objective"] + f" (recall >= {MIN_RECALL})",
        "runs": {},
    }
    for run in sorted(p for p in OUT.iterdir() if (p / "best.pt").exists()):
        predictions, saved = val_predictions(run)
        settings = json.loads((run / "registration.json").read_text())["settings"]
        result["runs"][run.name] = {
            "target": settings["target_name"],
            "training_clips": settings["training_clip_count"],
            "training_class_counts": settings["training_class_counts"],
            "best_step": max(
                saved["history"], key=lambda h: (h["validation"]["pr_auc"], h["validation"]["f1"])
            )["step"],
            "metrics": subset_metrics(
                predictions, attrs, threshold if settings["target_name"] == "violence" else 0.5
            ),
        }
    val = [r.video_id for r in rows if r.split == "val"]
    zero = sultani_scores(CACHE, val)
    labelled = [{"clip_id": k, "label": int(attrs[k]["label"]), "score": zero[k]} for k in val]
    result["baselines"] = {
        "zero_shot_sultani": {
            "threshold": 0.5,
            "note": "UCF Sultani max segment score on the same C3D bags (cascade anomaly stage)",
            "metrics": subset_metrics(labelled, attrs, 0.5),
            "scores": zero,
        }
    }
    curve = {}
    for fraction in FRACTIONS:
        values = [
            result["runs"][f"violence_f{int(fraction * 100):03d}_s{s}"]["metrics"]["overall"]
            for s in SEEDS
        ]
        unknown = [
            result["runs"][f"violence_f{int(fraction * 100):03d}_s{s}"]["metrics"][
                "unknown_geography"
            ]
            for s in SEEDS
        ]
        curve[str(fraction)] = {
            "training_clips": result["runs"][f"violence_f{int(fraction * 100):03d}_s7"][
                "training_clips"
            ],
            "training_class_counts": result["runs"][f"violence_f{int(fraction * 100):03d}_s7"][
                "training_class_counts"
            ],
            **{
                f"{name}_{metric}": [
                    float(np.mean([v[metric] for v in vals])),
                    float(np.std([v[metric] for v in vals])),
                ]
                for name, vals in (("overall", values), ("unknown", unknown))
                for metric in ("pr_auc", "roc_auc")
            },
        }
    result["learning_curve_val"] = curve
    (OUT.parent / "validation_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"runs", "baselines"}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    commands.add_parser("summarize")
    args = parser.parse_args()
    {"train": cmd_train, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    torch.set_num_threads(4)
    run_cli(main)
