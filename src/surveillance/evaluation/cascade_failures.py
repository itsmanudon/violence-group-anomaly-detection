"""Read-only, hash-verified attribution of an already completed cascade pass."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.evaluation.cascade import summarize_cascade


def load_registered_cascade(directory: Path) -> tuple[list[dict], dict]:
    directory = Path(directory)
    completion = json.loads((directory / "completion.json").read_text())
    registration = json.loads((directory / "registration.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    if not completion["complete"] or completion["clips"] != len(progress):
        raise ValueError("Incomplete registered cascade")
    if sha256(directory / "metrics.json") != completion["metrics_sha256"]:
        raise ValueError("Historical metrics hash mismatch")
    rows = []
    for clip_id, digest in sorted(progress.items()):
        if Path(clip_id).name != clip_id or "/" in clip_id or "\\" in clip_id:
            raise ValueError("Unsafe registered clip ID")
        path = directory / "clips" / f"{clip_id}.json"
        if sha256(path) != digest:
            raise ValueError(f"Saved prediction hash mismatch: {clip_id}")
        row = json.loads(path.read_text())
        if row["clip_id"] != clip_id or row["result"]["provenance"] != registration["provenance"]:
            raise ValueError("Saved prediction identity mismatch")
        rows.append(row)
    metrics = summarize_cascade(rows)
    if metrics != json.loads((directory / "metrics.json").read_text()):
        raise ValueError("Recomputed historical metrics differ")
    return rows, metrics


def attribute_failures(rows: list[dict]) -> dict:
    """Descriptive only: no alternate thresholds, policy sweeps or model selection."""
    metrics = summarize_cascade(rows)
    counts = Counter()
    examples = defaultdict(list)
    sources = defaultdict(Counter)
    classes = {name: Counter() for name in CLASSES}
    confidence = {"correct": [], "incorrect": []}
    for row in sorted(rows, key=lambda r: r["clip_id"]):
        result, target = row["result"], row["group_label"]
        windows = result["behavior"]["windows"]
        available = [w for w in windows if w["status"] == "available"]
        routed = bool(windows)
        label = row["binary_label"]
        outcomes = ["clips", "routed" if routed else "bypassed"]
        if not label and routed:
            outcomes.append("normal_false_alert")
        if label and not routed:
            outcomes.append("abnormal_bypassed")
        if windows and not available:
            outcomes.append("routed_no_actors")
        if available:
            chosen = next(
                (
                    w
                    for w in available
                    if w["interval"] == result["final_alert"]["timestamp_interval"]
                ),
                available[0],
            )
            prediction = int(np.argmax(chosen["probabilities"]))
            correct = prediction == target
            outcomes.extend(["covered", "behavior_correct" if correct else "behavior_incorrect"])
            confidence["correct" if correct else "incorrect"].append(max(chosen["probabilities"]))
            if any(w["status"] == "no_actors" for w in windows):
                outcomes.append("mixed_actor_coverage")
            if len({int(np.argmax(w["probabilities"])) for w in available}) > 1:
                outcomes.append("cross_window_behavior_disagreement")
        counts.update(outcomes)
        sources[row["source_video_id"]].update(outcomes)
        classes[CLASSES[target]].update(outcomes)
        for outcome in outcomes:
            if len(examples[outcome]) < 5:
                examples[outcome].append(row["clip_id"])
    return {
        "scope": "historical_held_out_description_only_no_selection",
        "counts": dict(counts),
        "per_source": {k: dict(v) for k, v in sorted(sources.items())},
        "per_class": {k: dict(v) for k, v in classes.items()},
        "examples_first_five_by_id": dict(examples),
        "chosen_softmax_maximum": {
            key: {"count": len(v), "median": float(np.median(v)) if v else None}
            for key, v in confidence.items()
        },
        "historical_metrics_recomputed": metrics,
        "causal_effects_of_temporal_calibration": "NOT MEASURED",
        "detector_false_negative_rate": "NOT MEASURED: no person ground truth",
        "unsupported_media_rate": "NOT MEASURED: completed predictions exclude ingest failures",
        "performance_selection": "NOT PERFORMED",
    }
