"""Group-only actor inputs and detector coverage without invented actor labels."""

from collections import Counter

import numpy as np
import torch

from surveillance.datasets.dcsass_protocol import CLASSES
from surveillance.datasets.detected_actors import collate_actor_inputs


def collate_behavior(samples: list[dict]) -> dict:
    """Reuse masked actor-input batching; attach only observed group targets."""
    labels = [s["group_label"] for s in samples]
    if any(type(label) is not int or not 0 <= label < len(CLASSES) for label in labels):
        raise ValueError("Expected observed six-class group targets")
    batch = collate_actor_inputs(samples)
    batch["group_labels"] = torch.tensor(labels, dtype=torch.long)
    return batch


def coverage_report(rows: list[dict]) -> dict:
    """Report class coverage and counts only: DCSASS has no GT actor boxes."""
    if not rows:
        raise ValueError("Coverage requires a nonempty clip population")
    counts = [len(r["boxes"]) for r in rows]
    per_class = {}
    for index, name in enumerate(CLASSES):
        group = [r for r in rows if r["group_label"] == index]
        empty = sum(not r["boxes"] for r in group)
        per_class[name] = {
            "total_clips": len(group),
            "covered_clips": len(group) - empty,
            "no_actor_clips": empty,
            "no_actor_rate": empty / len(group) if group else None,
        }
    empty = counts.count(0)
    scores = [s for r in rows for s in r["scores"]]
    return {
        "total_clips": len(rows),
        "covered_clips": len(rows) - empty,
        "no_actor_clips": empty,
        "coverage": 1 - empty / len(rows),
        "actor_count_histogram": dict(Counter(counts)),
        "confidence_quantiles": np.quantile(scores, [0, 0.25, 0.5, 0.75, 1]).tolist()
        if scores
        else [],
        "per_class": per_class,
        "ground_truth_actor_boxes_available": False,
        "optimization_policy": "exclude no-actor clips",
        "evaluation_policy": "conditional classification metrics plus all-population coverage",
    }
