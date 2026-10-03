"""Explicit class metrics for scene groups and valid annotated actors."""

from pathlib import Path

import numpy as np
import torch


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def classification_metrics(targets, predictions, num_classes: int) -> dict:
    """Macro F1 includes all configured classes; absent-class accuracy is null."""
    truth, predicted = _array(targets).reshape(-1), _array(predictions).reshape(-1)
    if num_classes < 1 or truth.size == 0 or truth.shape != predicted.shape:
        raise ValueError("Metrics require matching nonempty labels and positive class count")
    for values in (truth, predicted):
        if not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all():
            raise ValueError("Class labels must be finite integers")
        if (values < 0).any() or (values >= num_classes).any():
            raise ValueError("Class labels fall outside configured vocabulary")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (truth.astype(int), predicted.astype(int)), 1)
    support, predicted_count = confusion.sum(axis=1), confusion.sum(axis=0)
    correct = confusion.diagonal()
    denominator = support + predicted_count
    f1 = np.divide(2 * correct, denominator, out=np.zeros(num_classes), where=denominator > 0)
    return {
        "count": int(truth.size),
        "accuracy": float(correct.sum() / truth.size),
        "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(),
        "per_class_accuracy": [float(c / n) if n else None for c, n in zip(correct, support)],
        "support": support.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def group_activity_metrics(
    group_labels,
    group_predictions,
    actor_labels,
    actor_predictions,
    actor_valid_mask,
    num_group_classes: int,
    num_actor_classes: int,
) -> dict:
    """Exclude both masked actors and ignore-index labels from actor metrics."""
    labels, predictions, mask = map(_array, (actor_labels, actor_predictions, actor_valid_mask))
    if labels.shape != predictions.shape or labels.shape != mask.shape:
        raise ValueError("Actor labels, predictions and validity masks must have matching shapes")
    valid = mask.astype(bool) & (labels != -100)
    return {
        "group": classification_metrics(group_labels, group_predictions, num_group_classes),
        "actor": classification_metrics(labels[valid], predictions[valid], num_actor_classes),
    }


def evaluate(checkpoint: Path, manifest: Path, split: str = "test", device: str = "auto") -> dict:
    """Evaluate a checkpoint on an explicit split; never update/select checkpoints."""
    from surveillance.inference.group_activity_pipeline import GroupActivityPipeline
    from surveillance.training.actor_transformer_trainer import load_checkpoint, select_device

    system, saved = load_checkpoint(checkpoint, select_device(device))
    scenes = GroupActivityPipeline(system).predict_manifest(manifest, split)
    groups = [scene["group_label"] for scene in scenes]
    group_predictions = [scene["group_prediction"] for scene in scenes]
    actors = [actor for scene in scenes for actor in scene["actors"]]
    model = saved["config"]["model"]
    result = group_activity_metrics(
        groups,
        group_predictions,
        [actor["label"] for actor in actors],
        [actor["prediction"] for actor in actors],
        [True] * len(actors),
        model["num_group_classes"],
        model["num_actor_classes"],
    )
    result.update(
        split=split,
        checkpoint=str(checkpoint),
        explanation="Scene group and valid-actor classification; no anomaly scores.",
    )
    return result
