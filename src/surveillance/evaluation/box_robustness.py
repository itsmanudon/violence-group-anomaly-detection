"""Same-checkpoint GT/detected comparison with explicit coverage and abstentions."""

from pathlib import Path

from surveillance.detection.matching import MatchResult
from surveillance.evaluation.detection_metrics import aggregate_matches
from surveillance.evaluation.group_activity_metrics import classification_metrics
from surveillance.inference.detected_group_activity import DetectedGroupActivityPipeline
from surveillance.inference.group_activity_pipeline import GroupActivityPipeline
from surveillance.training.actor_transformer_trainer import load_checkpoint
from surveillance.training.sultani_trainer import select_device


def _metrics(labels: list[int], predictions: list[int], count: int) -> dict:
    if not labels:
        return {
            "count": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_class_accuracy": [None] * count,
            "confusion_matrix": None,
        }
    return classification_metrics(labels, predictions, count)


def compare_predictions(
    ground_truth: list[dict],
    detected: list[dict],
    matches: list,
    num_group_classes: int,
    num_actor_classes: int,
) -> dict:
    """Compare identical scenes; paired actor deltas use the identical matched GT subset.

    Group all-scene accuracy treats no-actor abstentions as incorrect. Macro F1
    is reported on supported scenes and compared to GT on that same subset.
    Conditional matched-actor accuracy must always be read alongside coverage.
    """
    if not ground_truth or len(ground_truth) != len(detected) or len(matches) != len(detected):
        raise ValueError("Comparison requires the same nonempty scene list and matching results")
    gt_groups, gt_predictions, gt_labels, gt_actor_predictions = [], [], [], []
    supported_labels, supported_gt, supported_det = [], [], []
    paired_labels, paired_gt, paired_det = [], [], []
    for gt, det, match in zip(ground_truth, detected, matches, strict=True):
        for key in ("dataset", "clip_id", "source_video_id"):
            if gt["metadata"].get(key) != det["metadata"].get(key):
                raise ValueError("GT/detected scene identity or ordering mismatch")
        if gt["group_label"] != det["group_label"]:
            raise ValueError("GT/detected group labels differ")
        gt_groups.append(gt["group_label"])
        gt_predictions.append(gt["group_prediction"])
        gt_labels.extend(a["label"] for a in gt["actors"])
        gt_actor_predictions.extend(a["prediction"] for a in gt["actors"])
        if det["status"] == "ok":
            supported_labels.append(gt["group_label"])
            supported_gt.append(gt["group_prediction"])
            supported_det.append(det["group_prediction"])
        for gi, di in zip(match.gt_indices, match.detection_indices, strict=True):
            label = gt["actors"][gi]["label"]
            if det["actors"][di]["label"] != label:
                raise ValueError("Matched actor label transfer differs from ground truth")
            paired_labels.append(label)
            paired_gt.append(gt["actors"][gi]["prediction"])
            paired_det.append(det["actors"][di]["prediction"])
    gt_group = _metrics(gt_groups, gt_predictions, num_group_classes)
    gt_actor = _metrics(gt_labels, gt_actor_predictions, num_actor_classes)
    det_group = _metrics(supported_labels, supported_det, num_group_classes)
    det_actor = _metrics(paired_labels, paired_det, num_actor_classes)
    paired_gt_group = _metrics(supported_labels, supported_gt, num_group_classes)
    paired_gt_actor = _metrics(paired_labels, paired_gt, num_actor_classes)
    all_accuracy = sum(y == p for y, p in zip(supported_labels, supported_det)) / len(detected)
    det_group["accuracy_all_scenes_abstentions_incorrect"] = all_accuracy
    coverage = aggregate_matches(matches)

    def delta(left, right, key):
        return None if left[key] is None or right[key] is None else left[key] - right[key]

    return {
        "protocol": "same_checkpoint_ground_truth_vs_detected_boxes",
        "ground_truth_boxes": {
            "group": gt_group,
            "actor": gt_actor,
            "mean_actors_per_frame": len(gt_labels) / len(detected),
        },
        "detected_boxes": {
            "group": det_group,
            "actor_matched_only": det_actor,
            "scene_count": len(detected),
            "predicted_scene_count": len(supported_labels),
            "no_actor_scene_count": len(detected) - len(supported_labels),
            "scene_coverage": len(supported_labels) / len(detected),
            "detection": coverage,
        },
        "paired_ground_truth": {
            "group_supported_scenes": paired_gt_group,
            "actor_matched_subset": paired_gt_actor,
        },
        "delta": {
            "group_accuracy_all_scenes": all_accuracy - gt_group["accuracy"],
            "group_accuracy_supported_scenes": delta(det_group, paired_gt_group, "accuracy"),
            "group_macro_f1_supported_scenes": delta(det_group, paired_gt_group, "macro_f1"),
            "actor_accuracy_matched_subset": delta(det_actor, paired_gt_actor, "accuracy"),
            "actor_macro_f1_matched_subset": delta(det_actor, paired_gt_actor, "macro_f1"),
        },
        "metric_policy": "Deltas are detected minus GT. Matched actor metrics exclude extras; "
        "coverage reports misses. Empty scenes abstain and count as incorrect in "
        "all-scene group accuracy; supported metrics use the same GT scene subset.",
    }


def evaluate_box_robustness(
    checkpoint: Path,
    manifest: Path,
    detections: Path,
    split: str = "test",
    iou_threshold: float = 0.5,
    device: str = "auto",
    return_attention: bool = False,
) -> dict:
    """Evaluate one trained checkpoint with GT and separately extracted detected features."""
    system, saved = load_checkpoint(checkpoint, select_device(device))
    model = saved["config"]["model"]
    gt = GroupActivityPipeline(system).predict_manifest(
        manifest, split, return_attention=return_attention
    )
    det = DetectedGroupActivityPipeline(system).predict_manifest(
        manifest,
        detections,
        split,
        iou_threshold,
        return_attention,
    )
    matches = [MatchResult(**scene["matching"]) for scene in det]
    result = compare_predictions(
        gt, det, matches, model["num_group_classes"], model["num_actor_classes"]
    )
    result.update(
        checkpoint=str(checkpoint),
        manifest=str(manifest),
        detections=str(detections),
        split=split,
        iou_threshold=iou_threshold,
        ground_truth_predictions=gt,
        detected_predictions=det,
    )
    return result
