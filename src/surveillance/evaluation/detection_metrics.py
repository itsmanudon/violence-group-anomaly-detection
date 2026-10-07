"""Micro-aggregated actor localization metrics; no AP or classification claim."""

from pathlib import Path

from surveillance.detection.matching import MatchResult, match_manifest


def aggregate_matches(results: list[MatchResult]) -> dict:
    """Undefined ratios are null; empty_frames counts frames without detections."""
    frame_count = len(results)
    gt_count = sum(result.gt_count for result in results)
    detection_count = sum(result.detection_count for result in results)
    ious = [iou for result in results for iou in result.ious]
    matched = len(ious)
    return {
        "matches": matched,
        "gt_count": gt_count,
        "detection_count": detection_count,
        "precision": matched / detection_count if detection_count else None,
        "recall": matched / gt_count if gt_count else None,
        "f1": 2 * matched / (gt_count + detection_count) if gt_count + detection_count else None,
        "mean_matched_iou": sum(ious) / matched if matched else None,
        "missed_gt_actors": gt_count - matched,
        "unmatched_detections": detection_count - matched,
        "mean_actors_per_frame": detection_count / frame_count if frame_count else None,
        "mean_gt_actors_per_frame": gt_count / frame_count if frame_count else None,
        "frame_count": frame_count,
        "empty_frames": sum(result.detection_count == 0 for result in results),
    }


def evaluate_detections(
    manifest: Path,
    detections_path: Path,
    split: str | None = "test",
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate all selected GT scenes, requiring explicit records for empty detections."""
    scenes = match_manifest(manifest, detections_path, split, iou_threshold)
    metrics = aggregate_matches([MatchResult(**scene["matches"]) for scene in scenes])
    return {
        **metrics,
        "split": split,
        "iou_threshold": iou_threshold,
        "manifest": str(manifest),
        "detections": str(detections_path),
        "scenes": scenes,
        "explanation": (
            "One-to-one actor localization at an inclusive IoU threshold, maximizing eligible "
            "match count then IoU. Precision/recall/F1 aggregate counts across scenes. "
            "mean_actors_per_frame and empty_frames describe detector output. "
            "This is not average precision or actor/group classification accuracy."
        ),
    }
