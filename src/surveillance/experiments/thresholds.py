"""Detector confidence selection using validation scenes and broad local candidates only."""

import hashlib
import json
import math
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import torch

from surveillance.datasets.collective import TEST_SEQUENCES, read_actor_manifest
from surveillance.datasets.common import resolve_path
from surveillance.detection.config import DetectionConfig
from surveillance.detection.matching import match_boxes
from surveillance.detection.person_detector import DetectionResult, filter_detections
from surveillance.detection.records import DetectionRecord, read_detections, write_detections
from surveillance.evaluation.detection_metrics import aggregate_matches

OBJECTIVE = "maximize_detector_f1_subject_to_min_recall"
CANDIDATE_FILTERS = DetectionConfig(0.0, 1.0, 1000, 0.0, 0.0, 0.0)


def _validation_rows(manifest: Path, validation_sequences: list[int] | None = None):
    all_rows = read_actor_manifest(manifest)
    test_sources = {f"collective:seq{sid:02d}" for sid in TEST_SEQUENCES}
    for row in all_rows:
        if (row.source_video_id in test_sources) != (row.split == "test"):
            raise ValueError("Official test sources cannot be relabeled for threshold tuning")
    rows = [row for row in all_rows if row.split == "val"]
    if validation_sequences is not None:
        if (
            not validation_sequences
            or len(set(validation_sequences)) != len(validation_sequences)
            or any(type(s) is not int or s not in range(1, 45) for s in validation_sequences)
        ):
            raise ValueError("validation_sequences must contain unique Collective IDs in 1..44")
        sources = {f"collective:seq{s:02d}" for s in validation_sequences}
        if not sources.issubset({row.source_video_id for row in rows}):
            raise ValueError("Requested sequences are not actual validation sources")
        rows = [row for row in rows if row.source_video_id in sources]
    if not rows:
        raise ValueError("Threshold tuning requires nonempty validation scenes")
    return sorted(rows, key=lambda row: (row.dataset, row.clip_id))


def _candidate_provenance(result: DetectionResult):
    metadata = result.metadata
    checkpoint_hash = metadata.get("checkpoint_sha256")
    filters = metadata.get("filter_config")
    counts = metadata.get("filter_counts")
    if (
        metadata.get("threshold_sweep_candidates") is not True
        or filters != asdict(CANDIDATE_FILTERS)
        or metadata.get("untrained") is not False
        or not isinstance(checkpoint_hash, str)
        or len(checkpoint_hash) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_hash)
        or metadata.get("backend_candidate_cap") != 1000
        or metadata.get("backend_box_score_threshold") != 0.0
        or metadata.get("backend_box_nms_threshold") != 1.0
        or metadata.get("before_count_scope") != "after_rpn_and_backend_candidate_cap"
        or metadata.get("truncated") is not False
        or not isinstance(counts, dict)
        or any(
            counts.get(stage) != 0 for stage in ("max_actors", "low_confidence", "nms", "too_small")
        )
        or len(result.boxes) > 1000
    ):
        raise ValueError(
            "Threshold candidate provenance requires a local pretrained checkpoint, confidence=0, "
            "NMS=1, zero size filters, cap=1000, and no additional candidate truncation; "
            "the backend/RPN cap must be explicitly disclosed"
        )


def collect_validation_candidates(
    manifest: Path, output: Path, checkpoint: Path, device: str = "auto"
) -> list[DetectionRecord]:
    """Detect only native validation center images, before confidence/NMS/size filtering.

    These are person candidates after the detector's RPN and 1000-output backend
    cap, not uncapped proposals. Local compatible pretrained weights are required.
    """
    from surveillance.detection.torchvision_detector import TorchvisionPersonDetector

    manifest, output = Path(manifest), Path(output)
    input_paths = {manifest.resolve()}
    if checkpoint is not None:
        input_paths.add(Path(checkpoint).resolve())
    if output.resolve() in input_paths:
        raise ValueError("Candidate output must differ from manifest and checkpoint inputs")
    rows = _validation_rows(manifest)
    try:
        detector = TorchvisionPersonDetector(checkpoint, CANDIDATE_FILTERS, device)
    except pickle.UnpicklingError as error:
        raise ValueError(
            f"Cannot read local detector checkpoint {checkpoint}; "
            "provide a compatible pretrained COCO Faster R-CNN tensor state_dict"
        ) from error
    records = []
    for row in rows:
        path = resolve_path(row.frame_paths[5], Path(manifest))
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing or undecodable validation middle frame: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.detect(torch.from_numpy(image).permute(2, 0, 1).float() / 255)
        if result.image_size != tuple(image.shape[:2]):
            raise ValueError("Candidate image_size differs from native validation center")
        result = replace(
            result,
            metadata={
                **result.metadata,
                "threshold_sweep_candidates": True,
                "reference_image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
        _candidate_provenance(result)
        records.append(
            DetectionRecord(
                row.dataset,
                row.video_id,
                row.source_video_id,
                row.clip_id,
                row.frame_indices[5],
                result,
            )
        )
    write_detections(records, output)
    return records


def tune_detector_thresholds(
    manifest: Path,
    candidates: Path,
    output: Path,
    confidence_values: list[float],
    filters: DetectionConfig,
    iou_threshold: float = 0.5,
    min_recall: float = 0.5,
    validation_sequences: list[int] | None = None,
) -> dict:
    """Select maximum micro detector F1 subject to validation recall >= min_recall.

    Tie order is higher recall then higher confidence. All other detector filters
    and matching IoU remain fixed across trials; this performs no model training,
    feature extraction, test-image inference, or joint threshold optimization.
    A failed recall constraint produces no receipt.
    """
    if not isinstance(filters, DetectionConfig):
        raise ValueError("filters must be a DetectionConfig")
    if not confidence_values or len(confidence_values) > 20:
        raise ValueError("Provide a small confidence sweep of 1..20 values")
    for confidence in confidence_values:
        replace(filters, confidence_threshold=confidence)
    if len(set(confidence_values)) != len(confidence_values):
        raise ValueError("Confidence sweep must not contain duplicates")
    if (
        type(min_recall) not in (int, float)
        or not math.isfinite(min_recall)
        or not 0 <= min_recall <= 1
    ):
        raise ValueError("min_recall must be finite and in [0,1]")
    if (
        type(iou_threshold) not in (int, float)
        or not math.isfinite(iou_threshold)
        or not 0 < iou_threshold <= 1
    ):
        raise ValueError("Matching IoU threshold must be finite and in (0,1]")
    manifest, candidates, output = Path(manifest), Path(candidates), Path(output)
    if output.resolve() in {manifest.resolve(), candidates.resolve()}:
        raise ValueError("Receipt output must differ from manifest and candidates")
    rows = _validation_rows(manifest, validation_sequences)
    expected = {(row.dataset, row.clip_id): row for row in rows}
    records = read_detections(candidates)
    if set(expected) != {record.key for record in records}:
        raise ValueError(
            "Threshold candidates must exactly match the selected validation-only population"
        )
    for record in records:
        row = expected[record.key]
        if (record.video_id, record.source_video_id, record.frame_index) != (
            row.video_id,
            row.source_video_id,
            row.frame_indices[5],
        ):
            raise ValueError(f"Validation candidate identity mismatch: {record.key}")
        if record.pose_feature_path or record.rgb_feature_path:
            raise ValueError("Threshold candidates must precede actor feature extraction")
        _candidate_provenance(record.result)
    checkpoint_hashes = {record.result.metadata["checkpoint_sha256"] for record in records}
    if len(checkpoint_hashes) != 1:
        raise ValueError("Threshold candidates must use a single detector checkpoint")
    trials = []
    for confidence in sorted(confidence_values):
        trial_filters = replace(filters, confidence_threshold=confidence)
        matches = []
        for record in records:
            result = record.result
            detection = filter_detections(
                result.boxes,
                result.scores,
                result.class_ids,
                result.image_size,
                trial_filters,
            )
            height, width = result.image_size
            gt_boxes = torch.tensor(expected[record.key].actor_boxes, dtype=torch.float64)
            gt_boxes *= torch.tensor([width, height, width, height], dtype=torch.float64)
            matches.append(match_boxes(gt_boxes, detection.boxes, iou_threshold))
        metrics = aggregate_matches(matches)
        trials.append(
            {
                "confidence_threshold": confidence,
                **metrics,
                "feasible": metrics["recall"] is not None and metrics["recall"] >= min_recall,
            }
        )
    feasible = [trial for trial in trials if trial["feasible"]]
    if not feasible:
        raise ValueError(
            f"No validation confidence meets minimum recall {min_recall}; no receipt frozen"
        )
    chosen = max(
        feasible,
        key=lambda trial: (
            trial["f1"],
            trial["recall"],
            trial["confidence_threshold"],
        ),
    )
    population = [
        {
            "dataset": row.dataset,
            "clip_id": row.clip_id,
            "source_video_id": row.source_video_id,
            "video_id": row.video_id,
            "frame_index": row.frame_indices[5],
            "actor_boxes": row.actor_boxes,
        }
        for row in rows
    ]
    population_bytes = json.dumps(
        population,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    receipt = {
        "schema_version": 1,
        "split": "val",
        "test_sources_used": False,
        "seed_independent": True,
        "objective": OBJECTIVE,
        "tie_break": ["higher_recall", "higher_confidence"],
        "min_recall": min_recall,
        "iou_threshold": iou_threshold,
        "selected_confidence": chosen["confidence_threshold"],
        "selected_filters": asdict(
            replace(filters, confidence_threshold=chosen["confidence_threshold"])
        ),
        "manifest_path": str(manifest.resolve()),
        "candidates_path": str(candidates.resolve()),
        "candidates_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        "detector_checkpoint_sha256": next(iter(checkpoint_hashes)),
        "candidate_scope": "person candidates after RPN and 1000-output backend cap",
        "validation_source_ids": sorted({row.source_video_id for row in rows}),
        "validation_population": population,
        "validation_population_sha256": hashlib.sha256(population_bytes).hexdigest(),
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt
