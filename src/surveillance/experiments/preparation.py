"""Explicit local preparation stages; no dataset or model downloads."""

import copy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import torch

from surveillance.datasets.collective import (
    prepare_collective,
    read_actor_manifest,
    write_actor_manifest,
)
from surveillance.datasets.collective_validation import validate_collective
from surveillance.datasets.common import resolve_path
from surveillance.detection.config import DetectionConfig
from surveillance.detection.records import DetectionRecord, write_detections
from surveillance.detection.torchvision_detector import TorchvisionPersonDetector
from surveillance.experiments.protocol import (
    actor_config,
    content_hash,
    feature_expectations,
    file_hash,
)
from surveillance.experiments.runner import detector_selection, run_experiment, write_json
from surveillance.experiments.thresholds import (
    collect_validation_candidates,
    tune_detector_thresholds,
)


def prepare_data(protocol: dict) -> dict:
    """Validate all 44 sources and create train/validation/test scenes without leakage."""
    root = Path(protocol["dataset"]["root"])
    output = Path(protocol["dataset"]["manifest"])
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}; choose a new protocol path")
    validation = protocol["dataset"]["validation_sequences"]
    box_policy = protocol["dataset"].get("annotation_box_policy", "strict")
    report = validate_collective(root, validation_sequences=validation, box_policy=box_policy)
    held_out = {f"collective:seq{sid:02d}" for sid in validation}
    records = [
        replace(row, split="val") if row.source_video_id in held_out else row
        for row in prepare_collective(root, box_policy=box_policy, corrections=[])
    ]
    write_actor_manifest(records, output)
    write_json(Path(protocol["output_root"]) / "dataset_validation.json", report)
    return report


def preflight_experiment(
    protocol: dict, experiment: str, seed: int, max_scenes: int = 10, max_iterations: int = 5
) -> dict:
    """Extract a bounded training/validation subset, then smoke-train on its real caches.

    A full real dataset validation still runs; expensive feature extraction and
    training are bounded and no test features or scores are inspected.
    """
    spec = protocol["experiments"][experiment]
    actor_config(protocol, experiment, seed)
    if spec["box_source"] != "ground_truth":
        raise ValueError(
            "Raw feature preflight uses GT boxes; detected preflight uses existing caches"
        )
    if max_scenes < 1 or max_iterations < 1:
        raise ValueError("Preflight bounds must be positive")
    authority = Path(protocol["dataset"]["manifest"])
    if protocol["evidence_kind"] == "real":
        validate_collective(
            Path(protocol["dataset"]["root"]),
            validation_sequences=protocol["dataset"]["validation_sequences"],
            manifest=authority,
            box_policy=protocol["dataset"].get("annotation_box_policy", "strict"),
        )
    rows = read_actor_manifest(authority)
    selected = []
    for split in ("train", "val"):
        subset = [row for row in rows if row.split == split][:max_scenes]
        if not subset:
            raise ValueError("Preflight requires training and validation scenes")
        selected.extend(
            replace(
                row,
                frame_paths=[str(resolve_path(p, authority)) for p in row.frame_paths],
                pose_feature_path=None,
                rgb_feature_path=None,
            )
            for row in subset
        )
    destination = Path(protocol["output_root"]) / "preflight" / "artifacts" / experiment
    if destination.exists():
        raise FileExistsError(f"Preflight artifacts already exist: {destination}")
    destination.mkdir(parents=True)
    subset_manifest = destination / "raw_subset.jsonl"
    write_actor_manifest(selected, subset_manifest)
    resolved = copy.deepcopy(protocol)
    resolved["dataset"]["manifest"] = str(subset_manifest)
    resolved["experiments"][experiment]["manifest"] = str(destination / "features.jsonl")
    extract_features(resolved, experiment)
    resolved["dataset"]["manifest"] = str(authority)
    return run_experiment(
        resolved,
        experiment,
        seed,
        dry_run=True,
        max_scenes=max_scenes,
        max_iterations=max_iterations,
    )


def inspect_features(protocol: dict, experiment: str, max_scenes: int = 10) -> None:
    """Check local backbone exports on training scenes, without measuring accuracy."""
    spec = protocol["experiments"][experiment]
    root = Path(protocol["output_root"]) / "inspection" / experiment
    root.mkdir(parents=True, exist_ok=True)
    source = Path(protocol["dataset"]["manifest"])
    if protocol["evidence_kind"] == "real":
        validate_collective(
            Path(protocol["dataset"]["root"]),
            validation_sequences=protocol["dataset"]["validation_sequences"],
            manifest=source,
            box_policy=protocol["dataset"].get("annotation_box_policy", "strict"),
        )
    rows = [row for row in read_actor_manifest(source) if row.split == "train"][:max_scenes]
    if not rows or max_scenes < 1:
        raise ValueError("Inspection requires a positive training-only subset")
    subset = root / "training_subset.jsonl"
    write_actor_manifest(
        [
            replace(row, frame_paths=[str(resolve_path(p, source)) for p in row.frame_paths])
            for row in rows
        ],
        subset,
    )
    for modality in feature_expectations(protocol, spec["mode"]):
        checkpoint = protocol["features"][modality]["checkpoint"]
        if checkpoint is None:
            raise ValueError(f"Inspection requires a local {modality} export")
        _script(
            "inspect_actor_features.py",
            [
                "--manifest",
                str(subset),
                "--checkpoint",
                checkpoint,
                "--modality",
                modality,
                "--max-scenes",
                str(max_scenes),
                "--image-size",
                *map(str, protocol["actor"]["data"]["image_size"]),
                "--device",
                protocol["actor"]["device"],
                "--report",
                str(root / f"{modality}.json"),
            ],
        )


def extract_features(protocol: dict, experiment: str) -> None:
    """Reuse the existing pose/RGB extractors and preserve a common actor box contract."""
    spec = protocol["experiments"][experiment]
    detected = spec["box_source"] == "detections"
    source = Path(spec["manifest"]) if detected else Path(protocol["dataset"]["manifest"])
    detection_source = Path(protocol["detector"]["detections"]) if detected else None
    destination = Path(spec["detections"] if detected else spec["manifest"])
    if destination.exists():
        raise FileExistsError(
            f"Feature manifest already exists: {destination}; use a new protocol path"
        )
    modalities = list(feature_expectations(protocol, spec["mode"]))
    for index, modality in enumerate(modalities):
        checkpoint = protocol["features"][modality]["checkpoint"]
        if checkpoint is None:
            raise ValueError(f"Extraction requires a local {modality} export")
        output = (
            destination
            if index == len(modalities) - 1
            else destination.with_suffix(f".{modality}.jsonl")
        )
        args = [
            "--manifest",
            str(source),
            "--output-manifest",
            str(output),
            "--feature-dir",
            str(Path(protocol["output_root"]) / "features" / experiment),
            "--checkpoint",
            checkpoint,
            "--device",
            protocol["actor"]["device"],
            "--image-size",
            *map(str, protocol["actor"]["data"]["image_size"]),
            "--box-source",
            spec["box_source"],
        ]
        if detected:
            args.extend(["--detections", str(detection_source)])
        _script(f"extract_{'pose' if modality == 'pose' else 'i3d'}_features.py", args)
        if detected:
            detection_source = output
        else:
            source = output


def tune_detector(protocol: dict, collect: bool = False) -> dict:
    """Collect validation-only candidates on request, then freeze one confidence sweep."""
    settings = protocol["detector"]
    manifest = Path(protocol["dataset"]["manifest"])
    if Path(settings["selection"]).exists():
        raise FileExistsError("Detector selection already exists; use a new protocol version")
    if collect:
        if settings["checkpoint"] is None:
            raise ValueError("Candidate collection requires a local pretrained detector checkpoint")
        if Path(settings["candidates"]).exists():
            raise FileExistsError(
                "Validation candidates already exist; reuse them or use a new path"
            )
        collect_validation_candidates(
            manifest,
            Path(settings["candidates"]),
            Path(settings["checkpoint"]),
            protocol["actor"]["device"],
        )
    return tune_detector_thresholds(
        manifest,
        Path(settings["candidates"]),
        Path(settings["selection"]),
        settings["confidence_values"],
        DetectionConfig(**settings["filters"]),
        settings["matching_iou"],
        settings["min_recall"],
        protocol["dataset"]["validation_sequences"],
    )


def detect_people(protocol: dict) -> None:
    """Use already selected validation filters for all sources, including held-out test."""
    selection = detector_selection(protocol)
    settings = protocol["detector"]
    destination = Path(settings["detections"])
    if destination.exists():
        raise FileExistsError(f"Detections already exist: {destination}")
    if settings["checkpoint"] is None:
        raise ValueError("Actual detection requires a local pretrained checkpoint")
    detector = TorchvisionPersonDetector(
        settings["checkpoint"],
        DetectionConfig(**selection["selected_filters"]),
        protocol["actor"]["device"],
    )
    manifest = Path(protocol["dataset"]["manifest"])
    records = []
    for row in read_actor_manifest(manifest):
        path = resolve_path(row.frame_paths[5], manifest)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing or undecodable reference frame: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.detect(torch.from_numpy(image).permute(2, 0, 1).float() / 255)
        result = replace(
            result,
            metadata={
                **result.metadata,
                "detector_selection_hash": content_hash(selection),
                "reference_image_sha256": file_hash(path),
            },
        )
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
    write_detections(records, destination)


def _script(name: str, args: list[str]) -> None:
    path = Path(__file__).resolve().parents[3] / "scripts" / name
    subprocess.run([sys.executable, str(path), *args], check=True)
