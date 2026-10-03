"""Small, explicitly artificial fixtures for protocol integration verification.

These arrays are not HRNet/I3D embeddings. Checkpoint fingerprints identify this
fixture generator, not pretrained weights. Outputs can never be benchmark evidence.
"""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from surveillance.datasets.collective import (
    ActorRecord,
    temporal_frame_indices,
    write_actor_manifest,
)
from surveillance.datasets.detected_actors import normalized_detection_boxes, source_fingerprint
from surveillance.detection.config import DetectionConfig
from surveillance.detection.person_detector import filter_detections
from surveillance.detection.records import DetectionRecord, detection_fingerprint, write_detections
from surveillance.experiments.protocol import content_hash, load_protocol
from surveillance.experiments.reporting import aggregate_experiments, aggregate_table_markdown
from surveillance.experiments.runner import freeze_experiment, run_experiment, write_json
from surveillance.experiments.thresholds import CANDIDATE_FILTERS, tune_detector_thresholds
from surveillance.features.provenance import (
    config_fingerprint,
    extraction_config,
    file_sha256,
    scene_fingerprint,
    source_content_fingerprint,
)


def synthetic_protocol(destination: Path) -> dict:
    """Create tiny variable-actor fixtures in a new directory, preserving split IDs."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Synthetic fixture destination already exists: {destination}")
    destination.mkdir(parents=True)
    candidate = Path(__file__).resolve().parents[3] / "configs/experiments/collective_protocol.yaml"
    protocol = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    protocol["evidence_kind"] = "synthetic"
    protocol["protocol_id"] = "synthetic_software_verification_only"
    protocol["seeds"] = [0, 1]
    protocol["dataset"]["root"] = str(destination / "frames")
    protocol["dataset"]["manifest"] = str(destination / "scenes.jsonl")
    protocol["dataset"]["validation_sequences"] = [1]
    protocol["output_root"] = str(destination / "runs")
    actor = protocol["actor"]
    actor["device"] = "cpu"
    actor["data"]["image_size"] = [16, 24]
    actor["model"].update(embedding_dim=8, pose_feature_dim=8, rgb_feature_dim=6)
    actor["model"]["transformer"].update(feedforward_dim=16, reference_size=[16, 24])
    actor["training"].update(
        max_iterations=3,
        batch_size=2,
        validation_interval=1,
        checkpoint_interval=1,
        lr_milestones=[2],
    )
    digest = content_hash({"fixture": "artificial features; not pretrained", "version": 1})
    for modality in ("pose", "rgb"):
        protocol["features"][modality]["checkpoint_sha256"] = digest
    detector = protocol["detector"]
    detector.update(
        checkpoint_sha256=digest,
        candidates=str(destination / "candidates.jsonl"),
        selection=str(destination / "selection.json"),
        detections=str(destination / "detections.jsonl"),
    )
    protocol["experiments"] = {
        "pose_gt": {
            "mode": "pose_only",
            "box_source": "ground_truth",
            "manifest": protocol["dataset"]["manifest"],
        },
        "detected_pose": {
            "mode": "pose_only",
            "box_source": "detections",
            "checkpoint_experiment": "pose_gt",
            "manifest": protocol["dataset"]["manifest"],
            "detections": detector["detections"],
        },
    }
    path = destination / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    protocol = load_protocol(path)
    rng = np.random.default_rng(123)
    rows = []
    for sequence, split, centers in ((4, "train", [1, 11]), (1, "val", [1]), (5, "test", [1, 11])):
        video = f"seq{sequence:02d}"
        folder = destination / "frames" / video
        folder.mkdir(parents=True)
        for index in range(1, 21):
            image = np.zeros((16, 24, 3), dtype=np.uint8)
            image[:, :12] = [20 + index, 80, 150]
            image[:, 12:] = [180, 30 + index, 50]
            if not cv2.imwrite(str(folder / f"frame{index:04d}.jpg"), image):
                raise OSError("Could not save synthetic image fixture")
        for scene, center in enumerate(centers):
            count = 1 if split == "train" and scene == 0 else (3 if scene else 2)
            boxes = [[0.02, 0.05, 0.30, 0.9], [0.40, 0.05, 0.75, 0.9], [0.80, 0.05, 0.99, 0.9]][
                :count
            ]
            indices = temporal_frame_indices(center, 20)
            row = ActorRecord(
                "collective",
                video,
                f"collective:{video}",
                f"{video}:{center:04d}",
                split,
                [str(folder / f"frame{i:04d}.jpg") for i in indices],
                indices,
                boxes,
                [(scene + i // 2) % 5 for i in range(count)],
                scene % 5,
            )
            feature = rng.normal(size=(count, 8)).astype(np.float32)
            feature_path = destination / f"{video}_{center}_pose.npy"
            np.save(feature_path, feature)
            row = replace(row, pose_feature_path=str(feature_path))
            _cache_metadata(
                row, Path(protocol["dataset"]["manifest"]), feature_path, protocol, boxes, None
            )
            rows.append(row)
    manifest = Path(protocol["dataset"]["manifest"])
    write_actor_manifest(rows, manifest)
    candidates = []
    for row in rows:
        if row.split != "val":
            continue
        boxes = torch.tensor(row.actor_boxes) * torch.tensor([24, 16, 24, 16])
        boxes = torch.cat([boxes, torch.tensor([[19.5, 1, 23.9, 14.4]])])
        result = filter_detections(
            boxes,
            torch.tensor([0.95, 0.80, 0.35]),
            torch.ones(3, dtype=torch.long),
            (16, 24),
            CANDIDATE_FILTERS,
        )
        metadata = {
            **result.metadata,
            "threshold_sweep_candidates": True,
            "untrained": False,
            "synthetic": True,
            "checkpoint_sha256": digest,
            "reference_image_sha256": file_sha256(Path(row.frame_paths[5])),
            "backend_candidate_cap": 1000,
            "backend_box_score_threshold": 0.0,
            "backend_box_nms_threshold": 1.0,
            "before_count_scope": "after_rpn_and_backend_candidate_cap",
        }
        result = replace(result, metadata=metadata)
        candidates.append(
            DetectionRecord(
                row.dataset,
                row.video_id,
                row.source_video_id,
                row.clip_id,
                row.frame_indices[5],
                result,
            )
        )
    write_detections(candidates, Path(detector["candidates"]))
    selection = tune_detector_thresholds(
        manifest,
        Path(detector["candidates"]),
        Path(detector["selection"]),
        detector["confidence_values"],
        DetectionConfig(**detector["filters"]),
        detector["matching_iou"],
        detector["min_recall"],
        [1],
    )
    detections = []
    for row in rows:
        if row.split == "test" and row.frame_indices[5] == 1:
            boxes, scores = torch.empty((0, 4)), torch.empty(0)
        else:
            boxes = torch.tensor(row.actor_boxes[:2]) * torch.tensor([24, 16, 24, 16])
            scores = torch.full((len(boxes),), 0.95)
            if row.split == "test":
                boxes = boxes[:1]
                boxes = torch.cat([boxes, torch.tensor([[14, 0, 18.9, 4.1]])])
                scores = torch.tensor([0.95, 0.8])
        result = filter_detections(
            boxes,
            scores,
            torch.ones(len(boxes), dtype=torch.long),
            (16, 24),
            DetectionConfig(**selection["selected_filters"]),
        )
        result = replace(
            result,
            metadata={
                **result.metadata,
                "synthetic": True,
                "checkpoint_sha256": digest,
                "untrained": False,
                "reference_image_sha256": file_sha256(Path(row.frame_paths[5])),
                "detector_selection_hash": content_hash(selection),
            },
        )
        record = DetectionRecord(
            row.dataset,
            row.video_id,
            row.source_video_id,
            row.clip_id,
            row.frame_indices[5],
            result,
        )
        feature_path = destination / f"{row.video_id}_{row.frame_indices[5]}_det_pose.npy"
        np.save(feature_path, rng.normal(size=(len(result.boxes), 8)).astype(np.float32))
        metadata = _cache_metadata(
            row,
            manifest,
            feature_path,
            protocol,
            normalized_detection_boxes(result).tolist(),
            result,
        )
        detections.append(
            replace(
                record, pose_feature_path=str(feature_path), feature_metadata={"pose": metadata}
            )
        )
    write_detections(detections, Path(detector["detections"]))
    return protocol


def _cache_metadata(row, manifest, path, protocol, boxes, result):
    box_source = "detections" if result is not None else "ground_truth"
    backbone = {"architecture": "pose_hrnet_w32", "endpoint": "pre_final_layer", "synthetic": True}
    digest = protocol["features"]["pose"]["checkpoint_sha256"]
    configuration = extraction_config("pose", [16, 24], digest, backbone, box_source)
    metadata = {
        "schema_version": 2,
        "dataset": row.dataset,
        "clip_id": row.clip_id,
        "source_video_id": row.source_video_id,
        "actor_boxes": boxes,
        "box_source": box_source,
        "scene_fingerprint": scene_fingerprint(row),
        "source_fingerprint": source_fingerprint(row, manifest),
        "source_content_fingerprint": source_content_fingerprint(row, manifest),
        "checkpoint_sha256": digest,
        "backbone": backbone,
        "image_size": [16, 24],
        "shape": [len(boxes), 8],
        "feature_sha256": file_sha256(path),
        "extraction_config": configuration,
        "extraction_config_hash": config_fingerprint(configuration),
    }
    if result is not None:
        metadata["detection_fingerprint"] = detection_fingerprint(result)
    write_json(path.with_suffix(".json"), metadata)
    return metadata


def smoke_workflow(destination: Path) -> dict:
    """Exercise config -> seeds -> checkpoints -> GT/detected -> aggregation/errors."""
    protocol = synthetic_protocol(destination)
    root = Path(protocol["output_root"])
    freeze_experiment(protocol, "pose_gt")
    for seed in protocol["seeds"]:
        run_experiment(protocol, "pose_gt", seed)
    freeze_experiment(protocol, "detected_pose")
    aggregates = []
    for experiment in ("pose_gt", "detected_pose"):
        if experiment == "detected_pose":
            for seed in protocol["seeds"]:
                run_experiment(protocol, experiment, seed)
        paths = [root / experiment / f"seed_{seed}" / "metrics.json" for seed in protocol["seeds"]]
        aggregate = aggregate_experiments(paths)
        aggregates.append(aggregate)
        write_json(root / f"{experiment}_aggregate.json", aggregate)
    (root / "ablations.md").write_text(aggregate_table_markdown(aggregates), encoding="utf-8")
    result = {
        "evidence_kind": "synthetic",
        "non_benchmark": True,
        "seeds": protocol["seeds"],
        "experiments": [a["experiment"] for a in aggregates],
        "output_root": str(root),
    }
    write_json(root / "smoke_verification.json", result)
    return result
