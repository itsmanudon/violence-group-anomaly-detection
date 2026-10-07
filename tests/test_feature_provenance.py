"""Cache integrity and local feature inspection, using synthetic archives."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from test_actor_backbones import archive
from test_collective_validation import manifest, row, sequence

from surveillance.actor_config import DEFAULT_CONFIG
from surveillance.datasets.detected_actors import source_fingerprint


def cache(tmp_path):
    from surveillance.features.provenance import (
        config_fingerprint,
        extraction_config,
        file_sha256,
        scene_fingerprint,
        source_content_fingerprint,
    )

    sequence(tmp_path)
    record = row(tmp_path)
    path = tmp_path / "pose.npy"
    np.save(path, np.zeros((1, 98304), np.float32))
    record = replace(record, pose_feature_path=path.name)
    source = manifest(tmp_path / "rows.jsonl", [record])
    backbone = {"architecture": "pose_hrnet_w32", "endpoint": "pre_final_layer"}
    config = extraction_config("pose", [12, 20], "a" * 64, backbone, "ground_truth")
    metadata = {
        "schema_version": 2,
        "dataset": record.dataset,
        "clip_id": record.clip_id,
        "source_video_id": record.source_video_id,
        "actor_boxes": record.actor_boxes,
        "box_source": "ground_truth",
        "checkpoint_sha256": "a" * 64,
        "backbone": backbone,
        "image_size": [12, 20],
        "shape": [1, 98304],
        "scene_fingerprint": scene_fingerprint(record),
        "source_fingerprint": source_fingerprint(record, source),
        "source_content_fingerprint": source_content_fingerprint(record, source),
        "extraction_config": config,
        "extraction_config_hash": config_fingerprint(config),
        "feature_sha256": file_sha256(path),
    }
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata))
    actor_config = copy.deepcopy(DEFAULT_CONFIG)
    actor_config["data"]["image_size"] = [12, 20]
    expected = {"pose": {"checkpoint_sha256": "a" * 64, **backbone}}
    return source, sidecar, metadata, actor_config, expected


def test_config_and_scene_fingerprints_are_deterministic_and_sensitive(tmp_path):
    from surveillance.features.provenance import config_fingerprint, scene_fingerprint

    assert config_fingerprint({"a": 1, "b": 2}) == config_fingerprint({"b": 2, "a": 1})
    original = row(tmp_path)
    assert scene_fingerprint(original) == scene_fingerprint(
        replace(original, pose_feature_path="x")
    )
    assert scene_fingerprint(original) != scene_fingerprint(replace(original, actor_labels=[1]))


def test_new_ground_truth_cache_is_valid(tmp_path):
    from surveillance.features.provenance import validate_feature_caches

    source, _, _, config, expected = cache(tmp_path)
    report = validate_feature_caches(source, config, expected)
    assert report["valid"] and report["checked_features"] == 1


def test_valid_same_size_image_replacement_invalidates_extracted_cache(tmp_path):
    import cv2

    from surveillance.datasets.collective_validation import DatasetValidationError
    from surveillance.features.provenance import validate_feature_caches

    source, _, _, config, expected = cache(tmp_path)
    frame = tmp_path / "seq01" / "frame0001.jpg"
    assert cv2.imwrite(str(frame), np.full((12, 20, 3), 200, np.uint8))
    with pytest.raises(DatasetValidationError, match="source_content_fingerprint"):
        validate_feature_caches(source, config, expected)


def test_source_content_fingerprint_preserves_order_and_deduplicates_reads(tmp_path, monkeypatch):
    from surveillance.features import provenance

    sequence(tmp_path)
    original = row(tmp_path)
    calls = []
    actual_hash = provenance.file_sha256

    def record_read(path):
        calls.append(path)
        return actual_hash(path)

    monkeypatch.setattr(provenance, "file_sha256", record_read)
    fingerprint = provenance.source_content_fingerprint(original, tmp_path / "rows.jsonl")
    assert len(calls) == len(set(original.frame_paths)) == 5
    # Resolved paths bind ordered identities even when the tiny images have identical bytes.
    changed = replace(original, frame_paths=list(reversed(original.frame_paths)))
    assert provenance.source_content_fingerprint(changed, tmp_path / "rows.jsonl") != fingerprint


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "hash",
        "scene",
        "source",
        "boxes",
        "checkpoint",
        "architecture",
        "endpoint",
        "config",
        "content",
        "shape",
    ],
)
def test_stale_or_malformed_cache_fails_closed(tmp_path, fault):
    from surveillance.datasets.collective_validation import DatasetValidationError
    from surveillance.features.provenance import validate_feature_caches

    source, sidecar, metadata, config, expected = cache(tmp_path)
    fields = {
        "schema": "schema_version",
        "hash": "feature_sha256",
        "scene": "scene_fingerprint",
        "source": "source_fingerprint",
        "boxes": "actor_boxes",
        "checkpoint": "checkpoint_sha256",
        "config": "extraction_config_hash",
        "shape": "shape",
    }
    if fault in fields:
        metadata[fields[fault]] = "invalid"
    elif fault in {"architecture", "endpoint"}:
        metadata["backbone"][fault] = "wrong"
    else:
        np.save(sidecar.with_suffix(".npy"), np.ones((1, 98304), np.float32))
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(DatasetValidationError) as caught:
        validate_feature_caches(source, config, expected)
    assert caught.value.report["errors"]


@pytest.mark.parametrize("modality", ["pose", "rgb"])
def test_raw_inspection_uses_real_extractor_and_local_synthetic_export(tmp_path, modality):
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    spec = spec_from_file_location(
        "inspect_actor_features", Path("scripts/inspect_actor_features.py")
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sequence(tmp_path)
    # Distinct left/right pixels make a swapped actor order observably wrong.
    import cv2

    image = np.zeros((12, 20, 3), np.uint8)
    image[:, 10:] = 255
    for index in range(1, 12):
        assert cv2.imwrite(str(tmp_path / "seq01" / f"frame{index:04d}.jpg"), image)
    record = row(tmp_path)
    record = replace(
        record, actor_boxes=[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]], actor_labels=[0, 1]
    )
    source = manifest(tmp_path / "rows.jsonl", [record])
    report = module.inspect_features(
        source, archive(tmp_path, modality), modality, image_size=(12, 20), max_scenes=1
    )
    assert report["valid"] and report["non_benchmark"]
    assert report["scenes"][0]["actor_order_verified"]
    assert report["scenes"][0]["repeatable"]


@pytest.mark.parametrize("empty", [False, True])
def test_detected_cache_checks_embedded_and_sidecar_provenance(tmp_path, empty):
    import torch

    from surveillance.datasets.detected_actors import normalized_detection_boxes
    from surveillance.detection.person_detector import DetectionResult
    from surveillance.detection.records import (
        DetectionRecord,
        detection_fingerprint,
        write_detections,
    )
    from surveillance.features.provenance import (
        config_fingerprint,
        extraction_config,
        file_sha256,
        validate_feature_caches,
    )

    source, sidecar, metadata, config, expected = cache(tmp_path)
    result = DetectionResult(
        torch.empty(0, 4) if empty else torch.tensor([[2.0, 1.0, 12.0, 11.0]]),
        torch.empty(0) if empty else torch.tensor([0.9]),
        torch.empty(0, dtype=torch.long) if empty else torch.tensor([1]),
        (12, 20),
        {},
    )
    np.save(sidecar.with_suffix(".npy"), np.zeros((len(result.boxes), 98304), np.float32))
    metadata.update(
        box_source="detections",
        actor_boxes=normalized_detection_boxes(result).tolist(),
        detection_fingerprint=detection_fingerprint(result),
        shape=[len(result.boxes), 98304],
        feature_sha256=file_sha256(sidecar.with_suffix(".npy")),
    )
    metadata["extraction_config"] = extraction_config(
        "pose", [12, 20], "a" * 64, metadata["backbone"], "detections"
    )
    metadata["extraction_config_hash"] = config_fingerprint(metadata["extraction_config"])
    sidecar.write_text(json.dumps(metadata))
    record = DetectionRecord(
        "collective",
        "seq01",
        "collective:seq01",
        "seq01:0001",
        1,
        result,
        pose_feature_path="pose.npy",
        feature_metadata={"pose": metadata},
    )
    detections = tmp_path / "detections.jsonl"
    write_detections([record], detections)
    assert validate_feature_caches(source, config, expected, detections)["checked_features"] == 1
    metadata["checkpoint_sha256"] = "b" * 64
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="sidecar"):
        validate_feature_caches(source, config, expected, detections)


def test_cli_missing_checkpoint_writes_non_benchmark_failure(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_actor_features.py",
            "--manifest",
            "missing.jsonl",
            "--checkpoint",
            "missing.pt",
            "--modality",
            "pose",
            "--report",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1, process.stderr
    report = json.loads(output.read_text())
    assert not report["valid"] and report["non_benchmark"] and report["errors"]
