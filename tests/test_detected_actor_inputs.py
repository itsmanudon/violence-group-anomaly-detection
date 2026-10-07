"""Detected actor adaptation keeps feature, geometry, and attention indices aligned."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_group_activity_inference import tiny_config

from surveillance.datasets.collective import ActorRecord, write_actor_manifest
from surveillance.datasets.detected_actors import (
    DetectedActorDataset,
    collate_actor_inputs,
    normalized_detection_boxes,
    source_fingerprint,
)
from surveillance.detection.person_detector import DetectionResult
from surveillance.detection.records import DetectionRecord, detection_fingerprint, write_detections
from surveillance.inference.detected_group_activity import DetectedGroupActivityPipeline
from surveillance.training.actor_transformer_trainer import ActorTransformerSystem


def result(boxes=None, image_size=(100, 200)):
    boxes = [[20, 20, 100, 80], [130, 20, 180, 80]] if boxes is None else boxes
    return DetectionResult(
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.linspace(0.9, 0.6, len(boxes)),
        torch.ones(len(boxes), dtype=torch.long),
        image_size,
        {"backend": "synthetic fixture"},
    )


def cache_fixture(tmp_path, *, detected=None, feature_dim=8):
    row = ActorRecord(
        "fixture",
        "video",
        "source",
        "clip",
        "test",
        [f"{i}.png" for i in range(10)],
        list(range(10)),
        [[0.1, 0.2, 0.5, 0.8]],
        [1],
        0,
    )
    manifest, detections = tmp_path / "gt.jsonl", tmp_path / "det.jsonl"
    write_actor_manifest([row], manifest)
    detected = result() if detected is None else detected
    features = tmp_path / "features.npy"
    np.save(
        features,
        np.arange(len(detected.boxes) * feature_dim, dtype=np.float32).reshape(
            len(detected.boxes), feature_dim
        ),
    )
    metadata = {
        "detection_fingerprint": detection_fingerprint(detected),
        "source_fingerprint": source_fingerprint(row, manifest),
        "feature_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
    }
    record = DetectionRecord(
        row.dataset,
        row.video_id,
        row.source_video_id,
        row.clip_id,
        5,
        detected,
        pose_feature_path=features.name,
        feature_metadata={"pose": metadata},
    )
    write_detections([record], detections)
    return manifest, detections, row, record, features


def pipeline(config=None):
    torch.manual_seed(13)
    return DetectedGroupActivityPipeline(ActorTransformerSystem(config or tiny_config()))


def test_variable_actor_collation_has_true_valid_mask_without_class_targets():
    boxes = normalized_detection_boxes(result())
    samples = [
        {"actor_boxes": boxes[:count], "pose_features": torch.full((count, 8), float(count))}
        for count in (1, 2)
    ]
    batch = collate_actor_inputs(samples)
    assert batch["actor_valid_mask"].tolist() == [[True, False], [True, True]]
    assert batch["pose_features"].shape == (2, 2, 8)
    assert batch["pose_features"][0, 1].count_nonzero() == 0
    assert "actor_labels" not in batch and "group_labels" not in batch
    with pytest.raises(ValueError, match="nonempty"):
        collate_actor_inputs([{"actor_boxes": torch.empty(0, 4)}])


def test_normalization_preserves_geometry_under_anisotropic_resize():
    normalized = normalized_detection_boxes(result([[50, 10, 150, 90]]))
    torch.testing.assert_close(normalized, torch.tensor([[0.25, 0.1, 0.75, 0.9]]))
    torch.testing.assert_close(
        normalized * torch.tensor([160, 90, 160, 90]),
        torch.tensor([[40.0, 9.0, 120.0, 81.0]]),
    )


def test_cache_transfers_only_matched_labels_and_preserves_ground_truth(tmp_path):
    manifest, detections, _, _, _ = cache_fixture(tmp_path)
    original = manifest.read_bytes()
    sample = DetectedActorDataset(manifest, detections, pose_feature_dim=8)[0]
    assert sample["actor_labels"].tolist() == [1, -100]
    assert sample["pose_features"].shape == (2, 8)
    scene = pipeline().predict_sample(sample, return_attention=True)
    assert [a["label"] for a in scene["actors"]] == [1, None]
    assert [a["matched_gt_index"] for a in scene["actors"]] == [0, None]
    assert manifest.read_bytes() == original
    assert len(scene["actors"]) == 2  # The unmatched detection participates in inference.
    json.dumps(scene, allow_nan=False)


def test_attention_confidence_and_absolute_boxes_follow_actor_permutations():
    model = pipeline()
    detection = result()
    features = torch.randn(2, 8)
    first = model.predict_detections(detection, pose_features=features, return_attention=True)
    permutation = torch.tensor([1, 0])
    permuted = DetectionResult(
        detection.boxes[permutation],
        detection.scores[permutation],
        detection.class_ids[permutation],
        detection.image_size,
    )
    second = model.predict_detections(
        permuted,
        pose_features=features[permutation],
        return_attention=True,
    )
    torch.testing.assert_close(
        torch.tensor(first["group_probabilities"]), torch.tensor(second["group_probabilities"])
    )
    for index, original_index in enumerate(permutation.tolist()):
        actor = second["actors"][index]
        assert actor["box_pixels"] == detection.boxes[original_index].tolist()
        assert actor["detection_score"] == pytest.approx(detection.scores[original_index].item())
        assert "label" not in actor
        torch.testing.assert_close(
            torch.tensor(actor["probabilities"]),
            torch.tensor(first["actors"][original_index]["probabilities"]),
        )
    assert first["attention"]
    for branch, weights in first["attention"].items():
        expected = torch.tensor(weights)[..., permutation, :][..., permutation]
        assert expected.shape[-2:] == (2, 2)
        torch.testing.assert_close(torch.tensor(second["attention"][branch]), expected)


def test_empty_detections_abstain_without_loading_features_or_calling_model(tmp_path, monkeypatch):
    manifest, detections, _, _, features = cache_fixture(tmp_path, detected=result([]))
    features.unlink()
    model = pipeline()

    def forbidden(*args, **kwargs):
        raise AssertionError("An empty scene must bypass the transformer")

    monkeypatch.setattr(model.system, "forward", forbidden)
    scene = model.predict_manifest(manifest, detections, return_attention=True)[0]
    assert scene["status"] == "no_actors_detected"
    assert scene["group_prediction"] is None and scene["group_probabilities"] is None
    assert scene["actors"] == [] and scene["attention"] == {}
    assert scene["group_label"] == 0


@pytest.mark.parametrize("mutation", ["boxes", "source", "bytes", "dimension"])
def test_cached_features_reject_stale_or_misaligned_content(tmp_path, mutation):
    manifest, detections, row, record, features = cache_fixture(tmp_path)
    expected = "stale"
    if mutation == "boxes":
        record = replace(record, result=result([[21, 20, 100, 80], [130, 20, 180, 80]]))
    elif mutation == "source":
        write_actor_manifest(
            [replace(row, frame_paths=[f"other/{i}.png" for i in range(10)])], manifest
        )
    else:
        np.save(features, np.ones((2, 7 if mutation == "dimension" else 8), dtype=np.float32))
        expected = "hash"
        if mutation == "dimension":
            record.feature_metadata["pose"]["feature_sha256"] = hashlib.sha256(
                features.read_bytes()
            ).hexdigest()
            expected = "expected finite"
    write_detections([record], detections)
    with pytest.raises(ValueError, match=expected):
        DetectedActorDataset(manifest, detections, pose_feature_dim=8)[0]


def test_dataset_rejects_wrong_detection_source(tmp_path):
    manifest, detections, _, record, _ = cache_fixture(tmp_path)
    write_detections([replace(record, source_video_id="unrelated")], detections)
    with pytest.raises(ValueError, match="source/reference"):
        DetectedActorDataset(manifest, detections, pose_feature_dim=8)


def test_decimal_gt_coordinates_match_exact_pixels_at_iou_one(tmp_path):
    from surveillance.detection.matching import match_manifest

    manifest, detections, row, _, _ = cache_fixture(
        tmp_path,
        detected=result([[60, 20, 100, 80]]),
    )
    write_actor_manifest([replace(row, actor_boxes=[[0.3, 0.2, 0.5, 0.8]])], manifest)
    offline = match_manifest(manifest, detections, iou_threshold=1.0)[0]
    assert offline["actor_labels"] == [1]
    sample = DetectedActorDataset(
        manifest,
        detections,
        pose_feature_dim=8,
        iou_threshold=1.0,
    )[0]
    assert sample["actor_labels"].tolist() == [1]
    assert sample["matching"].ious == [1.0]


def test_cache_rejects_features_extracted_at_different_image_size(tmp_path):
    manifest, detections, _, record, _ = cache_fixture(tmp_path)
    record.feature_metadata["pose"]["image_size"] = [480, 720]
    write_detections([record], detections)
    with pytest.raises(ValueError, match="image_size|image size|preprocessing"):
        DetectedActorDataset(
            manifest,
            detections,
            pose_feature_dim=8,
            image_size=(12, 20),
        )[0]
