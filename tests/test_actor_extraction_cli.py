"""Exercise actual extraction CLIs and relocated manifests using synthetic exports."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from test_actor_backbones import archive

from surveillance.datasets.actor_batch import collate_actors
from surveillance.datasets.collective import (
    ActorFeatureDataset,
    ActorRecord,
    read_actor_manifest,
    write_actor_manifest,
)
from surveillance.datasets.common import resolve_path
from surveillance.features.hrnet_pose import HRNetPoseExtractor
from surveillance.features.i3d import I3DActorExtractor

ROOT = Path(__file__).resolve().parents[1]


def test_extraction_cli_preserves_paths_actor_order_and_raw_parity(tmp_path):
    paths = []
    for index in range(10):
        path = tmp_path / f"frame{index}.png"
        pixels = np.arange(12 * 20 * 3, dtype=np.uint16).reshape(12, 20, 3)
        assert cv2.imwrite(str(path), ((pixels + index * 9) % 256).astype(np.uint8))
        paths.append(path.name)
    record = ActorRecord(
        dataset="synthetic",
        video_id="scene",
        source_video_id="scene",
        clip_id="scene:5",
        split="test",
        frame_paths=paths,
        frame_indices=list(range(10)),
        actor_boxes=[[0, 0, 0.5, 1], [0.5, 0.2, 1, 0.8]],
        actor_labels=[0, 1],
        group_label=0,
    )
    manifest = tmp_path / "input.jsonl"
    write_actor_manifest([record], manifest)
    raw = collate_actors(
        [ActorFeatureDataset(manifest, None, input_mode="raw", image_size=(12, 20))[0]]
    )
    for kind, extractor_type, dim in (
        ("pose", HRNetPoseExtractor, 98304),
        ("rgb", I3DActorExtractor, 20800),
    ):
        checkpoint = archive(tmp_path, kind)
        output = tmp_path / kind / "features.jsonl"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / f"extract_{kind if kind == 'pose' else 'i3d'}_features.py"),
                "--manifest",
                str(manifest),
                "--output-manifest",
                str(output),
                "--feature-dir",
                str(tmp_path / "arrays"),
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cpu",
                "--image-size",
                "12",
                "20",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        saved = read_actor_manifest(output)[0]
        assert [resolve_path(p, output).resolve() for p in saved.frame_paths] == [
            tmp_path / p for p in paths
        ]
        feature_path = resolve_path(getattr(saved, f"{kind}_feature_path"), output)
        values = np.load(feature_path)
        assert values.shape == (2, dim) and np.isfinite(values).all()
        extractor = extractor_type(checkpoint).eval()
        with torch.inference_mode():
            frames = raw["frames"][:, 5] if kind == "pose" else raw["frames"]
            expected = extractor(frames, raw["actor_boxes"], raw["actor_valid_mask"])[0]
        np.testing.assert_allclose(values, expected.numpy(), atol=1e-6)
        sidecar = json.loads(feature_path.with_suffix(".json").read_text())
        assert sidecar["backbone"] == extractor.metadata
        assert len(sidecar["checkpoint_sha256"]) == 64
        assert sidecar["actor_boxes"] == record.actor_boxes
        manifest = output
    both = ActorFeatureDataset(manifest, None, mode="pose_rgb_late_fusion")[0]
    assert both["pose_features"].shape == (2, 98304)
    assert both["rgb_features"].shape == (2, 20800)


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter path semantics")
def test_manifest_reference_across_windows_drives():
    spec = importlib.util.spec_from_file_location(
        "actor_features_cli", ROOT / "scripts/_actor_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.manifest_reference(Path("C:/data/frames/f.jpg"), Path("D:/runs/m.jsonl"))
        == "C:/data/frames/f.jpg"
    )
