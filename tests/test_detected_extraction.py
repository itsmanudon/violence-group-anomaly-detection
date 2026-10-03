"""CPU-only synthetic exports exercise live and CLI detected-box feature extraction."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from test_actor_backbones import archive
from test_detected_actor_inputs import cache_fixture, pipeline, result
from test_group_activity_inference import tiny_config

from surveillance.datasets.detected_actors import DetectedActorDataset
from surveillance.detection.records import read_detections
from surveillance.features.hrnet_pose import HRNetPoseExtractor
from surveillance.features.i3d import I3DActorExtractor


@pytest.mark.parametrize("input_mode", ["raw", "precomputed"])
def test_live_detector_uses_middle_frame_and_real_extraction_contracts(tmp_path, input_mode):
    config = tiny_config()
    config["data"].update(input_mode=input_mode, image_size=[12, 20])
    config["model"].update(
        mode="pose_rgb_early_fusion",
        pose_feature_dim=98304,
        rgb_feature_dim=20800,
    )
    config["backbones"] = {
        name: {"checkpoint": str(archive(tmp_path, name)), "frozen": True}
        for name in ("pose", "rgb")
    }
    model = pipeline(config)
    frames = torch.arange(10, dtype=torch.float32)[:, None, None, None]
    frames = (frames / 10).expand(10, 3, 30, 50).clone()
    detection = result([[5, 6, 25, 24], [32, 6, 45, 24]], image_size=(30, 50))
    calls, outputs = [], {}

    class Detector:
        def detect(self, frame):
            calls.append(frame.clone())
            return detection

    extractors = (
        dict(model.system.extractors.items())
        if input_mode == "raw"
        else {
            "pose": HRNetPoseExtractor(config["backbones"]["pose"]["checkpoint"]),
            "rgb": I3DActorExtractor(config["backbones"]["rgb"]["checkpoint"]),
        }
    )

    def capture(name):
        def hook(module, inputs, output):
            outputs[name] = (inputs, output)

        return hook

    for name, extractor in extractors.items():
        extractor.register_forward_hook(capture(name))
    scene = model.predict_clip(
        frames,
        Detector(),
        pose_extractor=extractors["pose"],
        rgb_extractor=extractors["rgb"],
        return_attention=True,
    )
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], frames[5])
    assert outputs["pose"][0][0].shape == (1, 3, 12, 20)
    assert outputs["rgb"][0][0].shape == (1, 10, 3, 12, 20)
    assert outputs["pose"][1].shape == (1, 2, 98304)
    assert outputs["rgb"][1].shape == (1, 2, 20800)
    torch.testing.assert_close(outputs["pose"][1], torch.ones(1, 2, 98304))
    torch.testing.assert_close(outputs["rgb"][1], torch.full((1, 2, 20800), 0.9))
    assert scene["status"] == "ok" and len(scene["actors"]) == 2
    assert scene["actors"][1]["box_pixels"] == [32, 6, 45, 24]


def test_live_empty_detector_bypasses_missing_feature_extractors():
    class Detector:
        def detect(self, frame):
            return result([], image_size=tuple(frame.shape[-2:]))

    scene = pipeline().predict_clip(torch.zeros(10, 3, 12, 20), Detector())
    assert scene["status"] == "no_actors_detected"


def test_detected_feature_clis_chain_pose_and_rgb_without_rewriting_gt(tmp_path):
    manifest, detections, _, _, _ = cache_fixture(tmp_path)
    original = manifest.read_bytes()
    for index in range(10):
        pixels = np.full((100, 200, 3), 20 + index * 10, dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / f"{index}.png"), pixels)
    root = Path(__file__).resolve().parents[1]
    previous = detections
    for name, script in (("pose", "extract_pose_features.py"), ("rgb", "extract_i3d_features.py")):
        output = tmp_path / name / "detected.jsonl"
        process = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / script),
                "--manifest",
                str(manifest),
                "--box-source",
                "detections",
                "--detections",
                str(previous),
                "--output-manifest",
                str(output),
                "--feature-dir",
                str(tmp_path / "extracted"),
                "--checkpoint",
                str(archive(tmp_path, name)),
                "--device",
                "cpu",
                "--image-size",
                "12",
                "20",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        previous = output
    record = read_detections(previous)[0]
    assert record.pose_feature_path and record.rgb_feature_path
    assert set(record.feature_metadata) == {"pose", "rgb"}
    sample = DetectedActorDataset(
        manifest,
        previous,
        mode="pose_rgb_early_fusion",
        image_size=(12, 20),
    )[0]
    assert sample["pose_features"].shape == (2, 98304)
    assert sample["rgb_features"].shape == (2, 20800)
    torch.testing.assert_close(sample["pose_features"], torch.full((2, 98304), 140 / 255))
    torch.testing.assert_close(sample["rgb_features"], torch.full((2, 20800), 130 / 255))
    assert sample["actor_labels"].tolist() == [1, -100]
    assert manifest.read_bytes() == original
