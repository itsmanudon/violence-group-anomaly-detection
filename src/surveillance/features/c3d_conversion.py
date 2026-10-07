"""Explicit conversion of OpenMMLab's Sports-1M C3D trunk to local FC6 keys."""

import hashlib
import json
from pathlib import Path

import torch

from surveillance.features.c3d import C3DFC6

OPENMMLAB_SHA256 = "dcc47ddcdc7dd62eabc24bc70491b47c82b3b1045b864c52f6bdb669243dfcff"


def convert_openmmlab_state(state: dict) -> dict[str, torch.Tensor]:
    """Strictly map all eight convolutions and FC6; validate then discard FC7.

    No transpose is needed: Conv3d and Linear layouts are native PyTorch.
    Unknown keys, omitted layers and damaged tensors fail before serialization.
    """
    with torch.device("meta"):
        expected = C3DFC6().state_dict()
    mapping = {}
    for target in expected:
        layer, parameter = target.split(".")
        if layer.startswith("conv"):
            layer = {"conv1": "conv1a", "conv2": "conv2a"}.get(layer, layer) + ".conv"
        mapping[f"{layer}.{parameter}"] = target
    shapes = {source: expected[target].shape for source, target in mapping.items()}
    shapes.update({"fc7.weight": (4096, 4096), "fc7.bias": (4096,)})
    if set(state) != set(shapes):
        raise ValueError(
            f"Unexpected C3D keys: missing={sorted(set(shapes) - set(state))}, "
            f"extra={sorted(set(state) - set(shapes))}"
        )
    for name, shape in shapes.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != torch.float32
            or value.device.type == "meta"
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid C3D tensor: {name}; expected finite float32 {shape}")
    return {target: state[source] for source, target in mapping.items()}


def export_openmmlab_c3d(checkpoint: Path, output: Path) -> dict:
    """Export the specifically approved checkpoint; never download or overwrite.

    Preprocessing records the modern upstream RGB configuration. This is not
    an assertion of paper-exact Caffe volume-mean preprocessing equivalence.
    """
    checkpoint, output = Path(checkpoint), Path(output)
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Refuse to overwrite C3D export: {output}")
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != OPENMMLAB_SHA256:
        raise ValueError("C3D checkpoint SHA256 differs from the vetted OpenMMLab asset")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    converted = convert_openmmlab_state(state)
    metadata = {
        "schema_version": 1,
        "architecture": "c3d",
        "endpoint": "post_relu_fc6",
        "feature_dim": 4096,
        "weights": "OpenMMLab Sports-1M pretrained PyTorch C3D",
        "source_checkpoint_sha256": digest,
        "source_url": "https://download.openmmlab.com/mmaction/recognition/c3d/"
        "c3d_sports1m_pretrain_20201016-dcc47ddc.pth",
        "conversion": "native tensor layout; rename conv1a/2a and .conv; omit fc7",
        "discarded_keys": ["fc7.weight", "fc7.bias"],
        "preprocessing": {
            "channel_order": "rgb",
            "mean": [104, 117, 128],
            "std": [1, 1, 1],
            "scale": "raw_0_255",
            "resize_hw": [128, 171],
            "center_crop_hw": [112, 112],
            "temporal_frames": 16,
            "policy": "modern RGB channel means; fixed repository resize/crop",
        },
        "paper_exact": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": converted, "metadata": metadata}, output)
    with output.open("rb") as stream:
        metadata["export_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
