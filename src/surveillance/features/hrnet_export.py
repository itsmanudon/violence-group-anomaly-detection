"""Export user-supplied upstream HRNet; no architecture or weights are vendored.

The upstream factory reads a mapping, so its model file can be imported without
the old training stack. Strictly load the complete pose model before replacing
``final_layer`` with Identity. Trace its native eval forward and script a spatial
input guard. These archives deliberately support frozen inference only.
"""

import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml
from torch import nn

from surveillance.features.actor_backbones import METADATA_FILENAME
from surveillance.features.hrnet_pose import HRNetPoseExtractor
from surveillance.features.provenance import file_sha256

CONFIG_PATH = Path("experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml")
MODEL_PATH = Path("lib/models/pose_hrnet.py")
UPSTREAM_URL = "https://github.com/leoxiaobin/deep-high-resolution-net.pytorch"


class _InputGuard(nn.Module):
    """Keep fixed crop geometry explicit around an eval-only traced upstream model."""

    def __init__(self, backbone: torch.jit.ScriptModule):
        super().__init__()
        self.backbone = backbone

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        torch._assert(crops.dim() == 4, "HRNet needs [N,3,256,192] normalized RGB")
        torch._assert(crops.size(0) > 0, "HRNet needs at least one crop")
        torch._assert(crops.size(1) == 3, "HRNet needs RGB crops")
        torch._assert(crops.size(2) == 256, "HRNet crop height must be 256")
        torch._assert(crops.size(3) == 192, "HRNet crop width must be 192")
        return self.backbone(crops)


def _read_config(path: Path) -> dict:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = config["MODEL"]
        expected = {
            "NAME": "pose_hrnet",
            "NUM_JOINTS": 17,
            "IMAGE_SIZE": [192, 256],  # Upstream YAML uses width, height.
            "HEATMAP_SIZE": [48, 64],
        }
        for key, value in expected.items():
            if model[key] != value:
                raise ValueError(f"{key} must be {value}")
        extra = model["EXTRA"]
        if extra["FINAL_CONV_KERNEL"] != 1:
            raise ValueError("FINAL_CONV_KERNEL must be 1")
        for stage, modules, channels in (
            (2, 1, [32, 64]),
            (3, 4, [32, 64, 128]),
            (4, 3, [32, 64, 128, 256]),
        ):
            settings = extra[f"STAGE{stage}"]
            required = {
                "NUM_MODULES": modules,
                "NUM_BRANCHES": len(channels),
                "NUM_CHANNELS": channels,
                "NUM_BLOCKS": [4] * len(channels),
                "BLOCK": "BASIC",
                "FUSE_METHOD": "SUM",
            }
            if any(settings[key] != value for key, value in required.items()):
                raise ValueError(f"STAGE{stage} must match official W32 settings")
        return config
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Incompatible HRNet configuration {path}: {error}") from error


def _load_model(source: Path, config: dict) -> nn.Module:
    name = f"_surveillance_external_hrnet_{file_sha256(source)}"
    specification = importlib.util.spec_from_file_location(name, source)
    if specification is None or specification.loader is None:
        raise ValueError(f"Cannot import supplied HRNet model: {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        # Execute from source without creating __pycache__ in the external checkout.
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
        factory = getattr(module, "get_pose_net", None)
        if not callable(factory):
            raise ValueError("Supplied HRNet model needs get_pose_net(cfg, is_train=False)")
        model = factory(config, is_train=False)
    finally:
        sys.modules.pop(name, None)
    if not isinstance(model, nn.Module):
        raise ValueError("Supplied HRNet factory did not return a torch.nn.Module")
    return model.cpu().eval()


def _load_weights(model: nn.Module, path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Cannot read HRNet checkpoint: {path}") from error
    wrapped = isinstance(checkpoint, Mapping) and "state_dict" in checkpoint
    state = checkpoint["state_dict"] if wrapped else checkpoint
    if not isinstance(state, Mapping) or not state:
        raise ValueError("HRNet checkpoint must contain a nonempty tensor state dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("HRNet checkpoint state dictionary must have string keys and tensors")
    prefixes = [key.startswith("module.") for key in state]
    if any(prefixes) and not all(prefixes):
        raise ValueError("HRNet checkpoint has mixed module. prefixes")
    state = {key[7:] if all(prefixes) else key: value for key, value in state.items()}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Incompatible HRNet checkpoint keys: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Incompatible HRNet checkpoint shape/dtype for {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"HRNet checkpoint contains nonfinite values: {key}")
    model.load_state_dict(state, strict=True)
    return {"format": "state_dict" if wrapped else "bare", "module_prefix_removed": all(prefixes)}


def _git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        # A supplied copy within another project must not inherit that project's commit.
        root = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if (
            result.returncode == root.returncode == 0
            and Path(root.stdout.strip()).resolve() == repo
        ):
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _check_features(features: torch.Tensor, count: int) -> None:
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != (count, 32, 64, 48):
        raise ValueError("HRNet pre_final_layer must be [N,32,64,48]; no reshaping is permitted")
    if not torch.isfinite(features).all():
        raise ValueError("HRNet pre_final_layer contains nonfinite values")


def _export_model(model: nn.Module) -> tuple[torch.jit.ScriptModule, dict]:
    head = getattr(model, "final_layer", None)
    if (
        not isinstance(head, nn.Conv2d)
        or head.in_channels != 32
        or head.out_channels != 17
        or head.kernel_size != (1, 1)
    ):
        raise ValueError("HRNet final_layer must be the COCO 32-to-17 1x1 heatmap head")
    crops = torch.randn(2, 3, 256, 192, generator=torch.Generator().manual_seed(0))
    captured = []
    hook = head.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].clone()))
    try:
        with torch.no_grad():
            heatmaps = model(crops)
    finally:
        hook.remove()
    if len(captured) != 1 or not isinstance(heatmaps, torch.Tensor):
        raise ValueError("HRNet native forward must call final_layer exactly once")
    if tuple(heatmaps.shape) != (2, 17, 64, 48) or not torch.isfinite(heatmaps).all():
        raise ValueError("HRNet COCO heatmap output must be finite [N,17,64,48]")
    _check_features(captured[0], 2)
    model.final_layer = nn.Identity()
    model.eval()
    with torch.no_grad():
        first, second = model(crops), model(crops)
        _check_features(first, 2)
        if not torch.equal(first, second) or not torch.equal(first, captured[0]):
            raise ValueError("HRNet endpoint bypass is not deterministic/equal to native pre-head")
        # Tracing serializes allow_tf32 into aten::_convolution. A CPU trace
        # must not silently switch to reduced-mantissa TF32 on CUDA, even when
        # the caller later disables it. Preserve native FP32 feature semantics.
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=False,
        ):
            traced = torch.jit.trace(
                model,
                crops,
                check_inputs=[(crops[:1],), (crops[:1].expand(3, -1, -1, -1),)],
                strict=True,
            )
        guarded = torch.jit.script(_InputGuard(traced).eval())
        for count in (1, 2, 3):
            sample = crops[:1].expand(count, -1, -1, -1).contiguous()
            features = guarded(sample)
            _check_features(features, count)
            if not torch.allclose(features, model(sample), rtol=1e-5, atol=1e-6):
                raise ValueError("Exported HRNet features differ from native eval output")
            if not torch.equal(features, guarded(sample)):
                raise ValueError("Exported HRNet eval output is not deterministic")
    return guarded, {
        "repeatable": True,
        "native_endpoint_verified": True,
        "batch_counts": [1, 2, 3],
    }


def export_hrnet_features(
    hrnet_repo: Path,
    checkpoint: Path,
    output: Path,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> dict:
    """Strictly export a supplied COCO W32 checkpoint using supplied upstream code.

    Inputs are trusted local Python code and tensor weights; provenance hashes do
    not establish authenticity. No network access, downloads or upstream edits.
    The archive accepts normalized RGB [N,3,256,192] and returns [N,32,64,48].
    Existing output files are never overwritten. Returned report is JSON-compatible.
    """
    repo, checkpoint, output = Path(hrnet_repo).resolve(), Path(checkpoint).resolve(), Path(output)
    source, config_path = repo / MODEL_PATH, repo / CONFIG_PATH
    for path, description in (
        (repo, "repository directory"),
        (source, "model file"),
        (config_path, "W32 256x192 configuration"),
        (checkpoint, "COCO checkpoint"),
    ):
        if not (path.is_dir() if path == repo else path.is_file()):
            raise FileNotFoundError(
                f"Missing HRNet {description}: {path}; supply local official assets; "
                "see docs/hrnet-export.md. Nothing is downloaded."
            )
    if output.exists():
        raise FileExistsError(f"HRNet export already exists: {output}; choose a new output path")
    checkpoint_hash, source_hash, config_hash = map(file_sha256, (checkpoint, source, config_path))
    if (
        expected_checkpoint_sha256 is not None
        and expected_checkpoint_sha256.lower() != checkpoint_hash
    ):
        raise ValueError("HRNet checkpoint SHA256 differs from the supplied expected fingerprint")
    config = _read_config(config_path)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = _load_model(source, config)
        loading = _load_weights(model, checkpoint)
        exported, checks = _export_model(model)
    metadata = {
        "schema_version": 1,
        "architecture": "pose_hrnet_w32",
        "endpoint": "pre_final_layer",
        "output_channels": 32,
        "feature_only": True,
        "feature_shape": [32, 64, 48],
        "exported_feature_dim": 98304,
        "inference_only": True,
        "provenance": (
            "User-supplied upstream architecture and COCO checkpoint; authenticity not certified"
        ),
        "preprocessing": {
            "color_order": "rgb",
            "input_range": [0, 1],
            "input_size": [256, 192],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "declared_training_dataset": "COCO keypoints (user-supplied; not authenticated)",
            **loading,
        },
        "source_repository": {
            "identifier": UPSTREAM_URL,
            "path": str(repo),
            "git_commit": _git_commit(repo),
            "model_source_sha256": source_hash,
            "config_sha256": config_hash,
        },
        "exporter": {
            "version": 2,
            "method": "native_eval_trace_with_scripted_input_guard",
            "cudnn_allow_tf32": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hrnet-export-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "features.pt"
        torch.jit.save(
            exported, str(candidate), _extra_files={METADATA_FILENAME: json.dumps(metadata)}
        )
        adapter = HRNetPoseExtractor(candidate, frozen=True).eval()
        with torch.inference_mode():
            frames = torch.full((1, 3, 256, 192), 0.5)
            boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
            mask = torch.ones(1, 1, dtype=torch.bool)
            features = adapter(frames, boxes, mask)
            if tuple(features.shape) != (1, 1, 98304) or not torch.isfinite(features).all():
                raise ValueError("Exported HRNet failed existing pose adapter contract")
            if not torch.equal(features, adapter(frames, boxes, mask)):
                raise ValueError("Exported HRNet adapter output is not deterministic")
        if [file_sha256(p) for p in (checkpoint, source, config_path)] != [
            checkpoint_hash,
            source_hash,
            config_hash,
        ]:
            raise ValueError(
                "HRNet checkpoint/source/config changed during export; retry with stable assets"
            )
        # Exclusive creation protects any pre-existing user artifact, including a concurrent writer.
        with candidate.open("rb") as stream, output.open("xb") as destination:
            shutil.copyfileobj(stream, destination)
    return {
        "schema_version": 1,
        "valid": True,
        "non_benchmark": True,
        "adapter_accepted": True,
        "input_shape": [None, 3, 256, 192],
        "feature_shape": [None, 32, 64, 48],
        "feature_dim": 98304,
        "archive": str(output.resolve()),
        "archive_sha256": file_sha256(output),
        "metadata": metadata,
        **checks,
    }
