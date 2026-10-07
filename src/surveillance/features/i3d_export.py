"""Export supplied upstream I3D blocks; never vendor, recreate or download a backbone."""

import importlib.util
import json
import pickle
import shutil
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from surveillance.features.actor_backbones import METADATA_FILENAME
from surveillance.features.i3d import I3DActorExtractor
from surveillance.features.provenance import file_sha256

PINNED_REVISION = "05783d11f9632b25fe3d50395a9c9bb51f848d6d"
PINNED_SOURCE_SHA256 = "c322c3ca015dab5ee43e4d15707149d510b98931406c4a5d58b27934ed6777ed"
UPSTREAM_URL = "https://github.com/piergiaj/pytorch-i3d"


class _InputGuard(nn.Module):
    """A fixed spatial/temporal trace with a runtime guard and variable batch size."""

    def __init__(self, backbone: torch.jit.ScriptModule, height: int, width: int):
        super().__init__()
        self.backbone = backbone
        self.height = height
        self.width = width

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        torch._assert(clips.dim() == 5, "I3D needs normalized RGB [B,3,10,H,W]")
        torch._assert(clips.size(0) > 0, "I3D needs at least one clip")
        torch._assert(clips.size(1) == 3, "I3D needs RGB channel order")
        torch._assert(clips.size(2) == 10, "I3D needs exactly ten frames")
        torch._assert(clips.size(3) == self.height, "I3D export height mismatch")
        torch._assert(clips.size(4) == self.width, "I3D export width mismatch")
        torch._assert(clips.dtype == torch.float32, "I3D export needs float32")
        return self.backbone(clips)


def _load_model(source: Path) -> nn.Module:
    name = f"_surveillance_external_i3d_{file_sha256(source)}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import supplied I3D source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
        factory = getattr(module, "InceptionI3d", None)
        if not callable(factory):
            raise ValueError("Upstream source must provide InceptionI3d")
        # Intermediate-endpoint construction returns before upstream build().
        # Construct and strictly validate the entire classifier before selecting blocks.
        model = factory(num_classes=400, in_channels=3, final_endpoint="Logits")
    finally:
        sys.modules.pop(name, None)
    if not isinstance(model, nn.Module):
        raise ValueError("I3D factory must return a torch.nn.Module")
    return model.cpu().eval()


def _load_weights(model: nn.Module, checkpoint: Path) -> dict:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Cannot read I3D checkpoint: {checkpoint}") from error
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        )
    ):
        raise ValueError("I3D checkpoint must be a nonempty bare tensor state dictionary")
    expected = model.state_dict()
    # PyTorch 0.3 checkpoints predate this integer bookkeeping buffer. This is
    # the only allowed addition, restricted to actual BatchNorm modules; it is
    # unused by eval inference. No learned tensor or running statistic is omitted.
    counters = {
        f"{name}.num_batches_tracked"
        for name, block in model.named_modules()
        if isinstance(block, nn.modules.batchnorm._BatchNorm) and block.track_running_stats
    }
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if missing - counters or unexpected:
        raise ValueError(
            f"Incompatible I3D checkpoint keys: missing={sorted(missing - counters)[:8]}, "
            f"unexpected={sorted(unexpected)[:8]}"
        )
    loaded = dict(state)
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Incompatible I3D checkpoint shape/dtype: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"I3D checkpoint contains nonfinite values: {key}")
    for key in missing:
        loaded[key] = torch.zeros_like(expected[key])
    model.load_state_dict(loaded, strict=True)
    return {"format": "bare_state_dict", "legacy_bn_counters_initialized": sorted(missing)}


def _endpoint(model: nn.Module) -> nn.Sequential:
    names = getattr(model, "VALID_ENDPOINTS", ())
    if "Mixed_4f" not in names or not isinstance(getattr(model, "end_points", None), dict):
        raise ValueError("Supplied I3D needs registered native Mixed_4f endpoint blocks")
    prefix = names[: names.index("Mixed_4f") + 1]
    if any(name not in model._modules or name not in model.end_points for name in prefix):
        raise ValueError("Upstream I3D endpoint blocks were not registered by build()")
    return nn.Sequential(OrderedDict((name, model._modules[name]) for name in prefix)).eval()


def _check_features(features: torch.Tensor, count: int, size: tuple[int, int]) -> None:
    expected = (count, 832, 3, size[0] // 16, size[1] // 16)
    if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected:
        raise ValueError(f"I3D Mixed_4f must return {expected}; no reshaping is permitted")
    if not torch.isfinite(features).all():
        raise ValueError("I3D Mixed_4f contains nonfinite values")


def export_i3d_features(
    i3d_repo: Path,
    checkpoint: Path,
    output: Path,
    *,
    source_revision: str = PINNED_REVISION,
    expected_source_sha256: str = PINNED_SOURCE_SHA256,
    expected_checkpoint_sha256: str | None = None,
    input_size: tuple[int, int] = (480, 720),
) -> dict:
    """Export local converted RGB I3D as a guarded Mixed_4f feature archive.

    Inputs to the archive are normalized float32 [B,3,10,H,W] RGB in [-1,1].
    The existing adapter consumes [B,10,3,H,W] RGB in [0,1], normalizes once,
    temporally averages maps, resizes to 90x160, and extracts 5x5 actor RoIs.
    Full-resolution benchmark exports use H=480/W=720. Other multiples of 16
    are explicit alternate contracts for offline fixtures, never hardware fallback.
    No network, upstream edits, parameter skipping or existing-file overwrite.
    """
    repo, checkpoint, output = Path(i3d_repo).resolve(), Path(checkpoint).resolve(), Path(output)
    source = repo / "pytorch_i3d.py"
    for path in (source, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Missing local I3D asset: {path}; see docs/i3d-export.md")
    if output.exists():
        raise FileExistsError(f"I3D export already exists: {output}; choose a new path")
    if len(input_size) != 2 or any(
        type(value) is not int or value <= 0 or value % 16 for value in input_size
    ):
        raise ValueError("I3D input_size must be positive [height,width], multiples of 16")
    if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise ValueError("I3D source_revision must be a full lowercase Git commit identifier")
    source_hash, checkpoint_hash = file_sha256(source), file_sha256(checkpoint)
    if source_hash != expected_source_sha256:
        raise ValueError("I3D source SHA256 differs from the supplied expected fingerprint")
    if expected_checkpoint_sha256 is not None and checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError("I3D checkpoint SHA256 differs from the supplied expected fingerprint")
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.random.default_generator.manual_seed(0)
        model = _load_model(source)
        loading = _load_weights(model, checkpoint)
        native = _endpoint(model)
        sample = torch.rand(1, 3, 10, *input_size) * 2 - 1
        reference = native(sample)
        _check_features(reference, 1, input_size)
        # SAME-padding branches are traced for this explicit fixed T/H/W. The
        # scripted guard rejects all other geometry rather than guessing padding.
        # As with HRNet, preserve cuDNN and bind full FP32 convolution semantics.
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=False,
        ):
            traced = torch.jit.trace(native, sample, strict=True, check_trace=False)
        exported = torch.jit.script(_InputGuard(traced, *input_size).eval())
        with torch.jit.optimized_execution(False):
            first, second = exported(sample), exported(sample)
        _check_features(first, 1, input_size)
        if not torch.allclose(first, reference, rtol=1e-5, atol=1e-6):
            raise ValueError("Exported I3D differs from native Mixed_4f")
        if not torch.equal(first, second):
            raise ValueError("Exported I3D is not repeatable in eval mode")
    metadata = {
        "schema_version": 1,
        "architecture": "i3d",
        "endpoint": "Mixed_4f",
        "model_family": "Inception-v1 I3D RGB",
        "output_channels": 832,
        "feature_only": True,
        "inference_only": True,
        "input_frames": 10,
        "feature_shape": [832, 3, input_size[0] // 16, input_size[1] // 16],
        "exported_actor_feature_dim": 20800,
        "provenance": (
            "Converted DeepMind ImageNet+Kinetics I3D RGB weights; not official PyTorch "
            "weights; source identity checked against supplied SHA256"
        ),
        "preprocessing": {
            "color_order": "rgb",
            "input_range": [0, 1],
            "input_size": list(input_size),
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "source_repository": {
            "identifier": UPSTREAM_URL,
            "path": str(repo),
            "revision": source_revision,
            "model_source_sha256": source_hash,
            "verification": (
                "Model-source SHA256 matched supplied expected fingerprint; "
                "revision declaration is not a Git checkout verification"
            ),
        },
        "source_checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash, **loading},
        "exporter": {
            "version": 1,
            "torch_version": str(torch.__version__),
            "method": "native_registered_endpoint_trace_with_scripted_input_guard",
            "cudnn_allow_tf32": False,
            "jit_execution_policy": "unoptimized_inference_only",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="i3d-export-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "features.pt"
        torch.jit.save(
            exported, str(candidate), _extra_files={METADATA_FILENAME: json.dumps(metadata)}
        )
        adapter = I3DActorExtractor(candidate, frozen=True).eval()
        with torch.inference_mode():
            frames = torch.full((1, 10, 3, *input_size), 0.5)
            boxes = torch.tensor([[[0.2, 0.2, 0.8, 0.8]]])
            mask = torch.ones(1, 1, dtype=torch.bool)
            features = adapter(frames, boxes, mask)
            if features.shape != (1, 1, 20800) or not torch.isfinite(features).all():
                raise ValueError("Exported I3D failed existing actor adapter contract")
            if not torch.allclose(features, adapter(frames, boxes, mask), rtol=1e-5, atol=1e-6):
                raise ValueError("Exported I3D actor adapter is not repeatable")
        if (file_sha256(source), file_sha256(checkpoint)) != (source_hash, checkpoint_hash):
            raise ValueError("I3D source/checkpoint changed during export")
        with candidate.open("rb") as stream, output.open("xb") as destination:
            shutil.copyfileobj(stream, destination)
    return {
        "schema_version": 1,
        "valid": True,
        "non_benchmark": True,
        "native_endpoint_verified": True,
        "repeatable": True,
        "adapter_accepted": True,
        "input_shape": [None, 3, 10, *input_size],
        "feature_shape": [None, *metadata["feature_shape"]],
        "feature_dim": 20800,
        "archive": str(output.resolve()),
        "archive_sha256": file_sha256(output),
        "metadata": metadata,
    }
