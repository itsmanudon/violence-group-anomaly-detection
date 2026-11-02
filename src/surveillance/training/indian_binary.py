"""Frozen-feature binary baseline with protected splits and deterministic resume."""

import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from surveillance.datasets.dcsass_audit import sha256
from surveillance.datasets.indian_domain import write_new_json
from surveillance.datasets.indian_split import verify_split
from surveillance.evaluation.indian_binary import calibrate_threshold, evaluate_binary
from surveillance.experiments.environment import environment_metadata
from surveillance.experiments.protocol import content_hash
from surveillance.models.indian_binary import FrozenBagBinaryHead
from surveillance.training.sultani_trainer import seed_everything


def binary_class_weights(labels: list[int]) -> list[float]:
    """Training-only N/(2*n_class); no valid negative examples are discarded."""
    if not labels or set(labels) != {0, 1}:
        raise ValueError("Both training classes are required for balanced weights")
    counts = np.bincount(labels, minlength=2)
    return (len(labels) / (2 * counts)).tolist()


def training_indices(labels, count, generator, sampling):
    if sampling == "uniform":
        return torch.randperm(len(labels), generator=generator)[:count].tolist()
    if sampling != "balanced":
        raise ValueError("Sampling must be uniform or balanced")
    weights = binary_class_weights(labels)
    probabilities = torch.tensor([weights[label] for label in labels], dtype=torch.float64)
    return torch.multinomial(probabilities, count, replacement=True, generator=generator).tolist()


def _device_memory(device) -> dict:
    if not str(device).startswith("cuda"):
        return {"gpu": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "total_bytes": total,
        "free_bytes_at_observation": free,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def profile_binary_device(output: Path, *, batch_sizes=(1, 8, 32), steps=10, precision="fp32"):
    """Bounded synthetic-head engineering profile, never a footage/model benchmark."""
    if not torch.cuda.is_available() or "4070" not in torch.cuda.get_device_name(0):
        raise ValueError("This local profile requires the available RTX 4070")
    if (
        type(steps) is not int
        or not 1 <= steps <= 100
        or not batch_sizes
        or any(type(b) is not int or not 1 <= b <= 64 for b in batch_sizes)
        or precision not in {"fp32", "amp_fp16"}
    ):
        raise ValueError("Use bounded batch sizes/steps and fp32 or amp_fp16")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite device profile: {output}")
    seed_everything(7)
    device = torch.device("cuda:0")
    before = _device_memory(device)
    trials = []
    for batch_size in batch_sizes:
        torch.cuda.reset_peak_memory_stats(device)
        model = FrozenBagBinaryHead(4096).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp_fp16")
        features = torch.randn(batch_size, 32, 4096, device=device)
        labels = torch.arange(batch_size, device=device) % 2

        def batch(model, optimizer, scaler, features, labels):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=precision == "amp_fp16", dtype=torch.float16):
                logits = model(features)
                loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        for _ in range(3):
            batch(model, optimizer, scaler, features, labels)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(steps):
            batch(model, optimizer, scaler, features, labels)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        trials.append(
            {
                "batch_size": batch_size,
                "steps": steps,
                "seconds": seconds,
                "examples_per_second": batch_size * steps / seconds,
                "memory": _device_memory(device),
            }
        )
        del model, optimizer, features, labels, scaler
    report = {
        "evidence_kind": "synthetic_head_engineering_only",
        "precision": precision,
        "environment": environment_metadata(content_hash({"precision": precision}), 7, str(device)),
        "device_before": before,
        "feature_shape_per_example": [32, 4096],
        "trials": trials,
        "real_feature_extraction_latency": None,
        "real_dataset_loading_throughput": None,
        "scope": (
            "resident synthetic C3D-shaped tensors; forward/backward/Adam; "
            "no real data or checkpoints"
        ),
        "code": _code_identity(),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "profile.json", report)
    return report


def committed_protocol(protocol: Path) -> dict:
    """Identity of an evaluation protocol that is tracked and unmodified in git HEAD."""
    protocol = Path(protocol).resolve()
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10
        )

    relative = str(protocol.relative_to(root))
    if git("ls-files", "--error-unmatch", relative).returncode != 0:
        raise ValueError("Evaluation protocol must be committed before test features exist")
    if git("diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        raise ValueError("Evaluation protocol has uncommitted changes")
    commit = git("log", "-1", "--format=%H", "--", relative).stdout.strip()
    return {"path": relative, "sha256": sha256(protocol), "commit": commit}


def cache_c3d(
    split_dir: Path,
    checkpoint: Path,
    output: Path,
    *,
    device="cpu",
    batch_size=2,
    splits=("train", "val"),
    strict_frame_count=True,
    test_protocol: Path | None = None,
):
    """Extract the requested splits into a fresh cache, binding all source bytes.

    Test features require a committed, unmodified evaluation protocol; its identity
    is recorded so the one-time test evaluation is auditable.
    """
    from surveillance.features.c3d import C3DExtractor

    if Path(output).exists():
        raise FileExistsError(f"Refuse to overwrite feature cache: {output}")
    if not set(splits) or set(splits) - {"train", "val", "test"}:
        raise ValueError("Splits must be a nonempty subset of train/val/test")
    protocol = None
    if "test" in splits:
        if test_protocol is None:
            raise ValueError("Protected test features require --test-protocol")
        protocol = committed_protocol(test_protocol)
    rows, receipt = verify_split(split_dir)
    checkpoint_hash = sha256(checkpoint)
    extraction_started = time.perf_counter()
    extractor = C3DExtractor(checkpoint, device, batch_size, (104, 117, 128), "rgb")
    model_loading_seconds = time.perf_counter() - extraction_started
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    features = {}
    video_timings = []
    for index, row in enumerate(r for r in rows if r.split in set(splits)):
        path = output / f"bag-{index:06d}.npy"  # never use untrusted clip IDs as paths
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        video_started = time.perf_counter()
        # Only non-strict decoding needs the keyword; strict extractors keep their signature.
        options = {} if strict_frame_count else {"strict_frame_count": False}
        array = extractor.extract_video(Path(row.path), 32, **options)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        video_timings.append(
            {"clip_id": row.video_id, "extraction_seconds": time.perf_counter() - video_started}
        )
        if array.shape != (32, 4096) or not np.isfinite(array).all():
            raise ValueError("C3D produced invalid feature bag")
        if sha256(Path(row.path)) != receipt["video_sha256"][row.video_id]:
            raise ValueError("Video changed during feature extraction")
        with path.open("xb") as stream:
            np.save(stream, array)
        features[row.video_id] = {"path": path.name, "sha256": sha256(path)}
    if sha256(checkpoint) != checkpoint_hash:
        raise ValueError("C3D checkpoint changed during extraction")
    registration = {
        "schema_version": 1,
        "evidence_kind": "frozen_feature_cache_not_an_evaluation",
        "split_freeze_hash": receipt["freeze_hash"],
        "checkpoint_sha256": checkpoint_hash,
        "feature_dim": 4096,
        "num_segments": 32,
        "preprocessing": {
            "mean": [104, 117, 128],
            "channel_order": "rgb",
            "aggregation": "32_l2_normalized_c3d_segments",
        },
        "features": features,
        "splits": sorted(splits),
        "strict_frame_count": strict_frame_count,
        "device": str(device),
        "protected_test_features_extracted": "test" in splits,
        "test_protocol": protocol,
        "timing": {
            "model_loading_seconds": model_loading_seconds,
            "total_cache_seconds": time.perf_counter() - extraction_started,
            "video_extraction": video_timings,
        },
    }
    write_new_json(output / "registration.json", registration)
    return registration


def load_binary_checkpoint(path: Path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("checkpoint_type") != "indian_frozen_bag_binary_v1" or saved.get("classes") != [
        "non_violence",
        "violence",
    ]:
        raise ValueError("Expected isolated Indian binary checkpoint")
    model = FrozenBagBinaryHead(saved["feature_dim"])
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device).eval(), saved


def calibrate_binary_checkpoint(
    split_dir: Path, checkpoint: Path, output: Path, *, min_recall=0.9, candidates=None
) -> dict:
    """Freeze validation-only calibration for a registered, validation-selected head."""
    if Path(output).exists():
        raise FileExistsError(f"Refuse to overwrite calibration: {output}")
    rows, receipt = verify_split(split_dir)
    checkpoint = Path(checkpoint)
    registration = json.loads((checkpoint.parent / "registration.json").read_text(encoding="utf-8"))
    _, saved = load_binary_checkpoint(checkpoint)
    if checkpoint.resolve() != (checkpoint.parent / "best.pt").resolve():
        raise ValueError("Calibrate the validation-selected best.pt only")
    if (
        saved["config_hash"] != registration["config_hash"]
        or content_hash(registration["settings"]) != saved["config_hash"]
        or registration["settings"]["split_freeze_hash"] != receipt["freeze_hash"]
    ):
        raise ValueError("Calibration checkpoint registration/split mismatch")
    expected = {row.video_id: row for row in rows if row.split == "val"}
    predictions = saved.get("validation_predictions", [])
    if len(predictions) != len(expected) or {p["clip_id"] for p in predictions} != set(expected):
        raise ValueError(
            "Calibration predictions must exactly match protected validation membership"
        )
    for prediction in predictions:
        row = expected[prediction["clip_id"]]
        if (prediction["label"], prediction["source_video_id"]) != (row.label, row.source_video_id):
            raise ValueError("Calibration prediction label/source differs from frozen validation")
    labels = [p["label"] for p in predictions]
    scores = [p["violence_score"] for p in predictions]
    selected = calibrate_threshold(
        labels,
        scores,
        split="val",
        min_recall=min_recall,
        candidates=candidates if candidates is not None else [i / 10 for i in range(1, 10)],
    )
    result = {
        **selected,
        "evidence_kind": saved["evidence_kind"],
        "checkpoint_sha256": sha256(checkpoint),
        "training_registration_sha256": sha256(checkpoint.parent / "registration.json"),
        "split_freeze_hash": receipt["freeze_hash"],
        "validation_prediction_hash": content_hash(predictions),
        "validation_clip_ids": sorted(expected),
        "test_predictions_used": False,
        "metrics_at_0_5": evaluate_binary(labels, scores),
        "selected_metrics": evaluate_binary(labels, scores, selected["selected_threshold"]),
        "code": _code_identity(),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "calibration.json", result)
    return result


def _code_identity():
    root = Path(__file__).resolve().parents[3]
    names = [
        "src/surveillance/datasets/indian_domain.py",
        "src/surveillance/datasets/indian_split.py",
        "src/surveillance/models/indian_binary.py",
        "src/surveillance/training/indian_binary.py",
        "src/surveillance/evaluation/indian_binary.py",
    ]
    files = {name: sha256(root / name) for name in names}
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"],
        capture_output=True,
        check=True,
        timeout=10,
    )
    import hashlib

    return {"files": files, "tracked_diff_sha256": hashlib.sha256(result.stdout).hexdigest()}


def train_binary(
    split_dir: Path,
    cache: Path,
    output: Path,
    *,
    steps=100,
    seed=7,
    learning_rate=1e-3,
    batch_size=8,
    device="cpu",
    resume: Path | None = None,
    class_weighting="none",
    sampling="uniform",
    patience=0,
    precision="fp32",
    accumulation_steps=1,
    train_groups=None,
    relabel: dict[str, int] | None = None,
    target_name="violence",
):
    """Train only on training labels; select on validation PR-AUC (then F1).

    train_groups restricts training to those source groups (learning curves);
    validation is never subset. relabel substitutes a 0/1 target for every
    train/val clip (e.g. a collection-shortcut probe); both are bound in the
    config hash. Only this new run's last.pt/best.pt/history are mutable. Existing output dirs
    require an explicit same-run last.pt resume. Test features are never loaded.
    """
    if (
        type(steps) is not int
        or steps < 1
        or type(batch_size) is not int
        or batch_size < 1
        or type(seed) is not int
        or seed < 0
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Invalid training hyperparameters")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable; smoke test before training")
    if (
        class_weighting not in {"none", "balanced"}
        or sampling not in {"uniform", "balanced"}
        or type(patience) is not int
        or patience < 0
        or precision not in {"fp32", "amp_fp16"}
        or type(accumulation_steps) is not int
        or accumulation_steps < 1
    ):
        raise ValueError("Invalid weighting/sampling/patience/precision/accumulation setting")
    if precision != "fp32" and not str(device).startswith("cuda"):
        raise ValueError("AMP requires an explicit CUDA device")
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    loading_started = time.perf_counter()
    rows, receipt = verify_split(split_dir)
    cache, output = Path(cache), Path(output)
    registry_path = cache / "registration.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    kind = registry.get("evidence_kind")
    if registry.get("schema_version") != 1 or kind not in {
        "frozen_feature_cache_not_an_evaluation",
        "synthetic_software_fixture_only",
    }:
        raise ValueError("Unrecognized feature cache provenance/schema")
    if kind == "frozen_feature_cache_not_an_evaluation":
        digest = registry.get("checkpoint_sha256", "")
        preprocessing = {
            "mean": [104, 117, 128],
            "channel_order": "rgb",
            "aggregation": "32_l2_normalized_c3d_segments",
        }
        if (
            registry.get("feature_dim") != 4096
            or registry.get("num_segments") != 32
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or registry.get("preprocessing") != preprocessing
        ):
            raise ValueError("Incomplete or incompatible C3D feature provenance")
    if registry["split_freeze_hash"] != receipt["freeze_hash"]:
        raise ValueError("Feature cache references a different protected split")
    required = {row.video_id for row in rows if row.split in {"train", "val"}}
    if set(registry["features"]) != required:
        raise ValueError("Cache must cover exactly training and validation clips")
    feature_data = {}
    for key in sorted(required):
        entry = registry["features"][key]
        path = (cache / entry["path"]).resolve()
        if not path.is_relative_to(cache.resolve()) or sha256(path) != entry["sha256"]:
            raise ValueError("Feature cache identity changed or path escaped")
        array = np.load(path, allow_pickle=False)
        if (
            array.shape != (registry["num_segments"], registry["feature_dim"])
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()
        ):
            raise ValueError("Invalid feature shape/type/values")
        feature_data[key] = torch.from_numpy(array).float()
    loading_seconds = time.perf_counter() - loading_started
    training = sorted((r for r in rows if r.split == "train"), key=lambda r: r.video_id)
    validation = sorted((r for r in rows if r.split == "val"), key=lambda r: r.video_id)
    if relabel is not None:
        needed = {r.video_id for r in training + validation}
        if set(relabel) != needed or set(relabel.values()) - {0, 1}:
            raise ValueError("Relabel must give a 0/1 target for exactly every train/val clip")
        training = [replace(r, label=relabel[r.video_id]) for r in training]
        validation = [replace(r, label=relabel[r.video_id]) for r in validation]
    if train_groups is not None:
        known = {r.source_video_id for r in training}
        if not train_groups or set(train_groups) - known:
            raise ValueError("Training subset must name existing training groups")
        training = [r for r in training if r.source_video_id in set(train_groups)]
    targets = [r.label for r in training]
    balanced_weights = binary_class_weights(targets)
    loss_weights = balanced_weights if class_weighting == "balanced" else [1.0, 1.0]
    settings = {
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(device),
        "checkpoint_selection": "validation_pr_auc_then_f1",
        "threshold": 0.5,
        "feature_dim": registry["feature_dim"],
        "cache_registration_sha256": sha256(registry_path),
        "split_freeze_hash": receipt["freeze_hash"],
        "code": _code_identity(),
        "class_weighting": class_weighting,
        "sampling": sampling,
        "patience": patience,
        "precision": precision,
        "accumulation_steps": accumulation_steps,
        "training_class_counts": np.bincount(targets, minlength=2).tolist(),
        "loss_weights": loss_weights,
        "target_name": target_name,
        "relabel_hash": content_hash(relabel) if relabel is not None else None,
        "train_groups": sorted(train_groups) if train_groups is not None else None,
        "training_clip_count": len(training),
    }
    config_hash = content_hash(settings)
    seed_everything(seed)
    model = FrozenBagBinaryHead(registry["feature_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp_fp16")
    start, best, history, elapsed_before = 0, (-1.0, -1.0), [], 0.0
    stale_steps = 0
    if resume:
        if Path(resume).resolve() != (output / "last.pt").resolve():
            raise ValueError("Resume must use this run's last.pt")
        registration = json.loads((output / "registration.json").read_text())
        model, saved = load_binary_checkpoint(resume, device)
        if saved["config_hash"] != config_hash or registration["config_hash"] != config_hash:
            raise ValueError("Resume configuration/data/code changed")
        if steps <= saved["step"]:
            raise ValueError("Resume horizon must extend completed steps")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(saved["optimizer_state"])
        start, best = saved["step"], tuple(saved["best_selection"])
        history, elapsed_before = saved["history"], saved["training_seconds"]
        stale_steps = saved.get("stale_steps", 0)
        scaler.load_state_dict(saved.get("scaler_state", {}))
        if saved.get("early_stopped"):
            raise ValueError("Early-stopped runs require a new experiment, not a horizon extension")
        if (
            torch.load(output / "best.pt", map_location="cpu", weights_only=True)["config_hash"]
            != config_hash
        ):
            raise ValueError("Selected checkpoint does not belong to this run")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_new_json(
            output / "registration.json",
            {
                "config_hash": config_hash,
                "settings": settings,
                "environment": environment_metadata(config_hash, seed, str(device)),
                "feature_provenance": registry,
                "test_used_for_selection": False,
                "evidence_kind": kind,
                "initial_requested_steps": steps,
            },
        )
    val_x = torch.stack([feature_data[r.video_id] for r in validation]).to(device)
    val_y = [r.label for r in validation]
    weights = torch.tensor(loss_weights, dtype=torch.float32, device=device)
    examples_seen = 0
    started = time.perf_counter()
    for step in range(start + 1, steps + 1):
        seed_everything(seed + step)
        generator = torch.Generator().manual_seed(seed + step)
        indices = training_indices(targets, batch_size * accumulation_steps, generator, sampling)
        batch = [training[i] for i in indices]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        # Normalize by total class weight, preserving weighted-CE semantics across microbatches.
        denominator = sum(loss_weights[r.label] for r in batch)
        for offset in range(0, len(batch), batch_size):
            micro = batch[offset : offset + batch_size]
            x = torch.stack([feature_data[r.video_id] for r in micro]).to(device)
            y = torch.tensor([r.label for r in micro], device=device)
            with torch.autocast(
                torch.device(device).type, enabled=precision == "amp_fp16", dtype=torch.float16
            ):
                loss = (
                    F.cross_entropy(model(x).float(), y, weight=weights, reduction="sum")
                    / denominator
                )
            scaler.scale(loss).backward()
            loss_value += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        examples_seen += len(batch)
        model.eval()
        with torch.inference_mode():
            scores = model(val_x).softmax(-1)[:, 1].cpu().tolist()
        metrics = evaluate_binary(val_y, scores)
        selection = (metrics["pr_auc"], metrics["f1"])
        improved = selection > best
        if improved:
            best = selection
        stale_steps = 0 if improved else stale_steps + 1
        early_stopped = patience > 0 and stale_steps >= patience
        history.append({"step": step, "training_loss": loss_value, "validation": metrics})
        saved = {
            "checkpoint_type": "indian_frozen_bag_binary_v1",
            "evidence_kind": kind,
            "classes": ["non_violence", "violence"],
            "feature_dim": registry["feature_dim"],
            "config_hash": config_hash,
            "step": step,
            "requested_horizon": steps,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_selection": list(best),
            "history": history,
            "scaler_state": scaler.state_dict(),
            "stale_steps": stale_steps,
            "early_stopped": early_stopped,
            "validation_predictions": [
                {
                    "clip_id": r.video_id,
                    "source_video_id": r.source_video_id,
                    "label": r.label,
                    "violence_score": score,
                }
                for r, score in zip(validation, scores, strict=True)
            ],
            "validation_source_metrics": {
                source: evaluate_binary(
                    [r.label for r in validation if r.source_video_id == source],
                    [
                        score
                        for r, score in zip(validation, scores, strict=True)
                        if r.source_video_id == source
                    ],
                )
                for source in sorted({r.source_video_id for r in validation})
            },
            "resources": {
                "device": str(device),
                "memory": _device_memory(device),
                "loading_seconds": loading_seconds,
                "feature_examples_loaded": len(required),
                "verified_feature_examples_per_second": len(required) / loading_seconds,
                "loading_scope": "split/media verification and feature hashing/loading",
                "optimizer_examples_this_segment": examples_seen,
            },
            "training_seconds": elapsed_before + time.perf_counter() - started,
        }
        # Atomic replacement is confined to artifacts owned by this new run.
        for name in ["last.pt"] + (["best.pt"] if improved else []):
            temporary = output / (name + ".tmp")
            with temporary.open("xb") as stream:
                torch.save(saved, stream)
            temporary.replace(output / name)
        if early_stopped:
            break
    return output / "best.pt"
