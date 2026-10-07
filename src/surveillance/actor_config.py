"""Actor-Transformer YAML defaults and validation, independent of MIL configs."""

import copy
import math
from pathlib import Path

import yaml

from surveillance.models.actor_transformer.actor_transformer import MODES

DEFAULT_CONFIG = {
    "seed": 42,
    "device": "auto",
    "data": {"num_frames": 10, "image_size": [480, 720], "input_mode": "precomputed"},
    "model": {
        "mode": "pose_only",
        "pose_feature_dim": 98304,
        "rgb_feature_dim": 20800,
        "embedding_dim": 128,
        "num_actor_classes": 5,
        "num_group_classes": 5,
        "late_fusion_pose_weight": 2.0,
        "transformer": {
            "num_layers": 1,
            "num_heads": 1,
            "feedforward_dim": 256,
            "dropout": 0.1,
            "positional_encoding": True,
            "reference_size": [480, 720],
        },
    },
    "backbones": {
        "pose": {"checkpoint": None, "frozen": True},
        "rgb": {"checkpoint": None, "frozen": True},
    },
    "loss": {"group_weight": 1.0, "actor_weight": 1.0},
    "training": {
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "max_iterations": 20000,
        "lr_milestones": [5000, 10000],
        "lr_gamma": 0.1,
        "batch_size": 16,
        "gradient_clip": 1.0,
        "validation_interval": 100,
        "checkpoint_interval": 100,
    },
    "actor_classes": ["crossing", "waiting", "queueing", "walking", "talking"],
    "group_classes": ["crossing", "waiting", "queueing", "walking", "talking"],
}


def _merge(target: dict, update: dict, prefix: str = "") -> None:
    for key, value in update.items():
        if key not in target:
            raise ValueError(f"Unknown Actor-Transformer config key: {prefix}{key}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be a mapping")
            _merge(target[key], value, f"{prefix}{key}.")
        else:
            target[key] = value


def _positive(value, name: str, integer: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'}")


def validate_actor_config(config: dict) -> dict:
    """Merge explicit defaults; reject unknown keys and incompatible tensor settings."""
    if not isinstance(config, dict):
        raise ValueError("Actor config must be a YAML mapping")
    merged = copy.deepcopy(DEFAULT_CONFIG)
    _merge(merged, config)
    data, model, training = merged["data"], merged["model"], merged["training"]
    if not isinstance(merged["seed"], int) or merged["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(merged["device"], str):
        raise ValueError("device must be auto, cpu, or a torch device string")
    if data["num_frames"] != 10:
        raise ValueError("This baseline uses num_frames=10 (middle index 5)")
    if data["input_mode"] not in {"precomputed", "raw"}:
        raise ValueError("data.input_mode must be precomputed or raw")
    if not isinstance(data["image_size"], (list, tuple)) or len(data["image_size"]) != 2:
        raise ValueError("data.image_size must be [height,width]")
    for value in data["image_size"]:
        _positive(value, "data.image_size", True)
    if model["mode"] not in MODES:
        raise ValueError(f"model.mode must be one of {sorted(MODES)}")
    for name in (
        "pose_feature_dim",
        "rgb_feature_dim",
        "embedding_dim",
        "num_actor_classes",
        "num_group_classes",
    ):
        _positive(model[name], f"model.{name}", True)
    transformer = model["transformer"]
    for name in ("num_layers", "num_heads", "feedforward_dim"):
        _positive(transformer[name], f"transformer.{name}", True)
    if model["embedding_dim"] % transformer["num_heads"]:
        raise ValueError("embedding_dim must be divisible by num_heads")
    if transformer["positional_encoding"] and model["embedding_dim"] % 4:
        raise ValueError("2D positional encoding requires embedding_dim divisible by four")
    if not isinstance(transformer["positional_encoding"], bool):
        raise ValueError("positional_encoding must be Boolean")
    if not isinstance(transformer["dropout"], (int, float)) or not 0 <= transformer["dropout"] < 1:
        raise ValueError("transformer.dropout must be in [0,1)")
    if (
        not isinstance(transformer["reference_size"], (list, tuple))
        or len(transformer["reference_size"]) != 2
    ):
        raise ValueError("reference_size must be [height,width]")
    for value in transformer["reference_size"]:
        _positive(value, "reference_size")
    _positive(model["late_fusion_pose_weight"], "late_fusion_pose_weight")
    for name in ("group_weight", "actor_weight"):
        value = merged["loss"][name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{name} must be nonnegative and finite")
    if training["optimizer"] not in {"adam", "adamw", "sgd"}:
        raise ValueError("training.optimizer must be adam, adamw, or sgd")
    for name in ("learning_rate", "eps", "lr_gamma"):
        _positive(training[name], f"training.{name}")
    for name in ("max_iterations", "batch_size", "validation_interval", "checkpoint_interval"):
        _positive(training[name], f"training.{name}", True)
    if training["gradient_clip"] is not None:
        _positive(training["gradient_clip"], "training.gradient_clip")
    if (
        not isinstance(training["betas"], (list, tuple))
        or len(training["betas"]) != 2
        or any(not isinstance(b, (int, float)) or not 0 <= b < 1 for b in training["betas"])
    ):
        raise ValueError("training.betas must be two values in [0,1)")
    milestones = training["lr_milestones"]
    if not isinstance(milestones, list):
        raise ValueError("lr_milestones must be a sorted list of unique positive iterations")
    for iteration in milestones:
        _positive(iteration, "lr_milestones", True)
    if milestones != sorted(set(milestones)):
        raise ValueError("lr_milestones must be sorted and unique")
    for modality in ("pose", "rgb"):
        backbone = merged["backbones"][modality]
        if not isinstance(backbone["frozen"], bool):
            raise ValueError("backbones frozen flags must be Boolean")
        if backbone["checkpoint"] is not None and not isinstance(backbone["checkpoint"], str):
            raise ValueError("backbone checkpoint must be a local path string or null")
    for key, count_key in (
        ("actor_classes", "num_actor_classes"),
        ("group_classes", "num_group_classes"),
    ):
        labels = merged[key]
        if (
            not isinstance(labels, list)
            or len(labels) != model[count_key]
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(f"{key} must list {model[count_key]} unique nonempty class names")
    return merged


def load_actor_config(path: Path) -> dict:
    """Read actor YAML; local backbone paths resolve relative to the YAML directory."""
    path = Path(path)
    config = validate_actor_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    for settings in config["backbones"].values():
        if settings["checkpoint"] is not None:
            checkpoint = Path(settings["checkpoint"])
            if not checkpoint.is_absolute():
                settings["checkpoint"] = str((path.resolve().parent / checkpoint).resolve())
    return config
