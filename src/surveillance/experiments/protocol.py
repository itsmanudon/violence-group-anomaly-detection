"""Resolved Collective protocols and immutable, content-addressed freeze receipts."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

from surveillance.actor_config import validate_actor_config
from surveillance.datasets.collective import CLASSES, TEST_SEQUENCES, TRAIN_SEQUENCES
from surveillance.detection.config import DetectionConfig


def content_hash(value: dict | list) -> str:
    """Hash canonical JSON, rejecting nonfinite protocol values."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_protocol(path: Path) -> dict:
    """Resolve one YAML protocol; unknown settings and test tuning splits are rejected."""
    path = Path(path).resolve()
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "evidence_kind",
        "seeds",
        "dataset",
        "output_root",
        "actor",
        "features",
        "detector",
        "experiments",
        "metrics",
    }
    if not isinstance(protocol, dict) or set(protocol) != required:
        raise ValueError(f"Protocol must contain exactly {sorted(required)}")
    if protocol["schema_version"] != 1 or protocol["evidence_kind"] not in {"real", "synthetic"}:
        raise ValueError("Protocol requires schema_version=1 and real/synthetic evidence_kind")
    if not isinstance(protocol["protocol_id"], str) or not protocol["protocol_id"]:
        raise ValueError("protocol_id must be nonempty")
    seeds = protocol["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    dataset = protocol["dataset"]
    dataset_required = {
        "root",
        "manifest",
        "validation_sequences",
        "train_sequences",
        "test_sequences",
    }
    if not dataset_required <= set(dataset) or set(dataset) - (
        dataset_required | {"annotation_box_policy"}
    ):
        raise ValueError("Invalid dataset protocol keys")
    if dataset.get("annotation_box_policy", "strict") not in {"strict", "clip_to_image"}:
        raise ValueError("annotation_box_policy must be strict or clip_to_image")
    if tuple(sorted(dataset["train_sequences"])) != tuple(sorted(TRAIN_SEQUENCES)) or (
        tuple(sorted(dataset["test_sequences"])) != tuple(sorted(TEST_SEQUENCES))
    ):
        raise ValueError("Protocol must preserve the documented Collective 32/12 split IDs")
    validation = dataset["validation_sequences"]
    if (
        not isinstance(validation, list)
        or not validation
        or len(set(validation)) != len(validation)
        or not set(validation) < set(TRAIN_SEQUENCES)
    ):
        raise ValueError(
            "Validation must hold out unique training-only sequences and leave training data"
        )
    actor = validate_actor_config(protocol["actor"])
    if (
        actor["actor_classes"] != list(CLASSES)
        or actor["group_classes"] != list(CLASSES)
        or actor["data"]["input_mode"] != "precomputed"
    ):
        raise ValueError(
            "Benchmark protocol requires Collective vocabulary and cached feature mode"
        )
    protocol["actor"] = actor
    if protocol["evidence_kind"] == "real" and (
        actor["model"]["pose_feature_dim"] != 98304 or actor["model"]["rgb_feature_dim"] != 20800
    ):
        raise ValueError("Real benchmark requires documented HRNet/I3D feature dimensions")
    features = protocol["features"]
    if (
        set(features) != {"normalization", "pose", "rgb"}
        or features["normalization"] != "backbone_only"
    ):
        raise ValueError("Features require pose/rgb and explicit backbone_only normalization")
    for name, architecture, endpoint in (
        ("pose", "pose_hrnet_w32", "pre_final_layer"),
        ("rgb", "i3d", "Mixed_4f"),
    ):
        settings = features[name]
        if set(settings) != {"architecture", "endpoint", "checkpoint", "checkpoint_sha256"}:
            raise ValueError(f"Invalid {name} feature settings")
        if settings["architecture"] != architecture or settings["endpoint"] != endpoint:
            raise ValueError(f"{name} must use the Milestone 2A feature endpoint")
    detector = protocol["detector"]
    if set(detector) != {
        "backend",
        "checkpoint",
        "checkpoint_sha256",
        "candidates",
        "selection",
        "detections",
        "filters",
        "matching_iou",
        "confidence_values",
        "min_recall",
    }:
        raise ValueError("Invalid detector protocol keys")
    if detector["backend"] != "torchvision_fasterrcnn":
        raise ValueError("Unsupported protocol detector backend")
    DetectionConfig(**detector["filters"])
    if not 0 < detector["matching_iou"] <= 1 or not 0 <= detector["min_recall"] <= 1:
        raise ValueError("Invalid fixed matching IoU or minimum validation recall")
    values = detector["confidence_values"]
    if not isinstance(values, list) or not values or any(not 0 <= v <= 1 for v in values):
        raise ValueError("confidence_values must be a nonempty probability list")
    expected_metrics = {
        "checkpoint_selection": "validation_group_accuracy",
        "macro_f1": "all_configured_classes",
        "detected_actors": "matched_only_with_coverage",
        "empty_scenes": "abstain_and_count_in_all_scene_group_accuracy",
        "std": "sample_ddof_1",
    }
    if protocol["metrics"] != expected_metrics:
        raise ValueError("Metric populations/reductions must use the documented definitions")
    experiments = protocol["experiments"]
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError("At least one experiment is required")
    for name, experiment in experiments.items():
        if (
            not isinstance(name, str)
            or not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
        ):
            raise ValueError("Experiment names must be safe lowercase directory names")
        allowed = {"mode", "box_source", "manifest", "checkpoint_experiment", "detections"}
        if set(experiment) - allowed or not {"mode", "box_source", "manifest"} <= set(experiment):
            raise ValueError(f"Invalid experiment {name}")
        validate_actor_config({**actor, "model": {**actor["model"], "mode": experiment["mode"]}})
        if experiment["box_source"] not in {"ground_truth", "detections"}:
            raise ValueError("box_source must be ground_truth or detections")
        if experiment["box_source"] == "detections":
            reference = experiments.get(experiment.get("checkpoint_experiment"), {})
            if (
                reference.get("box_source") != "ground_truth"
                or reference.get("mode") != experiment["mode"]
                or reference.get("manifest") != experiment["manifest"]
                or "detections" not in experiment
            ):
                raise ValueError(
                    "Detected experiment must reuse the matching GT mode/manifest/checkpoint"
                )

    def resolve(value):
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError("Protocol paths must be nonempty strings or null")
        return str((path.parent / value).resolve())

    for key in ("root", "manifest"):
        dataset[key] = resolve(dataset[key])
    protocol["output_root"] = resolve(protocol["output_root"])
    for settings in (features["pose"], features["rgb"], detector):
        settings["checkpoint"] = resolve(settings["checkpoint"])
        digest = settings["checkpoint_sha256"]
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA256 or null")
    for key in ("candidates", "selection", "detections"):
        detector[key] = resolve(detector[key])
    for experiment in experiments.values():
        for key in ("manifest", "detections"):
            if key in experiment:
                experiment[key] = resolve(experiment[key])
    return protocol


def actor_config(protocol: dict, experiment: str, seed: int) -> dict:
    """Resolve the full existing trainer config for exactly one declared seed/mode."""
    if seed not in protocol["seeds"] or experiment not in protocol["experiments"]:
        raise ValueError("Experiment and seed must be declared by the protocol")
    config = copy.deepcopy(protocol["actor"])
    config["seed"] = seed
    config["model"]["mode"] = protocol["experiments"][experiment]["mode"]
    return validate_actor_config(config)


def feature_expectations(protocol: dict, mode: str) -> dict:
    """Bind feature checkpoint bytes or an explicitly supplied imported-cache fingerprint."""
    expected = {}
    for name in ["pose", "rgb"] if "fusion" in mode else [mode.split("_")[0]]:
        settings = protocol["features"][name]
        digest = settings["checkpoint_sha256"]
        if settings["checkpoint"] is not None:
            actual = file_hash(Path(settings["checkpoint"]))
            if digest is not None and actual != digest:
                raise ValueError(f"{name} checkpoint fingerprint differs from protocol")
            digest = actual
        if digest is None:
            raise ValueError(f"Configure a local {name} export or its verified checkpoint_sha256")
        expected[name] = {
            "checkpoint_sha256": digest,
            "architecture": settings["architecture"],
            "endpoint": settings["endpoint"],
        }
    return expected


def freeze_receipt(
    protocol: dict, experiment: str, artifacts: dict, selection: dict | None, output: Path
) -> dict:
    """Create an immutable receipt; changed settings/artifacts require a new receipt path."""
    if experiment not in protocol["experiments"]:
        raise ValueError("Unknown experiment")
    payload = {
        "schema_version": 1,
        "protocol_hash": content_hash(protocol),
        "experiment": experiment,
        "evidence_kind": protocol["evidence_kind"],
        "artifacts": artifacts,
        "selection": selection,
        "resolved_protocol": protocol,
    }
    receipt = {**payload, "freeze_hash": content_hash(payload)}
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != receipt:
            raise ValueError(
                "Freeze receipt already exists with different content; use a new protocol version"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    return receipt


def verify_receipt(protocol: dict, experiment: str, path: Path) -> dict:
    """Reject altered receipts, changed protocol settings and stale artifact bytes."""
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {key: value for key, value in receipt.items() if key != "freeze_hash"}
    if (
        receipt.get("freeze_hash") != content_hash(payload)
        or receipt.get("protocol_hash") != content_hash(protocol)
        or receipt.get("experiment") != experiment
    ):
        raise ValueError("Protocol freeze hash/experiment mismatch")
    for path_string, digest in receipt["artifacts"].items():
        if file_hash(Path(path_string)) != digest:
            raise ValueError(f"Frozen artifact changed: {path_string}")
    return receipt
