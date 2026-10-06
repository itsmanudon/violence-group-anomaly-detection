"""Local-only loading for the two-paper cascade; missing assets fail explicitly."""

import os
from pathlib import Path

from surveillance.datasets.dcsass_audit import sha256
from surveillance.features.c3d import C3DExtractor
from surveillance.inference.anomaly_pipeline import AnomalyPipeline
from surveillance.inference.provenance import (
    ASSET_KEYS,
    pipeline_identity,
    verify_selected_checkpoint,
)
from surveillance.inference.rgb_behavior import RGBBehaviorAnalyzer
from surveillance.inference.surveillance_pipeline import SurveillancePipeline
from surveillance.training.sultani_trainer import load_checkpoint, seed_everything, select_device


def asset_status(config: dict, root: Path) -> dict[str, bool]:
    return {key: (Path(root) / config[key]).is_file() for key in ASSET_KEYS}


def load_surveillance_pipeline(config: dict, root: Path) -> SurveillancePipeline:
    root = Path(root)
    missing = [key for key, installed in asset_status(config, root).items() if not installed]
    if missing:
        raise FileNotFoundError(
            "Live inference requires local trained assets: "
            + ", ".join(missing)
            + ". See docs/end-to-end-required-assets.md and the execution journal."
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(0)
    device = select_device(config.get("device", "auto"))
    scorer, saved = load_checkpoint(root / config["sultani_checkpoint"], device)
    if saved.get("dry_run") or saved.get("selection_metric") == "preflight_only":
        raise ValueError("A preflight checkpoint cannot be used as a trained Sultani model")
    provenance = pipeline_identity(config)
    for key in ASSET_KEYS:
        if sha256(root / config[key]) != provenance["models"][key]:
            raise ValueError(f"Live model differs from frozen demo identity: {key}")
    for key in ("sultani_checkpoint", "behavior_checkpoint"):
        verify_selected_checkpoint(root / config[key])
    if scorer.feature_dim != 4096 or saved["config"]["num_segments"] != 32:
        raise ValueError("Cascade requires the approved 32-segment C3D/Sultani contract")
    extractor = C3DExtractor(
        root / config["c3d_checkpoint"],
        str(device),
        batch_size=config.get("c3d_batch_size", 4),
        mean=config["c3d_mean"],
        channel_order=config["c3d_channel_order"],
    )
    anomaly = AnomalyPipeline(scorer, extractor, 32)
    behavior = RGBBehaviorAnalyzer(
        root / config["behavior_checkpoint"],
        root / config["detector_checkpoint"],
        root / config["i3d_checkpoint"],
        str(device),
    )
    return SurveillancePipeline(
        anomaly,
        behavior,
        threshold=config["anomaly_threshold"],
        max_windows=config["max_behavior_windows"],
        provenance=provenance,
    )
