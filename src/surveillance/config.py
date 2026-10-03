"""Small YAML config loader; no global or machine-specific configuration."""

from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    """Load experiment YAML and require the core architecture fields."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (
        not isinstance(config, dict)
        or not {"num_segments", "feature_dim", "model", "training"} <= config.keys()
    ):
        raise ValueError("Config requires num_segments, feature_dim, model, training")
    if min(config["num_segments"], config["feature_dim"]) < 1:
        raise ValueError("num_segments and feature_dim must be positive")
    return config
