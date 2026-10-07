"""Frozen professor policy with machine-local path relocation only."""

import hashlib
import json
import os
from pathlib import Path

import yaml

from surveillance.inference.provenance import ASSET_KEYS, pipeline_identity


def resolve_path(value: str, root: Path) -> Path:
    return root / Path(os.path.expandvars(value)).expanduser()


def profile_hash(config: dict) -> str:
    payload = json.dumps(pipeline_identity(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_professor_profile(root: Path, local: Path | None = None) -> dict:
    """Relocate bytes without permitting checkpoint/policy edits in a local file."""
    config = yaml.safe_load((root / "configs/demo/professor_demo.yaml").read_text())
    canonical = yaml.safe_load((root / "configs/surveillance_demo_ucf_v1.yaml").read_text())
    if pipeline_identity(config) != pipeline_identity(canonical):
        raise ValueError("Professor scientific policy differs from the frozen UCF deployment")
    if local is not None and not local.is_file():
        raise FileNotFoundError(f"Local asset config missing: {local}")
    local = local or root / "configs/demo/local_assets.yaml"
    if local.is_file():
        overrides = yaml.safe_load(local.read_text()) or {}
        if not isinstance(overrides, dict):
            raise ValueError("Local assets must be a YAML mapping of paths")
        allowed = {*ASSET_KEYS, "examples", "outputs"}
        if set(overrides) - allowed:
            raise ValueError("Local assets may override paths only; model/policy editing is locked")
        for key, value in overrides.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Asset path must be a nonempty string: {key}")
            config[key] = value
    for key in (*ASSET_KEYS, "examples", "outputs"):
        config[key] = str(resolve_path(config[key], root))
    config["professor_locked"] = True
    return config
