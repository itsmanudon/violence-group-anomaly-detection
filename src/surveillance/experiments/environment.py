"""Small, allowlisted reproducibility metadata without machine/user identifiers."""

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def _version(package: str) -> str | None:
    try:
        return str(version(package))
    except PackageNotFoundError:
        return None


def environment_metadata(config_hash: str, seed: int, device: str) -> dict:
    """Record software, selected device and commit; never collect the environment."""
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    available = bool(torch.cuda.is_available())
    cuda_device = None
    if available and str(device).startswith("cuda"):
        selected = torch.device(device)
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        cuda_device = torch.cuda.get_device_name(index)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": _version("torchvision"),
        "cuda_available": available,
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_device": cuda_device,
        "device": str(device),
        "os": {"system": platform.system(), "release": platform.release()},
        "git_commit": commit,
        "config_hash": config_hash,
        "seed": seed,
    }
