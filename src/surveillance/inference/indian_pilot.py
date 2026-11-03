"""Optional Indian pilot binary score on the cascade's own C3D bag.

Pilot evidence only: a separately trained head whose training data and test results are
recorded in runs/indian_pilot (v3: Indian, DCSASS, UBI-Fights and XD-Violence clips). It
never changes the cascade alert and is shown separately for human review.
"""

import json
from pathlib import Path

import numpy as np
import torch

from surveillance.datasets.dcsass_audit import sha256
from surveillance.training.indian_binary import load_binary_checkpoint

SCOPE = (
    "Pilot score from a separately trained violence head, outside the cascade's validated "
    "alert. For human review only; it never decides anything."
)
SPEC_KEYS = {"checkpoint", "checkpoint_sha256", "threshold", "split_freeze_hash", "threshold_rule"}


def load_pilot_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(spec) - SPEC_KEYS - {"notes"} or SPEC_KEYS - set(spec):
        raise ValueError(f"Indian pilot spec needs exactly {sorted(SPEC_KEYS)} (+ notes)")
    if not 0 < float(spec["threshold"]) < 1:
        raise ValueError("Indian pilot threshold must be in (0, 1)")
    return spec


class IndianPilotHead:
    def __init__(self, spec: dict, root: Path):
        path = Path(root) / spec["checkpoint"]
        if sha256(path) != spec["checkpoint_sha256"]:
            raise ValueError("Indian pilot checkpoint differs from its pinned SHA256")
        self.model, _ = load_binary_checkpoint(path, "cpu")
        self.spec = spec

    @torch.inference_mode()
    def score(self, features) -> dict:
        bag = torch.as_tensor(np.asarray(features), dtype=torch.float32)[None]
        probability = float(self.model(bag).softmax(-1)[0, 1])
        threshold = float(self.spec["threshold"])
        return {
            "violence_score": probability,
            "threshold": threshold,
            "flagged": probability >= threshold,
            "threshold_rule": self.spec["threshold_rule"],
            "model_sha256": self.spec["checkpoint_sha256"],
            "split_freeze_hash": self.spec["split_freeze_hash"],
            "scope": SCOPE,
        }
