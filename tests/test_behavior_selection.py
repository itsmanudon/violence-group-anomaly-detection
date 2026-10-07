import json

import pytest
import yaml

from surveillance.datasets.dcsass_audit import sha256
from surveillance.experiments.behavior_selection import select_initialization_control


def make_run(path, initialization, macro_f1, seed=0):
    path.mkdir()
    (path / "best.pt").write_bytes(initialization.encode())
    config = {"initialization": initialization, "seed": seed}
    (path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (path / "selection.json").write_text(
        json.dumps(
            {
                "selection_frozen": True,
                "dry_run": False,
                "checkpoint_sha256": sha256(path / "best.pt"),
                "selected_iteration": 100,
                "validation": {"macro_f1": macro_f1, "accuracy": 0.5},
            }
        )
    )


def test_control_selection_uses_validation_only_without_opening_heldout(tmp_path):
    transfer, random = tmp_path / "transfer", tmp_path / "random"
    make_run(transfer, "collective_transfer", 0.3)
    make_run(random, "random", 0.4)
    (transfer / "held_out").mkdir()
    (transfer / "held_out/metrics.json").write_text("Must never read this")
    result = select_initialization_control(transfer, random)
    assert result["selected_run"] == str(random)
    assert result["selection_metric"] == "validation_macro_f1"
    assert result["held_out_used_for_selection"] is False


def test_control_refuses_other_optimization_changes(tmp_path):
    transfer, random = tmp_path / "transfer", tmp_path / "random"
    make_run(transfer, "collective_transfer", 0.3)
    make_run(random, "random", 0.4, seed=1)
    with pytest.raises(ValueError, match="initialization"):
        select_initialization_control(transfer, random)
