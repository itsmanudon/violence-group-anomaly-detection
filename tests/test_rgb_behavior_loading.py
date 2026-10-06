from types import SimpleNamespace

import pytest

from surveillance.datasets.dcsass_audit import sha256
from surveillance.inference import rgb_behavior


@pytest.mark.parametrize("changed", ["detector", "i3d"])
def test_live_behavior_rejects_changed_training_backbone_before_constructing_it(
    tmp_path, monkeypatch, changed
):
    assets = {name: tmp_path / (name + ".pt") for name in ["detector", "i3d"]}
    for path in assets.values():
        path.write_bytes(b"original compatible model")
    saved = {
        "dry_run": False,
        "config": {name: {"checkpoint_sha256": sha256(path)} for name, path in assets.items()},
    }
    assets[changed].write_bytes(b"changed compatible model")
    monkeypatch.setattr(
        rgb_behavior, "load_behavior_checkpoint", lambda *args: (SimpleNamespace(), saved)
    )
    monkeypatch.setattr(
        rgb_behavior,
        "TorchvisionPersonDetector",
        lambda *args: pytest.fail("Changed backbone must be rejected before construction"),
    )
    with pytest.raises(ValueError, match="backbone"):
        rgb_behavior.RGBBehaviorAnalyzer(
            tmp_path / "behavior.pt", assets["detector"], assets["i3d"], "cpu"
        )
