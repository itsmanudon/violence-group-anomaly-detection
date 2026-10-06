from types import SimpleNamespace

import pytest

from surveillance.inference.loading import asset_status, load_surveillance_pipeline


def test_missing_models_fail_explicitly_without_random_weight_fallback(tmp_path):
    config = {
        key: key + ".pt"
        for key in [
            "sultani_checkpoint",
            "c3d_checkpoint",
            "behavior_checkpoint",
            "detector_checkpoint",
            "i3d_checkpoint",
        ]
    }
    assert not any(asset_status(config, tmp_path).values())
    with pytest.raises(FileNotFoundError, match="sultani_checkpoint"):
        load_surveillance_pipeline(config, tmp_path)


@pytest.mark.parametrize("marker", [{"dry_run": True}, {"selection_metric": "preflight_only"}])
def test_live_loading_rejects_sultani_preflight_before_loading_backbones(
    tmp_path, monkeypatch, marker
):
    from surveillance.inference import loading

    config = {key: key + ".pt" for key in loading.ASSET_KEYS}
    for value in config.values():
        (tmp_path / value).touch()
    saved = {"config": {"num_segments": 32}, **marker}
    monkeypatch.setattr(
        loading, "load_checkpoint", lambda *args: (SimpleNamespace(feature_dim=4096), saved)
    )
    with pytest.raises(ValueError, match="preflight"):
        load_surveillance_pipeline(config, tmp_path)
