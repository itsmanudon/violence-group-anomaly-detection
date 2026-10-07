import pytest
import torch

from surveillance.features.c3d_conversion import convert_openmmlab_state


def upstream_state():
    # Expanded tensors provide real shape/dtype values without allocating 300 MB.
    from surveillance.features.c3d import C3DFC6

    with torch.device("meta"):
        shapes = C3DFC6().state_dict()
    state = {}
    for key, value in shapes.items():
        layer, parameter = key.split(".")
        if layer.startswith("conv"):
            layer = {"conv1": "conv1a", "conv2": "conv2a"}.get(layer, layer) + ".conv"
        state[f"{layer}.{parameter}"] = torch.full((1,), 0.125).expand(value.shape)
    state["fc7.weight"] = torch.zeros(1).expand(4096, 4096)
    state["fc7.bias"] = torch.zeros(1).expand(4096)
    return state


def test_conversion_preserves_fc6_and_every_convolution_without_fc7():
    state = upstream_state()
    converted = convert_openmmlab_state(state)
    assert converted["conv1.weight"] is state["conv1a.conv.weight"]
    assert converted["conv2.bias"] is state["conv2a.conv.bias"]
    assert converted["fc6.weight"] is state["fc6.weight"]
    assert len(converted) == 18
    assert "fc7.weight" not in converted


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nonfinite"])
def test_conversion_rejects_incomplete_or_corrupt_parameters(corruption):
    state = upstream_state()
    if corruption == "missing":
        del state["conv1a.conv.weight"]
    elif corruption == "extra":
        state["fc8.weight"] = torch.zeros(1)
    elif corruption == "shape":
        state["conv1a.conv.bias"] = torch.zeros(1)
    elif corruption == "dtype":
        state["conv1a.conv.bias"] = torch.zeros(64, dtype=torch.int64)
    else:
        state["conv1a.conv.bias"] = torch.full((64,), float("nan"))
    with pytest.raises(ValueError):
        convert_openmmlab_state(state)


@pytest.mark.parametrize("mean,order", [((0, 0, 0), "rgb"), ((104, 117, 128), "bgr")])
def test_exported_checkpoint_rejects_wrong_preprocessing(tmp_path, mean, order):
    from surveillance.features.c3d import C3DExtractor

    path = tmp_path / "export.pt"
    torch.save(
        {
            "state_dict": convert_openmmlab_state(upstream_state()),
            "metadata": {"preprocessing": {"mean": [104, 117, 128], "channel_order": "rgb"}},
        },
        path,
    )
    with pytest.raises(ValueError, match="preprocessing"):
        C3DExtractor(path, mean=mean, channel_order=order)
