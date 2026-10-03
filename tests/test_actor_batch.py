import pytest
import torch

from surveillance.datasets.actor_batch import collate_actors


def sample(n):
    return {
        "actor_boxes": torch.tensor([[0, 0, 0.5, 0.5]] * n).reshape(n, 4).float(),
        "actor_labels": torch.arange(n),
        "group_label": 2,
        "pose_features": torch.ones(n, 4),
        "metadata": {"clip_id": str(n)},
    }


def test_variable_actor_padding_masks_labels():
    batch = collate_actors([sample(1), sample(3)])
    assert batch["actor_boxes"].shape == (2, 3, 4)
    assert batch["actor_valid_mask"].tolist() == [[True, False, False], [True] * 3]
    assert batch["actor_labels"].tolist() == [[0, -100, -100], [0, 1, 2]]
    assert batch["group_labels"].tolist() == [2, 2]
    assert batch["pose_features"][0, 1:].count_nonzero() == 0


def test_reject_empty_and_mismatched_modality():
    with pytest.raises(ValueError):
        collate_actors([])
    with pytest.raises(ValueError):
        collate_actors([sample(0), sample(2)])
    a, b = sample(1), sample(2)
    del b["pose_features"]
    with pytest.raises(ValueError, match="modalit"):
        collate_actors([a, b])


def test_invalid_features_and_raw_stacking():
    a = sample(1)
    a["pose_features"] = torch.ones(2, 4)
    with pytest.raises(ValueError):
        collate_actors([a])
    a = sample(1)
    a["frames"] = torch.zeros(10, 3, 4, 6)
    assert collate_actors([a])["frames"].shape == (1, 10, 3, 4, 6)


@pytest.mark.parametrize("bad", ["boxes", "labels", "nan", "dimensions", "frames"])
def test_reject_invalid_batch_inputs(bad):
    a, b = sample(1), sample(2)
    if bad == "boxes":
        b["actor_boxes"][0, 2] = -1
    elif bad == "labels":
        b["actor_labels"] = torch.tensor([0])
    elif bad == "nan":
        b["pose_features"][0, 0] = float("nan")
    elif bad == "dimensions":
        b["pose_features"] = torch.zeros(2, 5)
    else:
        a["frames"] = torch.zeros(10, 3, 4, 6)
        b["frames"] = torch.zeros(10, 3, 8, 6)
    with pytest.raises(ValueError):
        collate_actors([a, b])
