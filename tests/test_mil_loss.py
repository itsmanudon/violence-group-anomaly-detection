import pytest
import torch

from surveillance.models.sultani.loss import MILLoss


def test_ranking_math():
    loss = MILLoss(sparsity_weight=0, smoothness_weight=0)
    assert loss(torch.ones(1, 32), torch.zeros(1, 32))["ranking"] == 0
    assert loss(torch.zeros(1, 32), torch.ones(1, 32))["ranking"] == 2
    assert loss(torch.tensor([[0.8, 0.2]]), torch.tensor([[0.1, 0.3]]))[
        "ranking"
    ].item() == pytest.approx(0.5)


def test_regularizers_and_backward():
    loss = MILLoss()
    low = loss(torch.zeros(1, 4), torch.zeros(1, 4))
    high = loss(torch.ones(1, 4), torch.zeros(1, 4))
    oscillating = loss(torch.tensor([[0.0, 1.0, 0.0, 1.0]], requires_grad=True), torch.zeros(1, 4))
    assert high["sparsity"] > low["sparsity"]
    assert oscillating["smoothness"] == 3
    oscillating["total"].backward()
    single = loss(torch.ones(1, 1), torch.zeros(1, 1))
    assert single["smoothness"] == 0
    assert torch.isfinite(single["total"])
    with pytest.raises(ValueError):
        loss(torch.empty(0, 32), torch.empty(0, 32))
