import pytest
import torch

from surveillance.models.sultani.model import SultaniScorer


def test_shape_range_gradients():
    model = SultaniScorer(feature_dim=8, hidden_dims=[6, 3], dropout=0)
    x = torch.randn(2, 32, 8, requires_grad=True)
    scores = model(x)
    assert scores.shape == (2, 32)
    assert ((scores >= 0) & (scores <= 1)).all()
    scores.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        model(torch.randn(32, 7))
