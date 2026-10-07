import numpy as np
import pytest

from surveillance.visualization.actor_attention import save_actor_attention


def test_attention_plot_uses_real_square_matrix(tmp_path):
    path = tmp_path / "attention.png"
    save_actor_attention(np.array([[0.2, 0.8], [0.6, 0.4]]), path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="square"):
        save_actor_attention(np.ones((2, 3)), path)
