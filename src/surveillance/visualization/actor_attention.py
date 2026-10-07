"""Save actor-to-actor attention matrices without a demo UI."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_actor_attention(
    attention: np.ndarray, output: Path, actor_names: list[str] | None = None
) -> Path:
    """Plot one unpadded [N,N] matrix, rows=queries and columns=attended actors."""
    values = np.asarray(attention, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[0] != values.shape[1]:
        raise ValueError("Attention must be a nonempty square [N,N] matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Attention must be finite and nonnegative")
    names = actor_names or [f"Actor {i + 1}" for i in range(len(values))]
    if len(names) != len(values):
        raise ValueError("actor_names must match attention dimension")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 5))
    picture = axis.imshow(values, vmin=0, vmax=1, cmap="Blues")
    axis.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Attended actor (key)",
        ylabel="Query actor",
        title="Actor self-attention",
    )
    axis.tick_params(axis="x", rotation=45)
    fig.colorbar(picture, ax=axis)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
