"""Lightweight file-based anomaly timeline plots."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_timeline(
    timestamps: Sequence[tuple[float, float]],
    scores: Sequence[float],
    output: Path,
    threshold: float = 0.5,
) -> Path:
    """Save a temporal anomaly step plot without requiring an interactive GUI."""
    if not timestamps or len(timestamps) != len(scores):
        raise ValueError("timestamps and scores must be nonempty and match in length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 3))
    edges = [timestamps[0][0], *[end for _, end in timestamps]]
    axis.stairs(scores, edges, linewidth=1.5, color="tab:red")
    axis.axhline(threshold, linestyle="--", color="gray", label=f"threshold {threshold:g}")
    axis.set(
        xlabel="Time (seconds)",
        ylabel="Anomaly score",
        ylim=(0, 1),
        title="Behavioral anomaly timeline",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
