"""Draw ROC and precision-recall curves from the saved one-time IITB test scores.

Reads runs/indian_pilot/test_results_v3.json only; it scores nothing and changes no threshold.

    python scripts/pilot_v3_figures.py --output docs/results/pilot_v3_iitb_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Validated categorical slots 1-3; line style is the secondary encoding.
MODELS = [
    ("v3_all_s7", "v3 (deployed)", "#2a78d6", "-", 0.40),
    ("v2_s7", "v2", "#eb6834", "--", 0.45),
    ("zero_shot_sultani", "Zero-shot Sultani", "#1baf7a", ":", 0.50),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=ROOT / "runs/indian_pilot/test_results_v3.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = json.loads(args.results.read_text(encoding="utf-8"))["clip_scores"]
    y = np.array([clip["label"] for clip in scores.values()])

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK})
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    for key, name, color, style, threshold in MODELS:
        s = np.array([clip[key] for clip in scores.values()])
        fpr, tpr, _ = roc_curve(y, s)
        roc_ax.plot(
            fpr, tpr, style, color=color, lw=2, label=f"{name}  AUC {roc_auc_score(y, s):.2f}"
        )
        precision, recall, _ = precision_recall_curve(y, s)
        pr_ax.plot(
            recall,
            precision,
            style,
            color=color,
            lw=2,
            drawstyle="steps-post",
            label=f"{name}  AP {average_precision_score(y, s):.2f}",
        )
        if key == "v3_all_s7":
            flagged = s >= threshold
            tp_rate = flagged[y == 1].mean()
            fp_rate = flagged[y == 0].mean()
            prec = flagged[y == 1].sum() / flagged.sum()
            roc_ax.plot(fp_rate, tp_rate, "o", ms=8, color=color, mec=SURFACE, mew=2)
            roc_ax.annotate(
                f"threshold {threshold:.2f}\n17/24 fights, 2/25 false alarms",
                (fp_rate, tp_rate),
                xytext=(0.22, 0.42),
                color=INK,
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1},
            )
            pr_ax.plot(tp_rate, prec, "o", ms=8, color=color, mec=SURFACE, mew=2)
    roc_ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=0)
    pr_ax.axhline(y.mean(), color=GRID, lw=1, zorder=0)
    pr_ax.text(0.02, y.mean() - 0.06, f"chance ({y.mean():.2f})", color=MUTED, fontsize=9)
    for ax, title, xl, yl in (
        (roc_ax, "ROC curve", "False positive rate", "True positive rate (recall)"),
        (pr_ax, "Precision-recall curve", "Recall", "Precision"),
    ):
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.04), xlabel=xl, ylabel=yl, facecolor=SURFACE)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "One-time IITB-Corridor test: 49 clips (24 fights, 25 non-violent), seed 7",
        x=0.01,
        ha="left",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
