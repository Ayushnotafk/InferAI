"""Publication-quality figure generation for InferAI research reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Consistent publication style (avoid AI-default purple themes).
_PALETTE = {
    "line": "#0F766E",
    "bar": "#115E59",
    "accent": "#B45309",
    "grid": "#94A3B8",
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }
)


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metric_vs_alpha_pub(
    df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    *,
    ylabel: str | None = None,
) -> Path:
    """Publication line plot of metric vs fusion alpha."""
    out = Path(out_path)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(
        df["alpha"],
        df[metric],
        marker="o",
        linewidth=2.2,
        markersize=7,
        color=_PALETTE["line"],
    )
    # Mark empirical optimum for accuracy / F1.
    if metric in {"accuracy", "macro_f1"} and len(df):
        i = int(df[metric].idxmax())
        ax.scatter(
            [df.loc[i, "alpha"]],
            [df.loc[i, metric]],
            s=90,
            zorder=5,
            color=_PALETTE["accent"],
            label=f"best α={df.loc[i, 'alpha']}",
        )
        ax.legend(frameon=False)
    ax.set_xlabel(r"Fusion weight $\alpha$ (ML contribution)")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, color=_PALETTE["grid"], alpha=0.35)
    fig.tight_layout()
    return _save(fig, out)


def plot_class_distribution(
    labels: list[str],
    out_path: str | Path,
    *,
    title: str = "Class distribution",
) -> Path:
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(counts.index.astype(str), counts.values, color=_PALETTE["bar"], edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _save(fig, Path(out_path))


def plot_confidence_distribution(
    confidences: list[float],
    out_path: str | Path,
    *,
    title: str = "Confidence distribution",
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(confidences, bins=15, color=_PALETTE["bar"], edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, Path(out_path))


def plot_alpha_distribution(
    alphas: list[float],
    out_path: str | Path,
    *,
    title: str = "Adaptive alpha distribution",
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(alphas, bins=12, color=_PALETTE["line"], edgecolor="white", linewidth=0.6)
    ax.axvline(float(np.mean(alphas)), color=_PALETTE["accent"], linestyle="--", label=f"mean={np.mean(alphas):.2f}")
    ax.set_xlabel(r"Adaptive $\alpha$")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, Path(out_path))


def plot_categorical_counts(
    counts: dict[str, int],
    out_path: str | Path,
    *,
    title: str,
    xlabel: str = "Category",
) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    ax.bar(keys, vals, color=_PALETTE["bar"], edgecolor="black", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save(fig, Path(out_path))


def plot_likert_bars(
    summary: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Bar chart of mean Likert scores (clarity / trust / plausibility)."""
    dims = summary.get("dimensions", {})
    names = list(dims.keys())
    means = [dims[n]["mean"] for n in names]
    stds = [dims[n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(names, means, yerr=stds, capsize=4, color=_PALETTE["bar"], edgecolor="black", linewidth=0.4)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean Likert (1–5)")
    ax.set_title("Explainability human ratings")
    fig.tight_layout()
    return _save(fig, Path(out_path))


def stitch_figures_pdf(figure_paths: list[str | Path], out_pdf: str | Path) -> Path:
    """Combine existing PNG/PDF figure files into a single multi-page PDF."""
    out = Path(out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for p in figure_paths:
            path = Path(p)
            if not path.is_file():
                continue
            img = plt.imread(path) if path.suffix.lower() == ".png" else None
            if img is None:
                continue
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(path.stem.replace("_", " "), fontsize=11)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return out
