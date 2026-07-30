"""Plotting utilities for InferAI benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_vs_alpha(
    df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Line plot of a metric against fusion alpha."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["alpha"], df[metric], marker="o", linewidth=2, color="#14B8A6")
    ax.set_xlabel("Alpha (ML weight)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} vs Alpha")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ablation_summary(df: pd.DataFrame, out_dir: str | Path) -> list[str]:
    """Generate standard ablation plots; return saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for metric in ("accuracy", "macro_f1", "ece"):
        if metric not in df.columns:
            continue
        p = out_dir / f"{metric}_vs_alpha.png"
        plot_metric_vs_alpha(df, metric, p)
        paths.append(str(p))
    return paths
