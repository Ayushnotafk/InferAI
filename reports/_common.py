"""
Shared helpers for InferAI research evaluation reports.

All report scripts should be run from the project root.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

LABELS = ["Pratyaksha", "Anumana", "Upamana", "Shabda"]
STRENGTH_LABELS = ["Weak", "Moderate", "Strong"]

# Publication-style defaults (matplotlib)
PLOT_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}


def apply_plot_style() -> None:
    """Apply consistent matplotlib styling for report figures."""
    plt.rcParams.update(PLOT_RC)


def ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def dataframe_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        # Fallback when tabulate is unavailable.
        headers = list(df.columns)
        if index:
            headers = [df.index.name or ""] + headers
        lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        if index:
            for idx, row in df.iterrows():
                cells = [str(idx)] + [str(row[c]) for c in df.columns]
                lines.append("| " + " | ".join(cells) + " |")
        else:
            for _, row in df.iterrows():
                cells = [str(row[c]) for c in df.columns]
                lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
