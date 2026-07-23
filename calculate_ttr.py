"""
Type-Token Ratio (TTR) comparison for NyayaXAI / InferAI training corpora.

TTR = unique_tokens / total_tokens

Tokens are lowercased words after punctuation removal.

Usage:
    python calculate_ttr.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS = [
    ("nyaya_dataset (raw)", PROJECT_ROOT / "dataset" / "raw" / "nyaya_dataset.csv"),
    ("real_world_dataset", PROJECT_ROOT / "dataset" / "real_world_dataset.csv"),
]

FIGURE_PATH = PROJECT_ROOT / "reports" / "ttr_comparison.png"

# Keep letters, digits, and whitespace; drop other punctuation.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize_corpus(texts: pd.Series) -> list[str]:
    """Lowercase, strip punctuation, return word tokens."""
    tokens: list[str] = []
    for raw in texts.astype(str):
        cleaned = _PUNCT_RE.sub(" ", raw.lower())
        tokens.extend(_TOKEN_RE.findall(cleaned))
    return tokens


def compute_ttr(texts: pd.Series) -> tuple[int, int, float]:
    """
    Return (total_tokens, unique_tokens, ttr).

    TTR is 0.0 when there are no tokens.
    """
    tokens = tokenize_corpus(texts)
    total = len(tokens)
    if total == 0:
        return 0, 0, 0.0
    unique = len(set(tokens))
    return total, unique, unique / total


def main() -> None:
    names: list[str] = []
    ttr_values: list[float] = []

    print("=" * 60)
    print("Type-Token Ratio (TTR) — lexical diversity")
    print("=" * 60)

    for display_name, path in DATASETS:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)
        if "text" not in df.columns:
            raise ValueError(f"{path} must contain a 'text' column")

        total, unique, ttr = compute_ttr(df["text"])

        print(f"\nDataset: {display_name}")
        print(f"  Path:           {path}")
        print(f"  Rows:           {len(df)}")
        print(f"  Total tokens:   {total:,}")
        print(f"  Unique tokens:  {unique:,}")
        print(f"  TTR score:      {ttr:.4f}")

        names.append(display_name)
        ttr_values.append(ttr)

    # Bar chart
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#059669", "#10b981"]
    bars = ax.bar(names, ttr_values, color=colors[: len(names)], edgecolor="#022c22", linewidth=0.8)
    ax.set_ylabel("Type-Token Ratio (unique / total)")
    ax.set_ylim(0, min(1.05, max(ttr_values) * 1.15 + 0.05 if ttr_values else 1.0))
    ax.set_title("Lexical diversity (TTR) by dataset")
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.6, label="TTR = 1.0")
    for bar, val in zip(bars, ttr_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="600",
        )
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print(f"\nChart saved: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
