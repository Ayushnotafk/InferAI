"""Automatic dataset statistics for InferAI publication reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.publication_figures import plot_class_distribution

_ROOT = Path(__file__).resolve().parent.parent
_TOKEN = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


def _label_col(df: pd.DataFrame) -> str:
    if "pramana_label" in df.columns:
        return "pramana_label"
    return "label"


def compute_dataset_statistics(
    csv_path: str | Path,
    *,
    name: str | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute class, length, and lexical diversity statistics for a CSV corpus."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    lab = _label_col(df)
    texts = df["text"].astype(str).tolist()
    token_lists = [_tokens(t) for t in texts]
    all_tokens = [tok for tl in token_lists for tok in tl]
    vocab = set(all_tokens)
    sent_lens = [len(tl) for tl in token_lists]
    char_lens = [len(t) for t in texts]

    claim_lens = (
        df["claim"].astype(str).map(lambda s: len(_tokens(s))).tolist()
        if "claim" in df.columns
        else []
    )
    premise_lens = []
    for col in ("premises", "evidence"):
        if col in df.columns:
            premise_lens = df[col].astype(str).map(lambda s: len(_tokens(s))).tolist()
            break

    n = len(df)
    ttr = (len(vocab) / len(all_tokens)) if all_tokens else 0.0
    guiraud = (len(vocab) / (len(all_tokens) ** 0.5)) if all_tokens else 0.0

    class_counts = df[lab].astype(str).value_counts().to_dict()
    stats: dict[str, Any] = {
        "name": name or path.stem,
        "path": str(path),
        "n_rows": n,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_proportions": {
            str(k): float(v) / n for k, v in class_counts.items()
        },
        "avg_sentence_tokens": float(sum(sent_lens) / n) if n else 0.0,
        "avg_sentence_chars": float(sum(char_lens) / n) if n else 0.0,
        "vocabulary_size": len(vocab),
        "total_tokens": len(all_tokens),
        "ttr": ttr,
        "guiraud_r": guiraud,
        "avg_claim_tokens": float(sum(claim_lens) / len(claim_lens)) if claim_lens else None,
        "avg_premise_tokens": (
            float(sum(premise_lens) / len(premise_lens)) if premise_lens else None
        ),
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = stats["name"]
        pd.DataFrame([stats]).to_csv(out / f"dataset_stats_{tag}.csv", index=False)
        pd.DataFrame(
            [{"label": k, "count": v} for k, v in stats["class_counts"].items()]
        ).to_csv(out / f"class_distribution_{tag}.csv", index=False)
        plot_class_distribution(
            df[lab].astype(str).tolist(),
            out / f"class_distribution_{tag}.png",
            title=f"Class distribution — {tag}",
        )
        # Sentence-length histogram (publication PNG + PDF).
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(sent_lens, bins=15, color="#115E59", edgecolor="white")
        ax.set_xlabel("Tokens per sentence")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Sentence length — {tag}")
        fig.tight_layout()
        png = out / f"sentence_length_{tag}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    return stats


def summarize_splits(
    train_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize train/test corpora used by InferAI."""
    train_csv = Path(train_csv or _ROOT / "dataset" / "raw" / "nyaya_dataset.csv")
    test_csv = Path(test_csv or _ROOT / "dataset" / "test_set.csv")
    result: dict[str, Any] = {}
    if train_csv.is_file():
        result["train"] = compute_dataset_statistics(
            train_csv, name="train", out_dir=out_dir
        )
    if test_csv.is_file():
        result["test"] = compute_dataset_statistics(
            test_csv, name="test", out_dir=out_dir
        )
    if out_dir and result:
        rows = []
        for split, st in result.items():
            row = {"split": split, **{k: v for k, v in st.items() if not isinstance(v, dict)}}
            rows.append(row)
        pd.DataFrame(rows).to_csv(Path(out_dir) / "dataset_split_summary.csv", index=False)
    return result
