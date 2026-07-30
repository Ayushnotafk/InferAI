"""
Dataset utilities for InferAI research workflows.

Does not modify on-disk training corpora; provides load, validate, split, and export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

VALID_LABELS = frozenset({"Pratyaksha", "Anumana", "Upamana", "Shabda"})

GOLD_REQUIRED_FIELDS = ("text", "claim", "premises", "label")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSON-lines gold dataset."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def validate_labels(df: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    """Check label column against the four Nyāya pramāṇa classes."""
    if label_col not in df.columns and "pramana_label" in df.columns:
        label_col = "pramana_label"
    unknown = sorted(set(df[label_col].astype(str)) - VALID_LABELS)
    counts = df[label_col].value_counts().to_dict()
    return {
        "valid": len(unknown) == 0,
        "label_column": label_col,
        "unknown_labels": unknown,
        "counts": counts,
        "n_rows": len(df),
    }


def validate_gold_format(df: pd.DataFrame) -> dict[str, Any]:
    """Verify gold-annotation schema consistency."""
    missing_cols = [c for c in GOLD_REQUIRED_FIELDS if c not in df.columns]
    issues: list[str] = list(missing_cols)
    if "label" in df.columns:
        bad = df[~df["label"].astype(str).isin(VALID_LABELS)]
        if len(bad):
            issues.append(f"{len(bad)} rows with invalid label values")
    empty_text = int((df.get("text", pd.Series(dtype=str)).astype(str).str.strip() == "").sum())
    if empty_text:
        issues.append(f"{empty_text} rows with empty text")
    return {
        "valid": len(issues) == 0,
        "required_fields": list(GOLD_REQUIRED_FIELDS),
        "issues": issues,
        "n_rows": len(df),
    }


def export_cleaned(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Write a normalized copy (whitespace-collapsed text fields)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    for col in ("text", "claim", "premises"):
        if col in clean.columns:
            clean[col] = clean[col].astype(str).map(lambda s: " ".join(s.split()))
    clean.to_csv(out, index=False)
    return out


def train_test_split_df(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    label_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split when labels are available."""
    label_col = label_col or (
        "pramana_label" if "pramana_label" in df.columns else "label"
    )
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col] if label_col in df.columns else None,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
