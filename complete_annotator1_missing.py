"""
Fill missing pramana_label values in annotator1_completed.csv using Nyaya rules.

Preserves any labels already present; only empty/missing cells are inferred.

Usage:
    python complete_annotator1_missing.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from auto_annotate_second_annotator import classify_pramana

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "dataset" / "annotations" / "annotator1_completed.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "annotations" / "annotator1_completed_final.csv"

VALID_LABELS = {"Pratyaksha", "Anumana", "Upamana", "Shabda"}


def _is_missing(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return str(value).strip() == ""


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    for col in ("id", "text"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "pramana_label" not in df.columns:
        df["pramana_label"] = pd.NA

    missing_mask = df["pramana_label"].map(_is_missing)
    n_missing = int(missing_mask.sum())
    n_total = len(df)

    filled = 0
    for idx in df.index[missing_mask]:
        text = str(df.at[idx, "text"])
        df.at[idx, "pramana_label"] = classify_pramana(text)
        filled += 1

    # Normalize existing labels (strip whitespace); validate
    df["pramana_label"] = df["pramana_label"].astype(str).str.strip()
    bad = set(df["pramana_label"].unique()) - VALID_LABELS
    if bad:
        raise ValueError(f"Invalid labels present: {bad}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Total rows: {n_total}")
    print(f"Missing labels filled: {filled}")
    print(f"Labels preserved (already present): {n_total - n_missing}")
    print(f"\nSaved: {OUTPUT_CSV}")
    print("\nClass distribution (final):")
    dist = df["pramana_label"].value_counts().reindex(sorted(VALID_LABELS), fill_value=0)
    for label in sorted(VALID_LABELS):
        count = int(dist[label])
        pct = 100.0 * count / n_total if n_total else 0.0
        print(f"  {label:12s} {count:4d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
