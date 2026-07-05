"""
Export reviewed AAEC manual labels as a training-ready table.

Inputs:
  dataset/processed/aaec_manual_review_250_corrected.csv

Outputs:
  dataset/processed/aaec_reviewed_dataset.csv
  reports/aaec_reviewed_dataset_stats.md

Usage:
    python dataset/build_aaec_reviewed_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.build_aaec_nyaya_annotation_sheet import PRAMANA_LABELS
from reports._common import dataframe_to_markdown, ensure_reports_dir

INPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "aaec_manual_review_250_corrected.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "aaec_reviewed_dataset.csv"
STATS_OUT = PROJECT_ROOT / "reports" / "aaec_reviewed_dataset_stats.md"

STRENGTH_LABELS = ("weak", "moderate", "strong")


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _normalize_strength(value: str) -> str:
    s = str(value).strip().lower()
    if s not in STRENGTH_LABELS:
        raise ValueError(f"Invalid strength_label: {value!r}")
    return s


def _normalize_pramana(value: str) -> str:
    label = str(value).strip()
    if label not in PRAMANA_LABELS:
        raise ValueError(f"Invalid pramana_label: {value!r}")
    return label


def build_reviewed_dataset() -> tuple[pd.DataFrame, int, int]:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    reviewed = df["reviewed"].astype(str).str.strip().str.lower() == "yes"
    filtered = df.loc[reviewed].copy()
    if filtered.empty:
        raise ValueError("No rows with reviewed=yes found.")

    before_dedup = len(filtered)
    out = pd.DataFrame()
    out["text"] = filtered["text"].astype(str).map(_normalize_text)
    out["pramana_label"] = filtered["final_pramana"].astype(str).map(_normalize_pramana)
    out["strength_label"] = filtered["strength_label"].astype(str).map(_normalize_strength)

    out = out.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(out)
    return out, before_dedup, duplicates_removed


def write_stats(df: pd.DataFrame, before_dedup: int, duplicates_removed: int) -> None:
    ensure_reports_dir()
    total = len(df)

    pramana = (
        df["pramana_label"]
        .value_counts()
        .reindex(PRAMANA_LABELS, fill_value=0)
        .rename_axis("pramana_label")
        .reset_index(name="count")
    )
    pramana["percentage"] = (100.0 * pramana["count"] / total).round(2)

    strength = (
        df["strength_label"]
        .value_counts()
        .reindex(list(STRENGTH_LABELS), fill_value=0)
        .rename_axis("strength_label")
        .reset_index(name="count")
    )
    strength["percentage"] = (100.0 * strength["count"] / total).round(2)

    lines: list[str] = []
    lines.append("# AAEC Reviewed Dataset — Statistics\n\n")
    lines.append(
        "Exported from `aaec_manual_review_250_corrected.csv` (`reviewed=yes` only).\n\n"
    )
    lines.append("## Summary\n\n")
    lines.append(f"- **Total rows:** {total}\n")
    lines.append(f"- Rows before deduplication: {before_dedup}\n")
    lines.append(f"- **Duplicates removed:** {duplicates_removed}\n\n")
    lines.append("## Class distribution (`pramana_label`)\n\n")
    lines.append(dataframe_to_markdown(pramana) + "\n\n")
    lines.append("## Strength distribution (`strength_label`)\n\n")
    lines.append(dataframe_to_markdown(strength) + "\n\n")
    lines.append("## Schema\n\n")
    lines.append("```text\n")
    lines.append("text, pramana_label, strength_label\n")
    lines.append("```\n")

    STATS_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    df, before_dedup, duplicates_removed = build_reviewed_dataset()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    write_stats(df, before_dedup, duplicates_removed)

    print("AAEC reviewed dataset")
    print("=" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print("\nClass distribution:")
    print(df["pramana_label"].value_counts().reindex(PRAMANA_LABELS, fill_value=0).to_string())
    print("\nStrength distribution:")
    print(df["strength_label"].value_counts().reindex(list(STRENGTH_LABELS), fill_value=0).to_string())
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {STATS_OUT}")


if __name__ == "__main__":
    main()
