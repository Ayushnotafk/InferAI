"""
Export reviewed IBM Nyaya annotations as a training-ready table.

Inputs:
  dataset/processed/nyaya_annotated_1.csv

Outputs:
  dataset/processed/ibm_reviewed_dataset.csv
  reports/ibm_reviewed_dataset_stats.md

Usage:
    python dataset/build_ibm_reviewed_dataset.py
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

INPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "nyaya_annotated_1.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "ibm_reviewed_dataset.csv"
STATS_OUT = PROJECT_ROOT / "reports" / "ibm_reviewed_dataset_stats.md"

SCHEMA_COLUMNS = ("text", "pramana_label", "strength_label")
STRENGTH_LABELS = ("weak", "moderate", "strong")
PREVIEW_N = 20


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


def build_reviewed_dataset() -> tuple[pd.DataFrame, int]:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    reviewed = df["reviewed"].astype(str).str.strip().str.lower() == "yes"
    filtered = df.loc[reviewed].copy()
    if filtered.empty:
        raise ValueError('No rows with reviewed="yes" found.')

    before_dedup = len(filtered)
    out = pd.DataFrame()
    out["text"] = filtered["text"].astype(str).map(_normalize_text)
    out["pramana_label"] = filtered["final_pramana"].astype(str).map(_normalize_pramana)
    out["strength_label"] = filtered["strength_label"].astype(str).map(_normalize_strength)

    out = out[out["text"].str.len() > 0]
    out = out.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(out)
    return out[list(SCHEMA_COLUMNS)], duplicates_removed


def write_stats(df: pd.DataFrame, duplicates_removed: int) -> None:
    ensure_reports_dir()
    total = len(df)
    avg_len = float(df["text"].str.len().mean()) if total else 0.0

    pramana = (
        df["pramana_label"]
        .value_counts()
        .reindex(PRAMANA_LABELS, fill_value=0)
        .rename_axis("pramana_label")
        .reset_index(name="count")
    )
    pramana["percentage"] = (100.0 * pramana["count"] / total).round(2) if total else 0.0

    strength = (
        df["strength_label"]
        .value_counts()
        .reindex(list(STRENGTH_LABELS), fill_value=0)
        .rename_axis("strength_label")
        .reset_index(name="count")
    )
    strength["percentage"] = (100.0 * strength["count"] / total).round(2) if total else 0.0

    lines: list[str] = []
    lines.append("# IBM Reviewed Dataset — Statistics\n\n")
    lines.append('Exported from `nyaya_annotated_1.csv` (`reviewed="yes"` only).\n\n')
    lines.append("## Summary\n\n")
    lines.append(f"- **Total rows:** {total}\n")
    lines.append(f"- **Duplicates removed:** {duplicates_removed}\n")
    lines.append(f"- **Average text length (chars):** {avg_len:.1f}\n\n")
    lines.append("## pramana_label distribution\n\n")
    lines.append(dataframe_to_markdown(pramana) + "\n\n")
    lines.append("## strength_label distribution\n\n")
    lines.append(dataframe_to_markdown(strength) + "\n\n")
    lines.append("## Schema\n\n")
    lines.append("```text\n")
    lines.append("text,pramana_label,strength_label\n")
    lines.append("```\n")

    STATS_OUT.write_text("".join(lines), encoding="utf-8")


def print_preview(df: pd.DataFrame, n: int = PREVIEW_N) -> None:
    print("\n" + "=" * 72)
    print(f"First {min(n, len(df))} exported samples")
    print("=" * 72)
    for i, row in enumerate(df.head(n).itertuples(index=False), 1):
        text = row.text if len(row.text) <= 200 else row.text[:200] + "…"
        print(f"\n{i}. pramana_label={row.pramana_label}, strength_label={row.strength_label}")
        print(f"   text: {text}")


def main() -> None:
    df, duplicates_removed = build_reviewed_dataset()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    write_stats(df, duplicates_removed)
    print_preview(df)

    print("\n" + "=" * 72)
    print("IBM reviewed dataset export complete")
    print("=" * 72)
    print(f"Total rows: {len(df)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Average text length: {df['text'].str.len().mean():.1f} chars")
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {STATS_OUT}")


if __name__ == "__main__":
    main()
