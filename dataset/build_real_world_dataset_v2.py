"""
Build real_world_dataset_v2.csv from the curated v1 corpus plus reviewed AAEC rows.

Inputs:
  - dataset/real_world_dataset.csv
  - dataset/processed/aaec_manual_review_250_corrected.csv

Outputs:
  - dataset/real_world_dataset_v2.csv
  - reports/real_world_dataset_v2_stats.md

Usage:
    python dataset/build_real_world_dataset_v2.py
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

V1_CSV = PROJECT_ROOT / "dataset" / "real_world_dataset.csv"
AAEC_CSV = PROJECT_ROOT / "dataset" / "processed" / "aaec_manual_review_250_corrected.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "real_world_dataset_v2.csv"
STATS_OUT = PROJECT_ROOT / "reports" / "real_world_dataset_v2_stats.md"

SCHEMA_COLUMNS = ("text", "pramana_label", "strength_label")
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


def load_v1() -> pd.DataFrame:
    df = pd.read_csv(V1_CSV)
    missing = set(SCHEMA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{V1_CSV} missing columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["text"] = df["text"].astype(str).map(_normalize_text)
    out["pramana_label"] = df["pramana_label"].astype(str).map(_normalize_pramana)
    out["strength_label"] = df["strength_label"].astype(str).map(_normalize_strength)
    out["source"] = "real_world_v1"
    return out


def load_reviewed_aaec() -> pd.DataFrame:
    if not AAEC_CSV.is_file():
        raise FileNotFoundError(f"Missing {AAEC_CSV}")

    df = pd.read_csv(AAEC_CSV)
    reviewed = df["reviewed"].astype(str).str.strip().str.lower() == "yes"
    df = df.loc[reviewed].copy()

    if df.empty:
        raise ValueError("No rows with reviewed=yes in AAEC corrected file.")

    out = pd.DataFrame()
    out["text"] = df["text"].astype(str).map(_normalize_text)
    out["pramana_label"] = df["final_pramana"].astype(str).map(_normalize_pramana)
    out["strength_label"] = df["strength_label"].astype(str).map(_normalize_strength)
    out["source"] = "aaec_manual_review"
    return out


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    deduped = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    return deduped, before - len(deduped)


def write_stats(
    v1_count: int,
    aaec_count: int,
    combined_before: int,
    duplicates_removed: int,
    final_df: pd.DataFrame,
) -> None:
    ensure_reports_dir()
    total = len(final_df)

    pramana = (
        final_df["pramana_label"]
        .value_counts()
        .reindex(PRAMANA_LABELS, fill_value=0)
        .rename_axis("pramana_label")
        .reset_index(name="count")
    )
    pramana["percentage"] = (100.0 * pramana["count"] / total).round(2)

    strength = (
        final_df["strength_label"]
        .value_counts()
        .reindex(list(STRENGTH_LABELS), fill_value=0)
        .rename_axis("strength_label")
        .reset_index(name="count")
    )
    strength["percentage"] = (100.0 * strength["count"] / total).round(2)

    source = final_df["source"].value_counts().reset_index()
    source.columns = ["source", "count"]

    lines: list[str] = []
    lines.append("# Real-World Dataset v2 — Statistics\n\n")
    lines.append(
        "Merged corpus from `real_world_dataset.csv` and reviewed AAEC rows "
        "(`aaec_manual_review_250_corrected.csv`, `reviewed=yes`).\n\n"
    )

    lines.append("## Summary\n\n")
    lines.append(f"- **Total rows:** {total}\n")
    lines.append(f"- Rows from v1 (`real_world_dataset.csv`): {v1_count}\n")
    lines.append(f"- Rows from reviewed AAEC: {aaec_count}\n")
    lines.append(f"- Combined before deduplication: {combined_before}\n")
    lines.append(f"- **Exact duplicates removed:** {duplicates_removed}\n\n")

    lines.append("## Class distribution (`pramana_label`)\n\n")
    lines.append(dataframe_to_markdown(pramana) + "\n\n")

    lines.append("## Strength distribution (`strength_label`)\n\n")
    lines.append(dataframe_to_markdown(strength) + "\n\n")

    lines.append("## Source breakdown\n\n")
    lines.append(dataframe_to_markdown(source) + "\n\n")

    lines.append("## Schema\n\n")
    lines.append("```text\n")
    lines.append("text, pramana_label, strength_label\n")
    lines.append("```\n")

    STATS_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    v1 = load_v1()
    aaec = load_reviewed_aaec()

    combined = pd.concat([v1, aaec], ignore_index=True)
    combined_before = len(combined)
    deduped, duplicates_removed = remove_exact_duplicates(combined)

    # Export schema-matched columns only
    export = deduped[list(SCHEMA_COLUMNS)].copy()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    write_stats(
        v1_count=len(v1),
        aaec_count=len(aaec),
        combined_before=combined_before,
        duplicates_removed=duplicates_removed,
        final_df=deduped,
    )

    print("Real-world dataset v2")
    print("=" * 50)
    print(f"Total rows: {len(export)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print("\nClass distribution:")
    print(export["pramana_label"].value_counts().reindex(PRAMANA_LABELS, fill_value=0).to_string())
    print("\nStrength distribution:")
    print(export["strength_label"].value_counts().reindex(list(STRENGTH_LABELS), fill_value=0).to_string())
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {STATS_OUT}")


if __name__ == "__main__":
    main()
