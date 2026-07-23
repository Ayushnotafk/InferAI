"""
Generate dataset audit statistics for InferAI.

Outputs:
  - reports/dataset_statistics.md
  - reports/csv/dataset_sizes.csv
  - reports/csv/pramana_label_counts_*.csv
  - reports/csv/strength_label_counts_*.csv
  - reports/csv/train_val_test_split.csv

Usage (from project root):
    python reports/generate_dataset_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_generator import (
    _load_test_exclude_set,
    _norm_text_key,
    _standardize_schema,
)
from reports._common import (
    LABELS,
    STRENGTH_LABELS,
    dataframe_to_markdown,
    ensure_reports_dir,
    save_dataframe_csv,
)

NYAYA_RAW = _ROOT / "dataset" / "raw" / "nyaya_dataset.csv"
REAL_WORLD = _ROOT / "dataset" / "real_world_dataset.csv"
TEST_SET = _ROOT / "dataset" / "test_set.csv"
MERGED = _ROOT / "dataset" / "raw" / "nyaya_dataset_merged.csv"
CSV_DIR = _ROOT / "reports" / "csv"


def _label_counts(df: pd.DataFrame, col: str = "pramana_label") -> pd.DataFrame:
    counts = df[col].value_counts().reindex(LABELS, fill_value=0).astype(int)
    return pd.DataFrame({"Label": counts.index, "Count": counts.values})


def _strength_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "strength_label" not in df.columns:
        return pd.DataFrame({"Strength": STRENGTH_LABELS, "Count": [0, 0, 0]})
    s = df["strength_label"].dropna().astype(str).str.strip()
    s = s[s.isin(STRENGTH_LABELS)]
    counts = s.value_counts().reindex(STRENGTH_LABELS, fill_value=0).astype(int)
    return pd.DataFrame({"Strength": counts.index, "Count": counts.values})


def _merge_audit_stats() -> dict:
    """Replicate merge pipeline counters without writing files."""
    nyaya = _standardize_schema(pd.read_csv(NYAYA_RAW), "nyaya_dataset.csv")
    real = _standardize_schema(pd.read_csv(REAL_WORLD), "real_world_dataset.csv")
    combined = pd.concat([real, nyaya], ignore_index=True)

    exclude = _load_test_exclude_set()
    removed_test_overlap = 0
    if exclude:
        mask = ~_norm_text_key(combined["text"]).isin(exclude)
        removed_test_overlap = int((~mask).sum())
        combined = combined.loc[mask].copy()

    before_dedup = len(combined)
    combined["_text_key"] = _norm_text_key(combined["text"])
    deduped = combined.drop_duplicates(subset=["_text_key"], keep="first").drop(
        columns=["_text_key"]
    )
    duplicates_removed = before_dedup - len(deduped)

    return {
        "synthetic_size": len(nyaya),
        "real_world_size": len(real),
        "combined_before_dedup": before_dedup,
        "test_overlap_removed": removed_test_overlap,
        "duplicates_removed": duplicates_removed,
        "final_merged_size": len(deduped),
        "merged_df": deduped,
    }


def _split_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """80/20 train/validation split (matches train_model.py random_state=42)."""
    indices = list(range(len(merged_df)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    test_size = len(pd.read_csv(TEST_SET)) if TEST_SET.is_file() else 0
    rows = [
        {"Split": "Train (80% of merged)", "Count": len(train_idx)},
        {"Split": "Validation (20% random holdout)", "Count": len(val_idx)},
        {"Split": "Held-out test (test_set.csv)", "Count": test_size},
        {"Split": "Total unique evaluation rows", "Count": len(val_idx) + test_size},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ensure_reports_dir()
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_SET.is_file():
        raise FileNotFoundError(f"Missing {TEST_SET}")

    test_df = pd.read_csv(TEST_SET)
    audit = _merge_audit_stats()
    merged_df = audit["merged_df"]

    # Prefer on-disk merged file for distribution tables if present.
    if MERGED.is_file():
        merged_df = pd.read_csv(MERGED)

    sizes_df = pd.DataFrame(
        [
            {"Metric": "Synthetic dataset (nyaya_dataset.csv)", "Count": audit["synthetic_size"]},
            {"Metric": "Real-world dataset (real_world_dataset.csv)", "Count": audit["real_world_size"]},
            {"Metric": "Held-out test set (test_set.csv)", "Count": len(test_df)},
            {
                "Metric": "Combined size before deduplication (post test exclusion)",
                "Count": audit["combined_before_dedup"],
            },
            {"Metric": "Rows removed (overlap with test_set.csv)", "Count": audit["test_overlap_removed"]},
            {"Metric": "Duplicate texts removed", "Count": audit["duplicates_removed"]},
            {"Metric": "Final merged training size", "Count": audit["final_merged_size"]},
        ]
    )

    save_dataframe_csv(sizes_df, CSV_DIR / "dataset_sizes.csv")

    label_tables: dict[str, pd.DataFrame] = {
        "synthetic": _label_counts(_standardize_schema(pd.read_csv(NYAYA_RAW), "nyaya")),
        "real_world": _label_counts(_standardize_schema(pd.read_csv(REAL_WORLD), "real")),
        "test_set": _label_counts(
            _standardize_schema(test_df, "test_set"), col="pramana_label"
        ),
        "merged": _label_counts(merged_df),
    }
    for name, tbl in label_tables.items():
        save_dataframe_csv(tbl, CSV_DIR / f"pramana_label_counts_{name}.csv")

    strength_tables: dict[str, pd.DataFrame] = {
        "real_world": _strength_counts(_standardize_schema(pd.read_csv(REAL_WORLD), "real")),
        "test_set": _strength_counts(_standardize_schema(test_df, "test_set")),
        "merged": _strength_counts(merged_df),
    }
    for name, tbl in strength_tables.items():
        save_dataframe_csv(tbl, CSV_DIR / f"strength_label_counts_{name}.csv")

    split_df = _split_summary(merged_df)
    save_dataframe_csv(split_df, CSV_DIR / "train_val_test_split.csv")

    lines: list[str] = []
    lines.append("# InferAI — Dataset Statistics Audit\n\n")
    lines.append(
        "Automated audit of corpus sizes, label balance, strength labels, and "
        "train/validation/test partitioning. Generated by `reports/generate_dataset_statistics.py`.\n\n"
    )

    lines.append("## 1. Dataset sizes\n\n")
    lines.append(dataframe_to_markdown(sizes_df) + "\n\n")

    lines.append("## 2. Class distributions (`pramana_label`)\n\n")
    for title, key in [
        ("Synthetic (`nyaya_dataset.csv`)", "synthetic"),
        ("Real-world (`real_world_dataset.csv`)", "real_world"),
        ("Held-out test (`test_set.csv`)", "test_set"),
        ("Merged training table (`nyaya_dataset_merged.csv`)", "merged"),
    ]:
        lines.append(f"### {title}\n\n")
        lines.append(dataframe_to_markdown(label_tables[key]) + "\n\n")

    lines.append("## 3. Strength distributions\n\n")
    for title, key in [
        ("Real-world dataset", "real_world"),
        ("Held-out test set", "test_set"),
        ("Merged training table", "merged"),
    ]:
        lines.append(f"### {title}\n\n")
        lines.append(dataframe_to_markdown(strength_tables[key]) + "\n\n")
    lines.append(
        "_Note: the synthetic corpus does not carry curated `strength_label` metadata._\n\n"
    )

    lines.append("## 4. Train / validation / test split summary\n\n")
    lines.append(
        "Training uses `nyaya_dataset_merged.csv` with an 80/20 stratified random split "
        "(`random_state=42`, matching `classification/train_model.py`). "
        "`dataset/test_set.csv` is held out entirely during merge and training.\n\n"
    )
    lines.append(dataframe_to_markdown(split_df) + "\n\n")

    lines.append("## 5. CSV exports\n\n")
    lines.append("| File | Description |\n")
    lines.append("| --- | --- |\n")
    for rel in sorted(CSV_DIR.glob("*.csv")):
        lines.append(f"| `{rel.relative_to(_ROOT).as_posix()}` | Tabular export |\n")

    out = _ROOT / "reports" / "dataset_statistics.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
