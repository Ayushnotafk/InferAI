"""
Prepare double-blind annotation sheets for inter-annotator agreement.

Loads ``dataset/real_world_dataset.csv``, assigns stable integer IDs, drops
gold labels, and writes two independently shuffled copies for annotators.

Usage (from project root):

    python prepare_annotation_sheet.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_CSV = PROJECT_ROOT / "dataset" / "real_world_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "annotations"
ANNOTATOR1_PATH = OUTPUT_DIR / "annotator1.csv"
ANNOTATOR2_PATH = OUTPUT_DIR / "annotator2.csv"


def main() -> None:
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_CSV}\n"
            "Run: python dataset/build_real_world_dataset.py"
        )

    df = pd.read_csv(SOURCE_CSV)

    required = {"text", "pramana_label", "strength_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source CSV missing columns: {sorted(missing)}")

    # Stable IDs 1..N in source row order (same ids in both annotator files).
    base = pd.DataFrame({"id": range(1, len(df) + 1), "text": df["text"].astype(str)})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Independent shuffles: same (id, text) pairs, different presentation order.
    annotator1 = base.sample(frac=1.0, random_state=42).reset_index(drop=True)
    annotator2 = base.sample(frac=1.0, random_state=137).reset_index(drop=True)

    annotator1.to_csv(ANNOTATOR1_PATH, index=False)
    annotator2.to_csv(ANNOTATOR2_PATH, index=False)

    n = len(base)
    print(f"Total examples processed: {n}")
    print(f"Annotator 1 sheet: {ANNOTATOR1_PATH}")
    print(f"Annotator 2 sheet: {ANNOTATOR2_PATH}")
    print(
        "Columns in each file: id, text "
        "(pramana_label and strength_label removed for blind annotation)."
    )


if __name__ == "__main__":
    main()
