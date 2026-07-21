"""
Generate professor-review documentation using canonical dataset counts.

Outputs:
  - docs/dataset_provenance.md
  - reports/dataset_summary.md
  - reports/train_test_split.md

Usage (from project root):
    python reports/generate_professor_documentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reports._common import LABELS, STRENGTH_LABELS, dataframe_to_markdown, ensure_reports_dir
from reports.dataset_canonical import (
    AAEC_REVIEWED,
    IBM_REVIEWED,
    MERGE_RAW_BEFORE_DEDUP,
    MERGED_CORPUS,
    REAL_WORLD_REVIEWED,
    RETRAIN_DUPES_REMOVED,
    SYNTHETIC_RAW,
    SYNTHETIC_USED,
    TRAIN_FRACTION,
    TRAIN_VAL_RANDOM_STATE,
)

SYNTHETIC = _ROOT / "dataset" / "raw" / "nyaya_dataset.csv"
REAL_WORLD = _ROOT / "dataset" / "real_world_dataset.csv"
TEST_SET = _ROOT / "dataset" / "test_set.csv"
EARLY_MERGED = _ROOT / "dataset" / "raw" / "nyaya_dataset_merged.csv"
MERGED_RETRAIN = _ROOT / "dataset" / "processed" / "merged_retraining_corpus.csv"
AAEC_EXTRACTED = _ROOT / "dataset" / "processed" / "aaec_extracted.csv"
IBM_RAW = _ROOT / "dataset" / "external" / "IBM_ARGKP" / "ArgKP_combined.csv"
IBM_EXTRACTED = _ROOT / "dataset" / "processed" / "ibm_argkp_extracted.csv"

PROVENANCE_OUT = _ROOT / "docs" / "dataset_provenance.md"
SUMMARY_OUT = _ROOT / "reports" / "dataset_summary.md"
SPLIT_OUT = _ROOT / "reports" / "train_test_split.md"


def _count(path: Path) -> int:
    return len(pd.read_csv(path)) if path.is_file() else 0


def _train_val_sizes() -> tuple[int, int]:
    idx = list(range(MERGED_CORPUS))
    train_idx, val_idx = train_test_split(
        idx, test_size=1.0 - TRAIN_FRACTION, random_state=TRAIN_VAL_RANDOM_STATE
    )
    return len(train_idx), len(val_idx)


def _optional_merged_class_table() -> str:
    """Include per-class table only when on-disk merge matches canonical size."""
    if not MERGED_RETRAIN.is_file():
        return ""
    df = pd.read_csv(MERGED_RETRAIN)
    if len(df) != MERGED_CORPUS:
        return (
            f"_Note: `merged_retraining_corpus.csv` currently has {len(df)} rows; "
            f"canonical finalized size is **{MERGED_CORPUS}**. Rebuild the merge CSV "
            "to refresh per-class counts below._\n\n"
        )
    vc = df["pramana_label"].value_counts().reindex(LABELS, fill_value=0).astype(int)
    tbl = pd.DataFrame({"Pramana": vc.index, "Count": vc.values})
    return dataframe_to_markdown(tbl) + "\n\n"


def write_provenance() -> None:
    aaec_components = _count(AAEC_EXTRACTED)
    ibm_raw = _count(IBM_RAW)
    ibm_clean = _count(IBM_EXTRACTED)
    early_merged = _count(EARLY_MERGED)
    test_n = _count(TEST_SET)
    real_world_n = _count(REAL_WORLD)

    lines: list[str] = []
    lines.append("# Dataset Provenance\n\n")
    lines.append(
        "Document describing the origin, processing, and final use of every corpus in InferAI. "
        "Canonical counts are defined in `reports/dataset_canonical.py`. Regenerate with "
        "`python reports/generate_professor_documentation.py`.\n\n"
    )

    lines.append("## Synthetic Nyaya Dataset\n\n")
    lines.append("- **Source:** Generated in-project by `dataset_generator.py` and `dataset/labeled_corpus.py`.\n")
    lines.append(
        "- **Why it was created:** To provide a large, balanced seed corpus of English arguments "
        "labelled with Nyāya pramāṇa categories when insufficient manually annotated data was available.\n"
    )
    lines.append(
        "- **Generation methodology:** Template-based expansion across eleven domain slots "
        "with deterministic label assignment per template (`dataset/labeled_corpus.py`).\n"
    )
    lines.append(f"- **Number of samples:** **{SYNTHETIC_RAW}** rows in `dataset/raw/nyaya_dataset.csv`.\n")
    lines.append(
        "- **Preprocessing:** Text normalised (whitespace collapse); schema standardised to "
        "`text`, `pramana_label` (or `label`).\n"
    )
    lines.append(
        "- **Deduplication:** Exact-text deduplication applied during merge with reviewed corpora "
        "(`classification/retrain_merged.py`).\n"
    )
    lines.append(
        f"- **Final samples used in retraining merge:** **{SYNTHETIC_USED}** of {SYNTHETIC_RAW} synthetic rows "
        f"retained in `merged_retraining_corpus.csv` after cross-corpus deduplication.\n\n"
    )

    lines.append("## AAEC\n\n")
    lines.append(
        "- **Full dataset name:** Argument Annotated Essays Corpus (AAEC) — brat standoff annotations "
        "of persuasive essays.\n"
    )
    lines.append("- **Original source:** `dataset/external/AAEC/` (402 essay `.txt` / `.ann` pairs).\n")
    lines.append(
        "- **License:** No separate license file is bundled in this repository; comply with the "
        "terms of the original AAEC distribution from which the files were obtained.\n"
    )
    lines.append("- **Number of essays:** **402** (`reports/aaec_statistics.md`).\n")
    lines.append(
        f"- **Number of extracted components:** **{aaec_components}** MajorClaim, Claim, and Premise "
        f"spans (`dataset/parse_aaec.py` → `aaec_extracted.csv`).\n"
    )
    lines.append(
        "- **Manual review process:** We built annotation sheets (`build_aaec_nyaya_annotation_sheet.py`), "
        "conducted manual Nyāya review in staged batches (including an initial 250-row sheet), and exported "
        "the finalized reviewed set to `aaec_reviewed_dataset.csv` where `reviewed=yes` "
        "(`dataset/build_aaec_reviewed_dataset.py`). Guidelines: `dataset/labeling_guidelines.md`.\n"
    )
    lines.append(
        f"- **Final reviewed count:** **{AAEC_REVIEWED}** rows in `aaec_reviewed_dataset.csv`.\n"
    )
    lines.append(
        "- **Preprocessing steps:** Parse brat `.ann` files; extract text-bound spans; remove empty "
        "and duplicate rows (39 removed at extraction); normalise whitespace; deduplicate on text at export.\n\n"
    )

    lines.append("## IBM ArgKP\n\n")
    lines.append("- **Full dataset name:** IBM Argument Knowledge Pairing (ArgKP) combined corpus.\n")
    lines.append("- **Source:** `dataset/external/IBM_ARGKP/ArgKP_combined.csv`.\n")
    lines.append(
        "- **License:** No separate license file is bundled in this repository; comply with the "
        "terms of the original IBM ArgKP release.\n"
    )
    lines.append(f"- **Raw size:** **{ibm_raw}** rows (`reports/ibm_argkp_statistics.md`).\n")
    lines.append(
        f"- **Cleaning pipeline:** `dataset/parse_ibm_argkp.py` — normalise text, drop empty rows, "
        f"deduplicate arguments → **{ibm_clean}** unique rows ({ibm_raw - ibm_clean} duplicates removed).\n"
    )
    lines.append(
        "- **Manual review:** 600-row Nyāya annotation sheet; human review in `nyaya_annotated_1.csv`; "
        "export where `reviewed=yes` (`dataset/build_ibm_reviewed_dataset.py`).\n"
    )
    lines.append(f"- **Final reviewed count:** **{IBM_REVIEWED}** rows in `ibm_reviewed_dataset.csv`.\n\n")

    lines.append("## Final Merged Dataset\n\n")
    lines.append(
        "The production retraining table is `dataset/processed/merged_retraining_corpus.csv`, "
        "built by `classification/retrain_merged.py`. The held-out evaluation set "
        "`dataset/test_set.csv` is **never** included in this merge.\n\n"
    )
    lines.append("**Table — Corpus pipeline summary**\n\n")
    lines.append("| Dataset | Raw | Clean | Reviewed | Used (retraining merge) |\n")
    lines.append("|---------|----:|------:|---------:|------------------------:|\n")
    lines.append(f"| Synthetic Nyaya | {SYNTHETIC_RAW} | {SYNTHETIC_RAW} | — | {SYNTHETIC_USED} |\n")
    lines.append(f"| Real-world (curated v1) | {real_world_n} | {real_world_n} | {real_world_n} | 0* |\n")
    lines.append(f"| AAEC | {aaec_components} | {aaec_components} | {AAEC_REVIEWED} | {AAEC_REVIEWED} |\n")
    lines.append(f"| IBM ArgKP | {ibm_raw} | {ibm_clean} | {IBM_REVIEWED} | {IBM_REVIEWED} |\n")
    lines.append(f"| **Merged retraining corpus** | {MERGE_RAW_BEFORE_DEDUP} | — | — | **{MERGED_CORPUS}** |\n")
    lines.append(f"| Held-out test (`test_set.csv`) | {test_n} | {test_n} | — | **0 (excluded)** |\n\n")
    lines.append(
        f"\\*The early merge `nyaya_dataset_merged.csv` ({early_merged} rows) combined synthetic + "
        "real-world v1 for initial experiments; the current retraining merge uses synthetic + AAEC + IBM reviewed.\n\n"
    )
    lines.append(
        f"**Composition:** {SYNTHETIC_USED} + {AAEC_REVIEWED} + {IBM_REVIEWED} = **{MERGED_CORPUS}** "
        f"(after {RETRAIN_DUPES_REMOVED} duplicate texts removed during merge).\n"
    )

    PROVENANCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {PROVENANCE_OUT}")


def write_dataset_summary() -> None:
    test_n = _count(TEST_SET)
    real_n = _count(REAL_WORLD)
    aaec_ext = _count(AAEC_EXTRACTED)
    ibm_raw = _count(IBM_RAW)
    ibm_clean = _count(IBM_EXTRACTED)
    train_n, val_n = _train_val_sizes()

    lines: list[str] = []
    lines.append("# Dataset Summary\n\n")
    lines.append(
        "Reproducible summary of InferAI corpora. Canonical counts from `reports/dataset_canonical.py`. "
        "Generated by `reports/generate_professor_documentation.py`.\n\n"
    )

    lines.append("## Corpus pipeline\n\n")
    lines.append("| Dataset | Raw | Clean | Reviewed | Final Used |\n")
    lines.append("|---------|----:|------:|---------:|-----------:|\n")
    lines.append(f"| Synthetic Nyaya | {SYNTHETIC_RAW} | {SYNTHETIC_RAW} | — | {SYNTHETIC_USED} |\n")
    lines.append(f"| Real-world (v1) | {real_n} | {real_n} | {real_n} | 0 |\n")
    lines.append(f"| AAEC | {aaec_ext} | {aaec_ext} | {AAEC_REVIEWED} | {AAEC_REVIEWED} |\n")
    lines.append(f"| IBM ArgKP | {ibm_raw} | {ibm_clean} | {IBM_REVIEWED} | {IBM_REVIEWED} |\n")
    lines.append(f"| **Merged retraining** | {MERGE_RAW_BEFORE_DEDUP} | — | — | **{MERGED_CORPUS}** |\n")
    lines.append(f"| Held-out test | {test_n} | {test_n} | — | **0** |\n\n")

    lines.append("## Final merged corpus size\n\n")
    lines.append(
        f"**{MERGED_CORPUS}** unique training rows "
        f"({SYNTHETIC_USED} synthetic + {AAEC_REVIEWED} AAEC reviewed + {IBM_REVIEWED} IBM reviewed) "
        "in `dataset/processed/merged_retraining_corpus.csv`.\n\n"
    )

    lines.append("### Manually reviewed real-world total\n\n")
    lines.append(f"**{REAL_WORLD_REVIEWED}** ({AAEC_REVIEWED} AAEC + {IBM_REVIEWED} IBM).\n\n")

    lines.append("### Per-class distribution (merged retraining corpus)\n\n")
    lines.append(_optional_merged_class_table())

    lines.append("## Train / validation / test counts\n\n")
    lines.append("| Split | Count | Notes |\n")
    lines.append("|-------|------:|-------|\n")
    lines.append(f"| Train (80% of merged) | {train_n} | `random_state={TRAIN_VAL_RANDOM_STATE}` |\n")
    lines.append(f"| Validation (20% of merged) | {val_n} | Random holdout |\n")
    lines.append(f"| Held-out test | {test_n} | `dataset/test_set.csv` |\n\n")

    lines.append(
        "**The held-out evaluation dataset is excluded from training.** "
        "`dataset/test_set.csv` is removed before merge in `dataset_generator.py` and is not "
        "concatenated in `classification/retrain_merged.py`.\n\n"
    )

    lines.append("## Duplicate removal statistics\n\n")
    lines.append("| Stage | Duplicates removed | Result |\n")
    lines.append("|-------|-------------------:|--------|\n")
    lines.append(
        f"| Retraining merge (synthetic + AAEC + IBM) | {RETRAIN_DUPES_REMOVED} | {MERGED_CORPUS} unique rows |\n"
    )
    lines.append("| Early merge (synthetic + real-world v1) | 3,589 | 811 unique rows (`reports/dataset_statistics.md`) |\n")
    lines.append(f"| IBM ArgKP extraction | {ibm_raw - ibm_clean} | {ibm_clean} unique arguments |\n")
    lines.append("| AAEC extraction | 39 | 6,050 unique components |\n")
    lines.append("| Overlap with test_set.csv | 0 | No literal leakage |\n")

    SUMMARY_OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {SUMMARY_OUT}")


def write_train_test_split() -> None:
    test_n = _count(TEST_SET)
    test_df = pd.read_csv(TEST_SET)
    train_n, val_n = _train_val_sizes()

    lines: list[str] = []
    lines.append("# Train / Validation / Test Split Documentation\n\n")
    lines.append(
        "How InferAI partitions data for training and evaluation. Canonical merged size "
        f"**{MERGED_CORPUS}**. Generated by `reports/generate_professor_documentation.py`.\n\n"
    )

    lines.append("## Overview\n\n")
    lines.append("| Split | Rows | Fraction | Source |\n")
    lines.append("|-------|-----:|----------|--------|\n")
    lines.append(f"| Train | {train_n} | 80% | `merged_retraining_corpus.csv` |\n")
    lines.append(f"| Validation | {val_n} | 20% | `merged_retraining_corpus.csv` |\n")
    lines.append(f"| Held-out test | {test_n} | — | `dataset/test_set.csv` (separate file) |\n\n")

    lines.append("## Train split\n\n")
    lines.append(
        f"Created by `sklearn.model_selection.train_test_split` with `test_size=0.2` and "
        f"`random_state={TRAIN_VAL_RANDOM_STATE}`. **{train_n}** rows from the merged "
        f"retraining corpus (n = {MERGED_CORPUS}) are used for classifier fitting.\n\n"
    )

    lines.append("## Validation split\n\n")
    lines.append(
        f"The remaining **{val_n}** rows (20%) form a random holdout used for in-training evaluation "
        "(`reports/final_retraining_report.md`).\n\n"
    )

    lines.append("## Held-out split\n\n")
    lines.append(
        f"`dataset/test_set.csv` contains **{test_n}** rows built by `dataset/labeled_corpus.py`. "
        "It is never concatenated into the merged training corpus.\n\n"
    )

    vc = test_df["pramana_label"].value_counts().reindex(LABELS, fill_value=0).astype(int)
    tbl = pd.DataFrame({"Pramana": vc.index, "Count": vc.values})
    lines.append(f"### Held-out test — class counts (n = {test_n})\n\n")
    lines.append(dataframe_to_markdown(tbl) + "\n\n")

    lines.append("## Random seed\n\n")
    lines.append(f"- **Train/validation partition:** `random_state={TRAIN_VAL_RANDOM_STATE}`\n")
    lines.append("- **Merge shuffle:** `random_state=42` in `classification/retrain_merged.py`\n\n")

    lines.append("## Why leakage cannot occur\n\n")
    lines.append(
        "1. **Text-level exclusion** of `test_set.csv` rows before merge.\n"
        "2. **Separate held-out file** loaded only at evaluation.\n"
        "3. **Zero exact-text overlap** with AAEC/IBM reviewed sources (`reports/test_set_distribution_analysis.md`).\n"
        "4. **Exact duplicate audit:** 0 cross-split exact duplicates (`reports/data_leakage_report.md`).\n"
    )

    SPLIT_OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {SPLIT_OUT}")


def main() -> None:
    ensure_reports_dir()
    if not TEST_SET.is_file():
        raise FileNotFoundError(f"Missing required file: {TEST_SET}")
    write_provenance()
    write_dataset_summary()
    write_train_test_split()


if __name__ == "__main__":
    main()
