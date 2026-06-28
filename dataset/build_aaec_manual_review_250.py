"""
Build a 250-row manual review sheet from the AAEC Nyaya annotation sheet.

Selection:
  - All non-Anumana rows (Upamana, Shabda, Pratyaksha)
  - Random Anumana sample to reach 250 total

Outputs:
  - dataset/processed/aaec_manual_review_250.csv
  - reports/aaec_manual_review_250_stats.md

Usage:
    python dataset/build_aaec_manual_review_250.py
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

INPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "aaec_nyaya_annotation_sheet.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "aaec_manual_review_250.csv"
STATS_OUT = PROJECT_ROOT / "reports" / "aaec_manual_review_250_stats.md"

TARGET_TOTAL = 250
RANDOM_SEED = 42

OUTPUT_COLUMNS = [
    "essay_id",
    "component_type",
    "text",
    "candidate_pramana",
    "final_pramana",
    "strength_label",
    "reviewed",
]


def build_manual_review(df: pd.DataFrame) -> pd.DataFrame:
    non_anumana = df[df["candidate_pramana"] != "Anumana"].copy()
    anumana_pool = df[df["candidate_pramana"] == "Anumana"]

    n_non = len(non_anumana)
    n_anumana_needed = TARGET_TOTAL - n_non
    if n_anumana_needed < 0:
        raise ValueError(
            f"Non-Anumana rows ({n_non}) exceed target total ({TARGET_TOTAL})."
        )
    if n_anumana_needed > len(anumana_pool):
        raise ValueError(
            f"Need {n_anumana_needed} Anumana rows but only {len(anumana_pool)} available."
        )

    anumana_sample = anumana_pool.sample(n=n_anumana_needed, random_state=RANDOM_SEED)
    combined = pd.concat([non_anumana, anumana_sample], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    out = pd.DataFrame()
    out["essay_id"] = combined["essay_id"]
    out["component_type"] = combined["component_type"]
    out["text"] = combined["text"]
    out["candidate_pramana"] = combined["candidate_pramana"]
    out["final_pramana"] = combined["candidate_pramana"]
    out["strength_label"] = ""
    out["reviewed"] = ""

    return out[OUTPUT_COLUMNS]


def write_stats(df_source: pd.DataFrame, review: pd.DataFrame) -> None:
    ensure_reports_dir()
    total = len(review)

    final_dist = (
        review["final_pramana"]
        .value_counts()
        .reindex(PRAMANA_LABELS, fill_value=0)
        .rename_axis("final_pramana")
        .reset_index(name="count")
    )
    final_dist["percentage"] = (100.0 * final_dist["count"] / total).round(2)

    comp_dist = review["component_type"].value_counts().reset_index()
    comp_dist.columns = ["component_type", "count"]

    cross = pd.crosstab(review["component_type"], review["final_pramana"], margins=True)
    cross = cross.reindex(columns=[*PRAMANA_LABELS, "All"], fill_value=0)

    lines: list[str] = []
    lines.append("# AAEC Manual Review 250 — Summary Statistics\n\n")
    lines.append(
        f"Built from `aaec_nyaya_annotation_sheet.csv`. Total rows: **{total}**.\n\n"
    )

    lines.append("## Selection\n\n")
    lines.append(
        f"- Non-Anumana rows (all included): **{len(review[review['candidate_pramana'] != 'Anumana'])}**\n"
    )
    lines.append(
        f"- Anumana rows (random sample): **{len(review[review['candidate_pramana'] == 'Anumana'])}**\n"
    )
    lines.append(f"- Unique essays: **{review['essay_id'].nunique()}**\n")
    lines.append(f"- Random seed: **{RANDOM_SEED}**\n\n")

    lines.append("## final_pramana distribution\n\n")
    lines.append(dataframe_to_markdown(final_dist) + "\n\n")

    lines.append("## component_type distribution\n\n")
    lines.append(dataframe_to_markdown(comp_dist) + "\n\n")

    lines.append("## component_type × final_pramana\n\n")
    lines.append(dataframe_to_markdown(cross.reset_index()) + "\n\n")

    lines.append("## Labeling status\n\n")
    lines.append("- `final_pramana` initialized from `candidate_pramana`\n")
    lines.append("- `strength_label` and `reviewed` left blank for manual annotation\n")

    STATS_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    review = build_manual_review(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    write_stats(df, review)

    print("AAEC manual review 250 — summary")
    print("=" * 50)
    print(f"Total rows: {len(review)}")
    print(f"Unique essays: {review['essay_id'].nunique()}")
    print(f"\nNon-Anumana included: {len(review[review['candidate_pramana'] != 'Anumana'])}")
    print(f"Anumana sampled: {len(review[review['candidate_pramana'] == 'Anumana'])}")
    print("\nfinal_pramana distribution:")
    print(review["final_pramana"].value_counts().reindex(PRAMANA_LABELS, fill_value=0).to_string())
    print("\ncomponent_type distribution:")
    print(review["component_type"].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {STATS_OUT}")


if __name__ == "__main__":
    main()
