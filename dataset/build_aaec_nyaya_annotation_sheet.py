"""
Build a Nyāya annotation sheet from AAEC extracted components.

Reads ``dataset/processed/aaec_extracted.csv``, removes near-duplicates
(RapidFuzz >= 90), samples 1500 diverse rows across component types and
essays, and assigns heuristic ``candidate_pramana`` labels.

Outputs:
  dataset/processed/aaec_nyaya_annotation_sheet.csv

Usage:
    python dataset/build_aaec_nyaya_annotation_sheet.py
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "aaec_extracted.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "aaec_nyaya_annotation_sheet.csv"

TARGET_TOTAL = 1500
TARGET_PER_TYPE = TARGET_TOTAL // 3  # 500 MajorClaim, 500 Claim, 500 Premise
FUZZ_THRESHOLD = 90
RANDOM_SEED = 42
MAX_PER_ESSAY = 4

COMPONENT_TYPES = ("MajorClaim", "Claim", "Premise")
PRAMANA_LABELS = ("Pratyaksha", "Anumana", "Upamana", "Shabda")
TIE_PRIORITY = ("Shabda", "Anumana", "Upamana", "Pratyaksha")

# User-specified heuristic cues (substring / phrase match, case-insensitive)
HEURISTIC_CUES: dict[str, list[str]] = {
    "Pratyaksha": [
        "observation",
        "measurement",
        "data",
        "records",
        "survey results",
    ],
    "Anumana": [
        "because",
        "therefore",
        "thus",
        "suggests",
        "implies",
        "likely",
        "causes",
    ],
    "Upamana": [
        "like",
        "similar to",
        "analogous to",
        "comparison",
        "example-based reasoning",
    ],
    "Shabda": [
        "according to",
        "experts",
        "researchers",
        "reports",
        "studies",
        "authorities",
    ],
}


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def remove_near_duplicates(df: pd.DataFrame, threshold: int = FUZZ_THRESHOLD) -> pd.DataFrame:
    """Greedy near-duplicate filter using RapidFuzz token-set ratio."""
    kept_rows: list[dict] = []
    kept_texts: list[str] = []

    for row in df.itertuples(index=False):
        text = _normalize_text(row.text)
        if not text:
            continue
        is_dup = any(fuzz.ratio(text, prev) >= threshold for prev in kept_texts)
        if is_dup:
            continue
        kept_rows.append(
            {
                "essay_id": row.essay_id,
                "component_id": getattr(row, "component_id", ""),
                "component_type": row.component_type,
                "text": text,
            }
        )
        kept_texts.append(text)

    return pd.DataFrame(kept_rows)


def _match_cues(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    hits: dict[str, list[str]] = {label: [] for label in PRAMANA_LABELS}
    for label, cues in HEURISTIC_CUES.items():
        for cue in cues:
            if cue.lower() in lowered:
                hits[label].append(cue)
    return hits


def classify_candidate(text: str) -> tuple[str, str]:
    """Return (candidate_pramana, candidate_reason) from heuristic cues."""
    hits = _match_cues(text)
    scores = {label: len(matched) for label, matched in hits.items()}
    best = max(scores.values())

    if best == 0:
        return "Anumana", "No strong cue; default to Anumana (inferential argumentative span)"

    winners = [label for label, score in scores.items() if score == best]
    if len(winners) == 1:
        label = winners[0]
    else:
        label = next(lab for lab in TIE_PRIORITY if lab in winners)

    matched = ", ".join(hits[label])
    reason = f"Matched {label} cues: {matched}"
    return label, reason


def sample_diverse(
    pool: pd.DataFrame,
    target: int,
    *,
    random_state: int = RANDOM_SEED,
    max_per_essay: int = MAX_PER_ESSAY,
) -> pd.DataFrame:
    """
    Sample rows with essay diversity via round-robin across essays.

    Each essay contributes at most ``max_per_essay`` rows.
    """
    if len(pool) <= target:
        return pool.copy()

    rng = random.Random(random_state)
    by_essay: dict[str, list[dict]] = defaultdict(list)
    for row in pool.itertuples(index=False):
        by_essay[row.essay_id].append(
            {
                "essay_id": row.essay_id,
                "component_type": row.component_type,
                "text": row.text,
            }
        )

    essay_ids = list(by_essay.keys())
    rng.shuffle(essay_ids)
    for eid in essay_ids:
        rng.shuffle(by_essay[eid])

    selected: list[dict] = []
    counts: dict[str, int] = defaultdict(int)

    while len(selected) < target:
        made_progress = False
        for eid in essay_ids:
            if len(selected) >= target:
                break
            if counts[eid] >= max_per_essay:
                continue
            if not by_essay[eid]:
                continue
            selected.append(by_essay[eid].pop(0))
            counts[eid] += 1
            made_progress = True
        if not made_progress:
            break

    # Fill remainder if round-robin stalls before target
    if len(selected) < target:
        remaining: list[dict] = []
        for eid in essay_ids:
            for row in by_essay[eid]:
                if counts[eid] < max_per_essay:
                    remaining.append(row)
        rng.shuffle(remaining)
        for row in remaining:
            if len(selected) >= target:
                break
            eid = row["essay_id"]
            if counts[eid] >= max_per_essay:
                continue
            selected.append(row)
            counts[eid] += 1

    return pd.DataFrame(selected[:target])


def build_annotation_sheet(df: pd.DataFrame) -> pd.DataFrame:
    deduped = remove_near_duplicates(df)
    print(f"After near-dedup (fuzz>={FUZZ_THRESHOLD}): {len(deduped)} rows")

    sampled_parts: list[pd.DataFrame] = []
    for comp_type in COMPONENT_TYPES:
        subset = deduped[deduped["component_type"] == comp_type]
        n_available = len(subset)
        n_target = min(TARGET_PER_TYPE, n_available)
        if n_available < TARGET_PER_TYPE:
            print(
                f"Warning: only {n_available} {comp_type} rows available "
                f"(target {TARGET_PER_TYPE})"
            )
        part = sample_diverse(subset, n_target, random_state=RANDOM_SEED + hash(comp_type) % 1000)
        sampled_parts.append(part)
        print(f"  Sampled {len(part)} {comp_type} from {subset['essay_id'].nunique()} essays")

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # Pad to TARGET_TOTAL if any type was short
    if len(sampled) < TARGET_TOTAL:
        chosen_texts = set(sampled["text"])
        remainder = deduped[~deduped["text"].isin(chosen_texts)]
        need = TARGET_TOTAL - len(sampled)
        extra = sample_diverse(remainder, need, random_state=RANDOM_SEED + 99, max_per_essay=2)
        sampled = pd.concat([sampled, extra], ignore_index=True)
        sampled = sampled.drop_duplicates(subset=["text"], keep="first").head(TARGET_TOTAL)

    candidates = sampled["text"].astype(str).map(classify_candidate)
    sampled["candidate_pramana"] = [c[0] for c in candidates]
    sampled["candidate_reason"] = [c[1] for c in candidates]
    sampled["final_pramana"] = ""
    sampled["strength_label"] = ""
    sampled["reviewed"] = ""

    columns = [
        "essay_id",
        "component_type",
        "text",
        "candidate_pramana",
        "candidate_reason",
        "final_pramana",
        "strength_label",
        "reviewed",
    ]
    return sampled[columns]


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Run dataset/parse_aaec.py first.")

    df = pd.read_csv(INPUT_CSV)
    required = {"essay_id", "component_type", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    sheet = build_annotation_sheet(df)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("AAEC Nyaya annotation sheet summary")
    print("=" * 60)
    print(f"Total rows: {len(sheet)}")
    print(f"Unique essays: {sheet['essay_id'].nunique()}")
    print("\nComponent type distribution:")
    print(sheet["component_type"].value_counts().to_string())
    print("\nCandidate pramana distribution:")
    print(sheet["candidate_pramana"].value_counts().reindex(PRAMANA_LABELS, fill_value=0).to_string())
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
