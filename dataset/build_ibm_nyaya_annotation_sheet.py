"""
Build a Nyaya manual annotation sheet from IBM ArgKP extracted arguments.

Reads ``dataset/processed/ibm_argkp_extracted.csv``, performs diverse
stratified sampling (topics, stance, argument length), removes near-duplicates
(RapidFuzz >= 90), and assigns heuristic candidate pramana labels.

Outputs:
  - dataset/processed/ibm_nyaya_annotation_sheet.csv
  - reports/ibm_annotation_sheet_stats.md

Usage:
    python dataset/build_ibm_nyaya_annotation_sheet.py
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "ibm_argkp_extracted.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "ibm_nyaya_annotation_sheet.csv"
REPORT_OUT = PROJECT_ROOT / "reports" / "ibm_annotation_sheet_stats.md"

TARGET_TOTAL = 600
FUZZ_THRESHOLD = 90
RANDOM_SEED = 42
SAMPLE_PRINT = 20

PRAMANA_LABELS = ("Pratyaksha", "Anumana", "Upamana", "Shabda")
TIE_PRIORITY = ("Shabda", "Anumana", "Upamana", "Pratyaksha")

HEURISTIC_CUES: dict[str, list[str]] = {
    "Pratyaksha": [
        "observation",
        "measured values",
        "statistics",
        "survey",
        "experiment",
        "evidence",
        "data",
        "record",
        "witness",
        "observed",
    ],
    "Shabda": [
        "according to",
        "expert",
        "researcher",
        "report",
        "study",
        "journal",
        "government",
        "authority",
        "evidence from",
    ],
    "Upamana": [
        "like",
        "similar to",
        "analogy",
        "analogous",
        "compared with",
        "as if",
        "resembles",
    ],
    "Anumana": [
        "because",
        "therefore",
        "thus",
        "hence",
        "implies",
        "suggests",
        "consequently",
        "likely",
        "causes",
    ],
}

IF_THEN_RE = re.compile(r"\bif\b.{1,120}?\bthen\b", re.IGNORECASE | re.DOTALL)
WHO_RE = re.compile(r"\bWHO\b")

OUTPUT_COLUMNS = [
    "source",
    "topic",
    "stance",
    "text",
    "candidate_pramana",
    "candidate_reason",
    "final_pramana",
    "strength_label",
    "reviewed",
]


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _length_bucket(length: int, p33: float, p66: float) -> str:
    if length <= p33:
        return "short"
    if length <= p66:
        return "medium"
    return "long"


def _match_cues(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    hits: dict[str, list[str]] = {label: [] for label in PRAMANA_LABELS}
    for label, cues in HEURISTIC_CUES.items():
        for cue in cues:
            if cue.lower() in lowered:
                hits[label].append(cue)
    if IF_THEN_RE.search(text):
        hits["Anumana"].append("if...then reasoning")
    if WHO_RE.search(text):
        hits["Shabda"].append("WHO")
    return hits


def classify_candidate(text: str) -> tuple[str, str]:
    hits = _match_cues(text)
    scores = {label: len(matched) for label, matched in hits.items()}
    best = max(scores.values())

    if best == 0:
        return "Anumana", "No heuristic cue matched; default to Anumana (inferential argumentative span)"

    winners = [label for label, score in scores.items() if score == best]
    if len(winners) == 1:
        label = winners[0]
    else:
        label = next(lab for lab in TIE_PRIORITY if lab in winners)

    matched = ", ".join(hits[label])
    return label, f"Matched {label} cues: {matched}"


def _is_near_duplicate(text: str, selected_texts: list[str], threshold: int) -> bool:
    return any(fuzz.ratio(text, prev) >= threshold for prev in selected_texts)


def _topic_quotas(topics: list[str], target: int) -> dict[str, int]:
    base, extra = divmod(target, len(topics))
    shuffled = topics[:]
    random.Random(RANDOM_SEED).shuffle(shuffled)
    return {topic: base + (1 if i < extra else 0) for i, topic in enumerate(shuffled)}


def _pick_row(
    pool: pd.DataFrame,
    selected_texts: list[str],
    stance_counts: dict[str, int],
    length_counts: dict[str, int],
    rng: random.Random,
) -> tuple[pd.Series | None, int]:
    """Return (best row, near_duplicate_skips)."""
    if pool.empty:
        return None, 0

    prefer_stance = "support" if stance_counts["support"] <= stance_counts["oppose"] else "oppose"
    prefer_len = min(length_counts, key=lambda k: length_counts[k]) if length_counts else "medium"

    def score_row(row: pd.Series) -> tuple[int, int, float]:
        stance_score = 1 if row["stance"] == prefer_stance else 0
        len_score = 1 if row["length_bucket"] == prefer_len else 0
        return (stance_score, len_score, rng.random())

    ranked = pool.copy()
    ranked["_score"] = ranked.apply(score_row, axis=1)
    ranked = ranked.sort_values("_score", ascending=False)

    near_dup_skips = 0
    for _, row in ranked.iterrows():
        if _is_near_duplicate(row["text"], selected_texts, FUZZ_THRESHOLD):
            near_dup_skips += 1
            continue
        return row, near_dup_skips
    return None, near_dup_skips


def sample_diverse(df: pd.DataFrame, target: int) -> tuple[pd.DataFrame, int]:
    rng = random.Random(RANDOM_SEED)
    lengths = df["text"].str.len()
    p33 = float(lengths.quantile(0.33))
    p66 = float(lengths.quantile(0.66))

    work = df.copy()
    work["text"] = work["text"].astype(str).map(_normalize_text)
    work = work[work["text"].str.len() > 0].reset_index(drop=True)
    work["text_len"] = work["text"].str.len()
    work["length_bucket"] = work["text_len"].apply(lambda x: _length_bucket(x, p33, p66))

    topics = sorted(work["topic"].unique().tolist())
    quotas = _topic_quotas(topics, target)

    selected_rows: list[dict] = []
    selected_texts: list[str] = []
    stance_counts: dict[str, int] = defaultdict(int)
    length_counts: dict[str, int] = defaultdict(int)
    near_dup_removed = 0

    topic_pools: dict[str, pd.DataFrame] = {}
    for topic in topics:
        pool = work[work["topic"] == topic].sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        topic_pools[topic] = pool

    for topic, quota in sorted(quotas.items(), key=lambda x: -x[1]):
        pool = topic_pools[topic]
        picked = 0
        while picked < quota and len(selected_rows) < target:
            if pool.empty:
                break
            row, skips = _pick_row(pool, selected_texts, stance_counts, length_counts, rng)
            near_dup_removed += skips
            if row is None:
                break
            selected_rows.append(row.to_dict())
            selected_texts.append(row["text"])
            stance_counts[row["stance"]] += 1
            length_counts[row["length_bucket"]] += 1
            pool = pool.drop(row.name, errors="ignore")
            picked += 1

    # Fill remaining slots if some topics lacked enough unique rows
    if len(selected_rows) < target:
        remaining = work[~work["text"].isin(selected_texts)].sample(frac=1.0, random_state=RANDOM_SEED + 1)
        for _, row in remaining.iterrows():
            if len(selected_rows) >= target:
                break
            if _is_near_duplicate(row["text"], selected_texts, FUZZ_THRESHOLD):
                near_dup_removed += 1
                continue
            selected_rows.append(row.to_dict())
            selected_texts.append(row["text"])
            stance_counts[row["stance"]] += 1
            length_counts[row["length_bucket"]] += 1

    if len(selected_rows) < target:
        raise ValueError(
            f"Could only sample {len(selected_rows)} rows after near-dedup; target was {target}."
        )

    out = pd.DataFrame(selected_rows[:target])
    return out, near_dup_removed


def build_annotation_sheet(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    sampled, near_dup_removed = sample_diverse(df, TARGET_TOTAL)

    out = pd.DataFrame()
    out["source"] = sampled["source"]
    out["topic"] = sampled["topic"]
    out["stance"] = sampled["stance"]
    out["text"] = sampled["text"]

    classified = out["text"].astype(str).map(classify_candidate)
    out["candidate_pramana"] = [c[0] for c in classified]
    out["candidate_reason"] = [c[1] for c in classified]
    out["final_pramana"] = ""
    out["strength_label"] = ""
    out["reviewed"] = ""

    return out[OUTPUT_COLUMNS], near_dup_removed


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, near_dup_removed: int) -> None:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    total = len(df)
    avg_len = float(df["text"].str.len().mean())

    stance = df["stance"].value_counts().reset_index()
    stance.columns = ["stance", "count"]

    pramana = (
        df["candidate_pramana"]
        .value_counts()
        .reindex(PRAMANA_LABELS, fill_value=0)
        .rename_axis("candidate_pramana")
        .reset_index(name="count")
    )
    pramana["percentage"] = (100.0 * pramana["count"] / total).round(2)

    lines: list[str] = []
    lines.append("# IBM ArgKP Nyaya Annotation Sheet — Statistics\n\n")
    lines.append(
        "Generated from `ibm_argkp_extracted.csv` with stratified sampling "
        f"(target={TARGET_TOTAL}, random_state={RANDOM_SEED}).\n\n"
    )
    lines.append("## Summary\n\n")
    lines.append(f"- **Sampled rows:** {total}\n")
    lines.append(f"- **Unique topics:** {df['topic'].nunique()}\n")
    lines.append(f"- **Average argument length (chars):** {avg_len:.1f}\n")
    lines.append(f"- **Near-duplicates removed during sampling:** {near_dup_removed}\n\n")

    lines.append("## Stance distribution\n\n")
    lines.append(_md_table(["stance", "count"], stance.values.tolist()) + "\n\n")

    lines.append("## candidate_pramana distribution\n\n")
    lines.append(
        _md_table(
            ["candidate_pramana", "count", "percentage"],
            pramana[["candidate_pramana", "count", "percentage"]].values.tolist(),
        )
        + "\n\n"
    )

    lines.append("## Length buckets (sampled)\n\n")
    p33 = df["text"].str.len().quantile(0.33)
    p66 = df["text"].str.len().quantile(0.66)
    buckets = df["text"].str.len().apply(lambda x: _length_bucket(x, p33, p66)).value_counts()
    lines.append(_md_table(["length_bucket", "count"], [[k, int(v)] for k, v in buckets.items()]) + "\n\n")

    lines.append("## Output columns\n\n")
    lines.append("```text\n")
    lines.append(", ".join(OUTPUT_COLUMNS) + "\n")
    lines.append("```\n")

    REPORT_OUT.write_text("".join(lines), encoding="utf-8")


def print_samples(df: pd.DataFrame, n: int = SAMPLE_PRINT) -> None:
    k = min(n, len(df))
    sample = df.sample(n=k, random_state=RANDOM_SEED)
    print("\n" + "=" * 72)
    print(f"Verification sample ({k} random rows)")
    print("=" * 72)
    for i, row in enumerate(sample.itertuples(index=False), 1):
        text = row.text if len(row.text) <= 220 else row.text[:220] + "…"
        print(f"\n{i}. topic: {row.topic}")
        print(f"   stance: {row.stance}")
        print(f"   candidate_pramana: {row.candidate_pramana}")
        print(f"   reason: {row.candidate_reason}")
        print(f"   text: {text}")


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing {INPUT_CSV}. Run dataset/parse_ibm_argkp.py first.")

    raw = pd.read_csv(INPUT_CSV)
    sheet, near_dup_removed = build_annotation_sheet(raw)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    write_report(sheet, near_dup_removed)
    print_samples(sheet)

    print("\n" + "=" * 72)
    print("IBM Nyaya annotation sheet complete")
    print("=" * 72)
    print(f"Sampled rows: {len(sheet)}")
    print(f"Unique topics: {sheet['topic'].nunique()}")
    print(f"Near-duplicates removed: {near_dup_removed}")
    print("\nStance distribution:")
    print(sheet["stance"].value_counts().to_string())
    print("\ncandidate_pramana distribution:")
    print(sheet["candidate_pramana"].value_counts().reindex(PRAMANA_LABELS, fill_value=0).to_string())
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()
