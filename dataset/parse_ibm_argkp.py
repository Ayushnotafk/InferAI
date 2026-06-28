"""
Parse IBM ArgKP combined corpus for Nyaya-style annotation.

Reads ``dataset/external/IBM_ARGKP/ArgKP_combined.csv``, inspects columns
automatically, extracts debate arguments with topic and stance, and writes
a cleaned export.

Outputs:
  - dataset/processed/ibm_argkp_extracted.csv
  - reports/ibm_argkp_statistics.md

Usage (from project root):
    python dataset/parse_ibm_argkp.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "dataset" / "external" / "IBM_ARGKP" / "ArgKP_combined.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "processed" / "ibm_argkp_extracted.csv"
REPORT_OUT = PROJECT_ROOT / "reports" / "ibm_argkp_statistics.md"

SOURCE_NAME = "IBM_ArgKP"
RANDOM_SEED = 42
SAMPLE_N = 20

# Column candidates for auto-mapping (first match wins).
TEXT_COLUMNS = ("argument", "text", "claim", "sentence")
TOPIC_COLUMNS = ("topic", "debate_topic", "motion")
STANCE_COLUMNS = ("stance", "position", "label_stance")


def _normalize_text(text: str) -> str:
    """Collapse internal whitespace and strip edges."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _inspect_dataset(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("IBM ArgKP — automatic dataset inspection")
    print("=" * 72)
    print(f"File: {INPUT_CSV}")
    print(f"Total rows (raw): {len(df)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print("\nNull counts:")
    print(df.isnull().sum().to_string())
    print("\nSample values (first non-null per column):")
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(1).tolist()
        preview = sample[0][:120] + ("…" if sample and len(sample[0]) > 120 else "")
        print(f"  {col}: {preview!r}")
    print()


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _map_stance(value) -> str:
    """Map numeric or string stance to support / oppose when possible."""
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"support", "pro", "for", "1", "+1"}:
            return "support"
        if s in {"oppose", "con", "against", "-1"}:
            return "oppose"
        return s
    try:
        num = int(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if num == 1:
        return "support"
    if num == -1:
        return "oppose"
    return str(num)


def extract_argkp(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    text_col = _pick_column(df, TEXT_COLUMNS)
    topic_col = _pick_column(df, TOPIC_COLUMNS)
    stance_col = _pick_column(df, STANCE_COLUMNS)

    if not text_col:
        raise ValueError(
            f"Could not find argument text column. Available: {list(df.columns)}"
        )
    if not topic_col:
        raise ValueError(
            f"Could not find topic column. Available: {list(df.columns)}"
        )

    print("Selected columns for Nyaya export:")
    print(f"  topic  <- {topic_col}")
    print(f"  text   <- {text_col}")
    print(f"  stance <- {stance_col or '(not found)'}")
    print()

    raw_total = len(df)
    missing_mask = (
        df[text_col].isna()
        | df[topic_col].isna()
        | (df[text_col].astype(str).str.strip() == "")
        | (df[topic_col].astype(str).str.strip() == "")
    )
    missing_count = int(missing_mask.sum())

    work = df.loc[~missing_mask, [topic_col, text_col] + ([stance_col] if stance_col else [])].copy()
    work["source"] = SOURCE_NAME
    work["topic"] = work[topic_col].astype(str).map(_normalize_text)
    work["text"] = work[text_col].astype(str).map(_normalize_text)
    if stance_col:
        work["stance"] = work[stance_col].map(_map_stance)
    else:
        work["stance"] = ""

    # Drop whitespace-only after normalization
    empty_text = work["text"].str.len() == 0
    empty_topic = work["topic"].str.len() == 0
    whitespace_only = int((empty_text | empty_topic).sum())
    work = work.loc[~(empty_text | empty_topic)].copy()

    before_dedup = len(work)
    work = work.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(work)

    out = work[["source", "topic", "text", "stance"]].copy()

    stats = {
        "raw_total": raw_total,
        "missing_count": missing_count,
        "whitespace_only_removed": whitespace_only,
        "rows_after_cleaning": len(out),
        "duplicates_removed": duplicates_removed,
        "unique_topics": int(out["topic"].nunique()),
        "avg_argument_length": float(out["text"].str.len().mean()) if len(out) else 0.0,
    }
    return out, stats


def _top_topics(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    top = (
        df["topic"]
        .value_counts()
        .head(n)
        .rename_axis("topic")
        .reset_index(name="count")
    )
    return top


def write_report(df: pd.DataFrame, stats: dict) -> None:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    top = _top_topics(df, 20)

    lines: list[str] = []
    lines.append("# IBM ArgKP — Extraction Statistics\n\n")
    lines.append(f"Source file: `{INPUT_CSV.relative_to(PROJECT_ROOT).as_posix()}`\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- **Total rows (raw):** {stats['raw_total']}\n")
    lines.append(f"- **Rows after cleaning:** {stats['rows_after_cleaning']}\n")
    lines.append(f"- **Duplicate arguments removed:** {stats['duplicates_removed']}\n")
    lines.append(f"- **Missing / empty topic or argument rows skipped:** {stats['missing_count']}\n")
    lines.append(
        f"- **Whitespace-only rows removed (post-normalize):** {stats['whitespace_only_removed']}\n"
    )
    lines.append(f"- **Unique topics:** {stats['unique_topics']}\n")
    lines.append(
        f"- **Average argument length (chars):** {stats['avg_argument_length']:.1f}\n\n"
    )

    lines.append("## Stance distribution\n\n")
    stance_dist = df["stance"].value_counts().reset_index()
    stance_dist.columns = ["stance", "count"]
    lines.append("| stance | count |\n")
    lines.append("| --- | ---: |\n")
    for _, row in stance_dist.iterrows():
        lines.append(f"| {row['stance']} | {int(row['count'])} |\n")
    lines.append("\n")

    lines.append("## Top 20 topics\n\n")
    lines.append("| topic | count |\n")
    lines.append("| --- | ---: |\n")
    for _, row in top.iterrows():
        topic = str(row["topic"]).replace("|", "\\|")
        lines.append(f"| {topic} | {int(row['count'])} |\n")
    lines.append("\n")

    lines.append("## Output schema\n\n")
    lines.append("```text\n")
    lines.append("source, topic, text, stance\n")
    lines.append("```\n")

    REPORT_OUT.write_text("".join(lines), encoding="utf-8")


def print_random_examples(df: pd.DataFrame, n: int = SAMPLE_N) -> None:
    random.seed(RANDOM_SEED)
    k = min(n, len(df))
    sample = df.sample(n=k, random_state=RANDOM_SEED)
    print("=" * 72)
    print(f"Random sample ({k} arguments) for manual inspection")
    print("=" * 72)
    for i, row in enumerate(sample.itertuples(index=False), 1):
        text = row.text
        if len(text) > 220:
            text = text[:220] + "…"
        print(f"\n{i}. topic: {row.topic}")
        print(f"   stance: {row.stance}")
        print(f"   text: {text}")


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    _inspect_dataset(df)

    extracted, stats = extract_argkp(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    extracted.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    write_report(extracted, stats)
    print_random_examples(extracted)

    print("\n" + "=" * 72)
    print("Extraction complete")
    print("=" * 72)
    print(f"Rows after cleaning: {stats['rows_after_cleaning']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Unique topics: {stats['unique_topics']}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()
