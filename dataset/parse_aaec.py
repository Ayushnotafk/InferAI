"""
Parse AAEC brat-style annotations into a clean component table.

Outputs:
  - dataset/processed/aaec_extracted.csv
  - reports/aaec_statistics.md
  - reports/csv/aaec_component_distribution.csv

Also prints 20 random examples from each component type for verification.

Usage:
    python dataset/parse_aaec.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AAEC_DIR = PROJECT_ROOT / "dataset" / "external" / "AAEC"
PROCESSED_OUT = PROJECT_ROOT / "dataset" / "processed" / "aaec_extracted.csv"
REPORT_OUT = PROJECT_ROOT / "reports" / "aaec_statistics.md"
DIST_OUT = PROJECT_ROOT / "reports" / "csv" / "aaec_component_distribution.csv"

TARGET_COMPONENTS = {"MajorClaim", "Claim", "Premise"}
RANDOM_SEED = 42
SAMPLE_PER_CLASS = 20


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip boundary spaces."""
    return " ".join(str(text).split()).strip()


def _parse_textbound_line(line: str) -> tuple[str, str, str] | None:
    """
    Parse brat text-bound line.

    Expected format:
      T1<TAB>Claim 10 30<TAB>component text
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        return None

    component_id, span_info, text = parts
    if not component_id.startswith("T"):
        return None

    span_tokens = span_info.split()
    if len(span_tokens) < 3:
        return None

    component_type = span_tokens[0]
    if component_type not in TARGET_COMPONENTS:
        return None

    # Validate offsets; supports discontinuous offsets (e.g. "10 20;30 40")
    # by checking numeric fragments after replacing separators.
    offset_tokens = " ".join(span_tokens[1:]).replace(";", " ").split()
    if len(offset_tokens) < 2:
        return None
    for tok in offset_tokens:
        if not tok.isdigit():
            return None

    cleaned = _clean_text(text)
    if not cleaned:
        return None

    return component_id, component_type, cleaned


def _collect_essay_pairs(aaec_dir: Path) -> list[tuple[Path, Path]]:
    txt_files = {p.stem: p for p in aaec_dir.glob("essay*.txt")}
    ann_files = {p.stem: p for p in aaec_dir.glob("essay*.ann")}
    shared = sorted(set(txt_files) & set(ann_files))
    return [(txt_files[stem], ann_files[stem]) for stem in shared]


def parse_aaec() -> tuple[pd.DataFrame, dict[str, int]]:
    if not AAEC_DIR.is_dir():
        raise FileNotFoundError(f"AAEC directory not found: {AAEC_DIR}")

    pairs = _collect_essay_pairs(AAEC_DIR)
    rows: list[dict[str, str]] = []

    malformed_annotations = 0
    total_candidate_annotations = 0

    for _txt_path, ann_path in pairs:
        essay_id = ann_path.stem
        for raw_line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or not line.startswith("T"):
                continue

            total_candidate_annotations += 1
            parsed = _parse_textbound_line(line)
            if parsed is None:
                malformed_annotations += 1
                continue

            component_id, component_type, text = parsed
            rows.append(
                {
                    "essay_id": essay_id,
                    "component_id": component_id,
                    "component_type": component_type,
                    "text": text,
                }
            )

    if not rows:
        raise ValueError("No valid AAEC components were extracted.")

    df_raw = pd.DataFrame(rows)
    before_clean = len(df_raw)

    # Remove duplicate texts globally as requested.
    df_clean = (
        df_raw[df_raw["text"].astype(str).str.strip() != ""]
        .drop_duplicates(subset=["text"], keep="first")
        .reset_index(drop=True)
    )

    removed_duplicates_or_empty = before_clean - len(df_clean)

    stats = {
        "total_essays": len(pairs),
        "total_components": len(df_clean),
        "majorclaim_count": int((df_clean["component_type"] == "MajorClaim").sum()),
        "claim_count": int((df_clean["component_type"] == "Claim").sum()),
        "premise_count": int((df_clean["component_type"] == "Premise").sum()),
        "unique_text_count": int(df_clean["text"].nunique()),
        "candidate_textbound_annotations": total_candidate_annotations,
        "malformed_annotations_removed": malformed_annotations,
        "duplicates_or_empty_removed": removed_duplicates_or_empty,
    }
    return df_clean, stats


def write_outputs(df: pd.DataFrame, stats: dict[str, int]) -> None:
    PROCESSED_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DIST_OUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_OUT, index=False, encoding="utf-8")

    dist = (
        df["component_type"]
        .value_counts()
        .reindex(["MajorClaim", "Claim", "Premise"], fill_value=0)
        .rename_axis("component_type")
        .reset_index(name="count")
    )
    dist.to_csv(DIST_OUT, index=False, encoding="utf-8")

    lines: list[str] = []
    lines.append("# AAEC Corpus Statistics\n\n")
    lines.append("Automated extraction summary for `dataset/external/AAEC`.\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- Total essays: **{stats['total_essays']}**\n")
    lines.append(f"- Total components: **{stats['total_components']}**\n")
    lines.append(f"- MajorClaim count: **{stats['majorclaim_count']}**\n")
    lines.append(f"- Claim count: **{stats['claim_count']}**\n")
    lines.append(f"- Premise count: **{stats['premise_count']}**\n")
    lines.append(f"- Unique text count: **{stats['unique_text_count']}**\n")
    lines.append("\n## Cleaning details\n\n")
    lines.append(
        f"- Candidate text-bound annotations scanned: **{stats['candidate_textbound_annotations']}**\n"
    )
    lines.append(
        f"- Malformed annotations removed: **{stats['malformed_annotations_removed']}**\n"
    )
    lines.append(
        f"- Duplicate/empty rows removed: **{stats['duplicates_or_empty_removed']}**\n"
    )
    lines.append("\n## Class distribution CSV\n\n")
    lines.append(f"- `{DIST_OUT.relative_to(PROJECT_ROOT).as_posix()}`\n")
    REPORT_OUT.write_text("".join(lines), encoding="utf-8")


def print_samples(df: pd.DataFrame) -> None:
    random.seed(RANDOM_SEED)
    print("\n" + "=" * 72)
    print("AAEC verification samples (20 random per component type)")
    print("=" * 72)

    for comp_type in ["MajorClaim", "Claim", "Premise"]:
        subset = df[df["component_type"] == comp_type]
        n = min(SAMPLE_PER_CLASS, len(subset))
        sampled = subset.sample(n=n, random_state=RANDOM_SEED) if n > 0 else subset
        print(f"\n[{comp_type}] showing {n} samples")
        for _, row in sampled.iterrows():
            text = row["text"]
            print(f"- {row['essay_id']},{row['component_id']}: {text}")


def main() -> None:
    df, stats = parse_aaec()
    write_outputs(df, stats)
    print_samples(df)
    print("\nSaved:")
    print(f"- {PROCESSED_OUT}")
    print(f"- {REPORT_OUT}")
    print(f"- {DIST_OUT}")


if __name__ == "__main__":
    main()
