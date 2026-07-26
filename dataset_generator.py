"""
Build and merge NyayaXAI / InferAI training corpora.

Primary pipeline (``main``)
---------------------------
1. Load ``dataset/raw/nyaya_dataset.csv`` and ``dataset/real_world_dataset.csv``.
2. Normalize to ``text``, ``pramana_label``, ``strength_label``.
3. Concatenate and drop duplicate ``text`` rows (real-world rows kept on ties).
4. Remove any example whose text appears in ``dataset/test_set.csv`` (no leakage).
5. Save ``dataset/raw/nyaya_dataset_merged.csv``.

Legacy helpers for synthetic expansion remain available for ad-hoc regeneration
of ``nyaya_dataset.csv`` via ``build_legacy_synthetic_dataset()``.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

from dataset.labeled_corpus import write_csvs

random.seed(42)

ROOT = Path(__file__).resolve().parent

NYAYA_RAW = ROOT / "dataset" / "raw" / "nyaya_dataset.csv"
REAL_WORLD = ROOT / "dataset" / "real_world_dataset.csv"
TEST_SET = ROOT / "dataset" / "test_set.csv"
MERGED_OUT = ROOT / "dataset" / "raw" / "nyaya_dataset_merged.csv"

# -----------------------------------------------------------------------------
# Merge pipeline
# -----------------------------------------------------------------------------


def _norm_text_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _standardize_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map varied inputs to text + pramana_label + strength_label."""
    if "text" not in df.columns:
        raise ValueError(f"{source}: missing 'text' column")

    out = pd.DataFrame()
    out["text"] = df["text"].astype(str).str.strip()

    if "pramana_label" in df.columns:
        out["pramana_label"] = df["pramana_label"].astype(str).str.strip()
    elif "label" in df.columns:
        out["pramana_label"] = df["label"].astype(str).str.strip()
    else:
        raise ValueError(f"{source}: need 'pramana_label' or 'label' column")

    if "strength_label" in df.columns:
        out["strength_label"] = (
            df["strength_label"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
        )
    else:
        out["strength_label"] = pd.NA

    return out


def _load_test_exclude_set() -> set[str]:
    if not TEST_SET.is_file():
        return set()
    test_df = pd.read_csv(TEST_SET)
    return set(_norm_text_key(test_df["text"]))


def merge_training_datasets() -> pd.DataFrame:
    """Merge raw + real-world, exclude test-set texts, deduplicate by text."""
    if not NYAYA_RAW.is_file():
        raise FileNotFoundError(f"Missing {NYAYA_RAW}")
    if not REAL_WORLD.is_file():
        raise FileNotFoundError(
            f"Missing {REAL_WORLD}. Run: python dataset/build_real_world_dataset.py"
        )

    nyaya = _standardize_schema(pd.read_csv(NYAYA_RAW), "nyaya_dataset.csv")
    real = _standardize_schema(pd.read_csv(REAL_WORLD), "real_world_dataset.csv")

    rows_before = len(nyaya) + len(real)
    print(f"Rows before merge (sum of sources): {rows_before}")
    print(f"  nyaya_dataset.csv:      {len(nyaya)}")
    print(f"  real_world_dataset.csv: {len(real)}")

    # Real-world first so deduplication keeps curated strength labels when text overlaps.
    combined = pd.concat([real, nyaya], ignore_index=True)

    exclude = _load_test_exclude_set()
    if exclude:
        mask = ~_norm_text_key(combined["text"]).isin(exclude)
        removed = int((~mask).sum())
        combined = combined.loc[mask].copy()
        print(f"Removed {removed} rows overlapping dataset/test_set.csv (leakage guard)")

    before_dedup = len(combined)
    combined["_text_key"] = _norm_text_key(combined["text"])
    combined = combined.drop_duplicates(subset=["_text_key"], keep="first").drop(
        columns=["_text_key"]
    )
    after_dedup = len(combined)
    print(f"Rows after concatenation (pre-dedup): {before_dedup}")
    print(f"Rows after deduplication:           {after_dedup}")
    print(f"Duplicate texts removed:            {before_dedup - after_dedup}")

    print("\nClass distribution (pramana_label):")
    dist = combined["pramana_label"].value_counts()
    for label, count in dist.items():
        pct = 100.0 * count / len(combined) if len(combined) else 0.0
        print(f"  {label:12s} {count:5d}  ({pct:5.1f}%)")

    if "strength_label" in combined.columns:
        known = combined["strength_label"].notna().sum()
        print(f"\nRows with strength_label: {known} / {len(combined)}")

    return combined


def main() -> None:
    if not NYAYA_RAW.is_file():
        build_legacy_synthetic_dataset()
    merged = merge_training_datasets()
    MERGED_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUT, index=False)
    infer_out = ROOT / "dataset" / "raw" / "infer_dataset.csv"
    merged.to_csv(infer_out, index=False)
    print(f"\nSaved: {MERGED_OUT} ({len(merged)} rows)")
    print(f"Saved: {infer_out} ({len(merged)} rows)")


# -----------------------------------------------------------------------------
# Legacy synthetic builder (optional; not run by default)
# -----------------------------------------------------------------------------

_DOMAINS = [
    "climate modeling",
    "clinical triage",
    "constitutional law",
    "cybersecurity operations",
    "econometrics",
    "history of science",
    "urban planning",
    "school policy",
    "software reliability",
    "astrobiology",
    "journalism ethics",
    "supply chains",
]

_OPENERS = [
    "Taken together, the observations suggest",
    "On balance, the evidence indicates",
    "From what we can verify on the record,",
    "If we treat the premises charitably,",
    "Under ordinary conditions in this domain,",
]

_CONNECTORS = [
    "therefore",
    "hence",
    "so",
    "which implies",
    "which makes it plausible that",
    "which supports the view that",
]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _synthetic_pratyaksha() -> str:
    d = random.choice(_DOMAINS)
    return _clean(
        f"In {d}, the instrument trace shows a repeatable pattern across three runs. "
        f"The operator log notes the same anomaly at the same timestamp window."
    )


def _synthetic_anumana() -> str:
    d = random.choice(_DOMAINS)
    c = random.choice(_CONNECTORS)
    return _clean(
        f"{random.choice(_OPENERS)} a bottleneck in {d}. "
        f"The downstream metric lags whenever the upstream queue saturates; {c} the system is contention-limited."
    )


def _synthetic_upamana() -> str:
    d = random.choice(_DOMAINS)
    return _clean(
        f"Explaining {d} to newcomers is similar to teaching navigation: "
        f"you start from landmarks people already trust, then map unfamiliar terrain onto those anchors."
    )


def _synthetic_shabda() -> str:
    d = random.choice(_DOMAINS)
    return _clean(
        f"According to the published standard for {d}, reviewers must document assumptions and cite the controlling source. "
        f"Experts in the field treat that requirement as a minimum bar for credibility."
    )


def _synthetic_mixed() -> tuple[str, str]:
    d = random.choice(_DOMAINS)
    if random.random() < 0.5:
        text = _clean(
            f"We measured the artifact directly in {d}. "
            f"Still, the manual states a different nominal range, so practitioners defer to the handbook when reporting."
        )
        return text, "Shabda"
    text = _clean(
        f"The dashboard in {d} looked stable at first glance. "
        f"Yet latency spikes only during failover, which suggests a race in the controller path."
    )
    return text, "Anumana"


def build_synthetic_rows(target_count: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    generators = [
        ("Pratyaksha", _synthetic_pratyaksha),
        ("Anumana", _synthetic_anumana),
        ("Upamana", _synthetic_upamana),
        ("Shabda", _synthetic_shabda),
    ]
    while len(rows) < target_count:
        if random.random() < 0.12:
            rows.append(_synthetic_mixed())
            continue
        label, fn = random.choice(generators)
        rows.append((fn(), label))
    return rows[:target_count]


def build_legacy_synthetic_dataset() -> None:
    """Regenerate ``dataset/raw/nyaya_dataset.csv`` from master + synthetic rows."""
    write_csvs(ROOT / "dataset")
    master = pd.read_csv(ROOT / "dataset" / "master_dataset.csv")
    test_df = pd.read_csv(TEST_SET)
    exclude = set(_norm_text_key(test_df["text"]))

    master_rows: list[tuple[str, str]] = []
    for _, r in master.iterrows():
        t = str(r["text"]).strip()
        if _norm_text_key(pd.Series([t])).iloc[0] in exclude:
            continue
        master_rows.append((t, str(r["pramana_label"]).strip()))

    target_total = 4200
    synth_n = max(0, target_total - len(master_rows))
    synth_rows = build_synthetic_rows(synth_n)
    combined = master_rows + synth_rows
    random.shuffle(combined)
    out_rows = [(t, lab) for t, lab in combined if t.strip().lower() not in exclude]
    out = pd.DataFrame(out_rows, columns=["text", "label"])
    out.to_csv(NYAYA_RAW, index=False)
    print(f"Wrote {NYAYA_RAW} with {len(out)} rows")


if __name__ == "__main__":
    main()
