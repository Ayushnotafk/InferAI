"""
Add group-level provenance keys to all dataset sources so downstream
splits can be made group-disjoint (reviewer issues C-03, M-15, MISS-2).

Outputs
-------
dataset/processed/aaec_reviewed_with_groups.csv
    Adds: essay_id  (from aaec_extracted.csv join on text)
dataset/processed/ibm_reviewed_with_groups.csv
    Adds: topic_group  (from ibm_argkp_extracted.csv join on text)
dataset/raw/nyaya_dataset_with_groups.csv
    Adds: template_family  (extracted from text prefix like "Bench notes (science):")
dataset/processed/merged_corpus_with_groups.csv
    Union of all three sources; also writes split_manifests/

Usage (from project root):
    python scripts/add_group_keys.py
"""

from __future__ import annotations

import re
import sys
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "dataset" / "processed"
RAW = ROOT / "dataset" / "raw"
MANIFESTS = ROOT / "dataset" / "split_manifests"
MANIFESTS.mkdir(parents=True, exist_ok=True)

AAEC_REVIEWED = PROCESSED / "aaec_reviewed_dataset.csv"
AAEC_EXTRACTED = PROCESSED / "aaec_extracted.csv"
IBM_REVIEWED = PROCESSED / "ibm_reviewed_dataset.csv"
IBM_EXTRACTED = PROCESSED / "ibm_argkp_extracted.csv"

SYNTHETIC_CANDIDATES = [
    RAW / "nyaya_dataset.csv",
    ROOT / "dataset" / "raw" / "nyaya_dataset.csv",
]

OUT_AAEC = PROCESSED / "aaec_reviewed_with_groups.csv"
OUT_IBM = PROCESSED / "ibm_reviewed_with_groups.csv"
OUT_SYNTHETIC = RAW / "nyaya_dataset_with_groups.csv"
OUT_MERGED = PROCESSED / "merged_corpus_with_groups.csv"


def _norm(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


# ---------------------------------------------------------------------------
# AAEC — join on normalized text to recover essay_id
# ---------------------------------------------------------------------------

def enrich_aaec() -> pd.DataFrame:
    reviewed = pd.read_csv(AAEC_REVIEWED)
    extracted = pd.read_csv(AAEC_EXTRACTED)

    extracted["_norm"] = extracted["text"].astype(str).map(_norm)
    reviewed["_norm"] = reviewed["text"].astype(str).map(_norm)

    lookup = extracted.drop_duplicates("_norm").set_index("_norm")["essay_id"]
    reviewed["essay_id"] = reviewed["_norm"].map(lookup)

    # For rows that don't match exactly (paraphrased during review), assign
    # a stable synthetic essay_id based on a hash of the normalized text so
    # every row still has a non-null group key.
    missing_mask = reviewed["essay_id"].isna()
    reviewed.loc[missing_mask, "essay_id"] = (
        reviewed.loc[missing_mask, "_norm"]
        .apply(lambda t: "aaec_anon_" + hashlib.md5(t.encode()).hexdigest()[:8])
    )

    matched = int((~missing_mask).sum())
    total = len(reviewed)
    print(f"AAEC: {matched}/{total} rows matched to essay_id by text join.")

    reviewed = reviewed.drop(columns=["_norm"])
    reviewed["source"] = "aaec_reviewed"
    reviewed["topic_group"] = ""
    reviewed["template_family"] = ""
    reviewed.to_csv(OUT_AAEC, index=False)
    print(f"  Saved: {OUT_AAEC.relative_to(ROOT)}")
    return reviewed


# ---------------------------------------------------------------------------
# IBM ArgKP — join on normalized text to recover topic (=group key)
# ---------------------------------------------------------------------------

def enrich_ibm() -> pd.DataFrame:
    reviewed = pd.read_csv(IBM_REVIEWED)
    extracted = pd.read_csv(IBM_EXTRACTED)

    extracted["_norm"] = extracted["text"].astype(str).map(_norm)
    reviewed["_norm"] = reviewed["text"].astype(str).map(_norm)

    lookup = extracted.drop_duplicates("_norm").set_index("_norm")["topic"]
    reviewed["topic_group"] = reviewed["_norm"].map(lookup)

    missing_mask = reviewed["topic_group"].isna()
    reviewed.loc[missing_mask, "topic_group"] = (
        reviewed.loc[missing_mask, "_norm"]
        .apply(lambda t: "ibm_anon_" + hashlib.md5(t.encode()).hexdigest()[:8])
    )

    matched = int((~missing_mask).sum())
    total = len(reviewed)
    print(f"IBM: {matched}/{total} rows matched to topic_group by text join.")

    reviewed = reviewed.drop(columns=["_norm"])
    reviewed["source"] = "ibm_reviewed"
    reviewed["essay_id"] = ""
    reviewed["template_family"] = ""
    reviewed.to_csv(OUT_IBM, index=False)
    print(f"  Saved: {OUT_IBM.relative_to(ROOT)}")
    return reviewed


# ---------------------------------------------------------------------------
# Synthetic — extract template_family from text prefix patterns
# Patterns: "Bench notes (science): ...", "Test augmentation (education): ..."
# Also: "In constitutional law, ...", domain parentheticals
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?P<explicit_tag>[A-Z][^:(]{2,40})\s*\((?P<domain>[^)]{2,40})\)\s*:"  # "Bench notes (science):"
    r"|In (?P<in_domain>[a-z][^,]{2,40}),?"                                   # "In constitutional law,"
    r")",
    re.IGNORECASE,
)
_AUGMENT_RE = re.compile(r"synthetic vignette uid (test|master)-pad-", re.IGNORECASE)


def _extract_template_family(text: str) -> str:
    m = _PREFIX_RE.match(str(text).strip())
    if m:
        tag = m.group("explicit_tag") or ""
        dom = m.group("domain") or m.group("in_domain") or ""
        if tag:
            # Normalise tag: "Bench notes" → "bench_notes"
            tag_key = re.sub(r"\s+", "_", tag.strip().lower())
            dom_key = re.sub(r"\s+", "_", dom.strip().lower())
            return f"{tag_key}__{dom_key}"
        elif dom:
            return "in_domain__" + re.sub(r"\s+", "_", dom.strip().lower())

    if _AUGMENT_RE.search(text):
        return "generic_pad_augmentation"

    return "no_template_family"


def enrich_synthetic() -> pd.DataFrame:
    src = None
    for candidate in SYNTHETIC_CANDIDATES:
        if candidate.is_file():
            src = candidate
            break
    if src is None:
        raise FileNotFoundError("Synthetic dataset not found.")

    df = pd.read_csv(src)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    df = df.rename(columns={label_col: "pramana_label"})

    df["template_family"] = df["text"].astype(str).map(_extract_template_family)
    df["source"] = "nyaya_synthetic"
    df["essay_id"] = ""
    df["topic_group"] = ""

    if "strength_label" not in df.columns:
        df["strength_label"] = ""

    df.to_csv(OUT_SYNTHETIC, index=False)
    fam_counts = df["template_family"].value_counts()
    print(f"Synthetic: {len(df)} rows, {len(fam_counts)} template families.")
    print(f"  Top families:\n{fam_counts.head(10).to_string()}")
    print(f"  Saved: {OUT_SYNTHETIC.relative_to(ROOT)}")
    return df


# ---------------------------------------------------------------------------
# Merge all three and write the group-aware corpus
# ---------------------------------------------------------------------------

def build_merged(aaec: pd.DataFrame, ibm: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    KEEP = ["text", "pramana_label", "strength_label", "source",
            "essay_id", "topic_group", "template_family"]

    parts = []
    for name, df in [("aaec", aaec), ("ibm", ibm), ("synthetic", synthetic)]:
        df2 = df.copy()
        for col in KEEP:
            if col not in df2.columns:
                df2[col] = ""
        parts.append(df2[KEEP])

    merged = pd.concat(parts, ignore_index=True)

    # Deduplicate on normalized text
    merged["_norm"] = merged["text"].astype(str).map(_norm)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["_norm"], keep="first").reset_index(drop=True)
    merged = merged.drop(columns=["_norm"])
    after = len(merged)
    print(f"\nMerged corpus: {before} rows → {after} after dedup ({before-after} removed)")

    merged.to_csv(OUT_MERGED, index=False, encoding="utf-8")
    print(f"  Saved: {OUT_MERGED.relative_to(ROOT)}")

    # Write manifest: group key summary
    manifest = merged.groupby("source").agg(
        rows=("text", "count"),
        unique_groups=("essay_id", lambda x: x.replace("", pd.NA).dropna().nunique() if x.replace("", pd.NA).dropna().any() else merged.loc[x.index, "topic_group"].replace("", pd.NA).dropna().nunique())
    ).reset_index()
    manifest.to_csv(MANIFESTS / "source_manifest.csv", index=False)
    print(f"  Manifest: {(MANIFESTS / 'source_manifest.csv').relative_to(ROOT)}")

    # Class distribution
    print("\nClass distribution (merged with groups):")
    print(merged["pramana_label"].value_counts().to_string())
    print("\nGroup key coverage:")
    print(f"  essay_id filled:      {(merged['essay_id'] != '').sum()}/{len(merged)}")
    print(f"  topic_group filled:   {(merged['topic_group'] != '').sum()}/{len(merged)}")
    print(f"  template_family filled: {(merged['template_family'] != '').sum()}/{len(merged)}")

    return merged


def main() -> None:
    print("=" * 70)
    print("Step 1: Enrich AAEC reviewed dataset with essay_id")
    print("=" * 70)
    aaec = enrich_aaec()

    print("\n" + "=" * 70)
    print("Step 2: Enrich IBM reviewed dataset with topic_group")
    print("=" * 70)
    ibm = enrich_ibm()

    print("\n" + "=" * 70)
    print("Step 3: Enrich synthetic dataset with template_family")
    print("=" * 70)
    synthetic = enrich_synthetic()

    print("\n" + "=" * 70)
    print("Step 4: Merge into group-aware corpus")
    print("=" * 70)
    build_merged(aaec, ibm, synthetic)

    print("\n✅ Group keys added. Next step: run scripts/build_clean_splits.py")


if __name__ == "__main__":
    main()
