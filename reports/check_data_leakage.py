"""
Data leakage and duplicate audit across train, validation, and held-out test splits.

Uses exact text matching, RapidFuzz string similarity (>90%), and cosine similarity
on Sentence-BERT embeddings (>0.90).

Outputs:
  - reports/data_leakage_report.md
  - reports/csv/data_leakage_summary.csv
  - reports/csv/data_leakage_suspicious_examples.csv

Does NOT delete duplicates — reports only.

Usage:
    python reports/check_data_leakage.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings
from dataset_generator import _norm_text_key
from reports._common import dataframe_to_markdown, ensure_reports_dir, save_dataframe_csv

MERGED = _ROOT / "dataset" / "raw" / "nyaya_dataset_merged.csv"
TEST_SET = _ROOT / "dataset" / "test_set.csv"

FUZZ_THRESHOLD = 90
COSINE_THRESHOLD = 0.90
MAX_NEAR_EXAMPLES = 25


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_splits() -> dict[str, pd.DataFrame]:
    if not MERGED.is_file():
        raise FileNotFoundError(f"Missing {MERGED}. Run dataset_generator.py first.")
    if not TEST_SET.is_file():
        raise FileNotFoundError(f"Missing {TEST_SET}")

    merged = pd.read_csv(MERGED)
    indices = list(range(len(merged)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_df = merged.iloc[train_idx][["text"]].copy().reset_index(drop=True)
    train_df["split"] = "train"
    val_df = merged.iloc[val_idx][["text"]].copy().reset_index(drop=True)
    val_df["split"] = "validation"

    test_df = pd.read_csv(TEST_SET)[["text"]].copy()
    test_df["split"] = "held_out_test"

    return {"train": train_df, "validation": val_df, "held_out_test": test_df}


def _exact_duplicates(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> list[dict]:
    keys_a = _norm_text_key(df_a["text"])
    keys_b = _norm_text_key(df_b["text"])
    set_b = set(keys_b)
    examples: list[dict] = []
    for i, key in enumerate(keys_a):
        if key in set_b:
            j = int(keys_b[keys_b == key].index[0])
            examples.append(
                {
                    "pair_type": "exact_duplicate",
                    "split_a": name_a,
                    "split_b": name_b,
                    "text_a": df_a.iloc[i]["text"][:200],
                    "text_b": df_b.loc[j, "text"][:200],
                    "fuzz_ratio": 100.0,
                    "cosine_similarity": None,
                }
            )
    return examples


def _near_duplicates(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    emb_a: np.ndarray | None = None,
    emb_b: np.ndarray | None = None,
) -> list[dict]:
    """Find near duplicates via RapidFuzz and embedding cosine similarity."""
    examples: list[dict] = []
    texts_a = df_a["text"].astype(str).tolist()
    texts_b = df_b["text"].astype(str).tolist()
    keys_a = _norm_text_key(df_a["text"]).tolist()
    keys_b = set(_norm_text_key(df_b["text"]).tolist())

    for i, (ta, ka) in enumerate(zip(texts_a, keys_a)):
        if ka in keys_b:
            continue  # exact dup handled separately
        for j, tb in enumerate(texts_b):
            ratio = float(fuzz.ratio(ta, tb))
            cos_sim = None
            if emb_a is not None and emb_b is not None:
                cos_sim = _cosine_sim(emb_a[i], emb_b[j])
            is_near = ratio >= FUZZ_THRESHOLD or (
                cos_sim is not None and cos_sim >= COSINE_THRESHOLD
            )
            if is_near:
                examples.append(
                    {
                        "pair_type": "near_duplicate",
                        "split_a": name_a,
                        "split_b": name_b,
                        "text_a": ta[:200],
                        "text_b": tb[:200],
                        "fuzz_ratio": round(ratio, 2),
                        "cosine_similarity": round(cos_sim, 4) if cos_sim is not None else None,
                    }
                )
    # Deduplicate symmetric near pairs (keep highest score per text_a)
    examples.sort(key=lambda x: (x["fuzz_ratio"], x["cosine_similarity"] or 0), reverse=True)
    seen_a: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        key = ex["text_a"]
        if key in seen_a:
            continue
        seen_a.add(key)
        unique.append(ex)
    return unique


def main() -> None:
    ensure_reports_dir()
    splits = _load_splits()

    exact_all: list[dict] = []
    near_all: list[dict] = []

    pair_names = list(combinations(splits.keys(), 2))
    for name_a, name_b in pair_names:
        df_a, df_b = splits[name_a], splits[name_b]
        exact_all.extend(_exact_duplicates(df_a, df_b, name_a, name_b))

    # Embedding-based near-duplicate scan (train vs test and val vs test are highest risk).
    for name_a, name_b in [("train", "held_out_test"), ("validation", "held_out_test"), ("train", "validation")]:
        df_a, df_b = splits[name_a], splits[name_b]
        print(f"Embedding near-duplicate scan: {name_a} vs {name_b} ...")
        emb_a = generate_embeddings(df_a["text"].astype(str).tolist())
        emb_b = generate_embeddings(df_b["text"].astype(str).tolist())
        near_all.extend(_near_duplicates(df_a, df_b, name_a, name_b, emb_a, emb_b))

    # Remove near pairs that are exact duplicates
    exact_keys = {(e["text_a"], e["text_b"]) for e in exact_all}
    near_all = [n for n in near_all if (n["text_a"], n["text_b"]) not in exact_keys]

    summary_df = pd.DataFrame(
        [
            {"Pair Type": "Exact duplicates (cross-split)", "Count": len(exact_all)},
            {"Pair Type": f"Near duplicates (fuzz≥{FUZZ_THRESHOLD}% or cosine≥{COSINE_THRESHOLD})", "Count": len(near_all)},
        ]
    )
    save_dataframe_csv(summary_df, _ROOT / "reports" / "csv" / "data_leakage_summary.csv")

    suspicious = (exact_all + near_all)[:MAX_NEAR_EXAMPLES]
    if suspicious:
        susp_df = pd.DataFrame(suspicious)
        save_dataframe_csv(susp_df, _ROOT / "reports" / "csv" / "data_leakage_suspicious_examples.csv")

    lines: list[str] = []
    lines.append("# InferAI — Data Leakage and Duplicate Audit\n\n")
    lines.append(
        "Cross-split duplicate detection between training (80%), validation (20%), and "
        "held-out test data. **No rows were removed** — this is an audit-only report.\n\n"
    )

    lines.append("## Methodology\n\n")
    lines.append(
        f"- **Exact duplicates:** normalized lowercase stripped text equality.\n"
        f"- **Near duplicates:** RapidFuzz `ratio` ≥ {FUZZ_THRESHOLD}% **or** cosine similarity "
        f"≥ {COSINE_THRESHOLD} on `all-MiniLM-L6-v2` embeddings.\n"
        "- **Splits:** merged training table partitioned with `train_test_split(test_size=0.2, random_state=42)`; "
        "`test_set.csv` is fully held out during merge.\n\n"
    )

    lines.append("## Summary\n\n")
    lines.append(dataframe_to_markdown(summary_df) + "\n\n")

    lines.append("## Split sizes\n\n")
    sizes = pd.DataFrame(
        [{"Split": k, "Rows": len(v)} for k, v in splits.items()]
    )
    lines.append(dataframe_to_markdown(sizes) + "\n\n")

    if exact_all:
        lines.append("## Exact duplicate examples\n\n")
        for i, ex in enumerate(exact_all[:10], 1):
            lines.append(
                f"{i}. **{ex['split_a']} ↔ {ex['split_b']}**  \n"
                f"   - Text: \"{ex['text_a']}\"\n"
            )
    else:
        lines.append(
            "## Exact duplicates\n\n"
            "No exact text duplicates were found across train, validation, and held-out test splits. "
            "The merge pipeline's test-set exclusion guard appears effective.\n\n"
        )

    if near_all:
        lines.append(f"\n## Near-duplicate examples (showing up to {MAX_NEAR_EXAMPLES})\n\n")
        for i, ex in enumerate(near_all[:MAX_NEAR_EXAMPLES], 1):
            cos = ex["cosine_similarity"]
            cos_str = f"{cos:.3f}" if cos is not None else "n/a"
            lines.append(
                f"{i}. **{ex['split_a']} ↔ {ex['split_b']}** "
                f"(fuzz={ex['fuzz_ratio']:.1f}%, cosine={cos_str})  \n"
                f"   - A: \"{ex['text_a']}…\"  \n"
                f"   - B: \"{ex['text_b']}…\"\n"
            )
    else:
        lines.append(
            "\n## Near duplicates\n\n"
            "No near duplicates above the configured thresholds were detected across splits.\n"
        )

    lines.append("\n## Interpretation\n\n")
    if len(exact_all) == 0 and len(near_all) == 0:
        lines.append(
            "The held-out test set is **textually disjoint** from the merged training pool at "
            "both exact and near-duplicate thresholds, supporting the claim that reported "
            "held-out metrics are not inflated by verbatim leakage.\n"
        )
    else:
        lines.append(
            "Some cross-split similarity was detected. Review listed examples manually; "
            "near duplicates may reflect paraphrase rather than true leakage but can still "
            "inflate validation optimism if semantically identical arguments appear in both "
            "training and test corpora.\n"
        )

    out = _ROOT / "reports" / "data_leakage_report.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
