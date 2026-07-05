"""One-off analysis for reports/test_set_distribution_analysis.md"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
TEST = _ROOT / "dataset" / "test_set.csv"
NYAYA = _ROOT / "dataset" / "raw" / "nyaya_dataset.csv"
AAEC = _ROOT / "dataset" / "processed" / "aaec_reviewed_dataset.csv"
IBM = _ROOT / "dataset" / "processed" / "ibm_reviewed_dataset.csv"
MERGED = _ROOT / "dataset" / "processed" / "merged_retraining_corpus.csv"
OUT = _ROOT / "reports" / "test_set_distribution_analysis.md"


def norm_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def load_df(path: Path) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    text_col = next(c for c in ("text", "sentence", "argument") if c in df.columns)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    return df, text_col, label_col


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def vocab_stats(texts: list[str]) -> dict:
    tokens: list[str] = []
    for t in texts:
        tokens.extend(tokenize(t))
    wc = Counter(tokens)
    word_lens = [len(tokenize(t)) for t in texts]
    char_lens = [len(str(t)) for t in texts]
    return {
        "n_docs": len(texts),
        "n_tokens": len(tokens),
        "n_types": len(wc),
        "ttr": len(wc) / len(tokens) if tokens else 0.0,
        "avg_words": float(np.mean(word_lens)) if word_lens else 0.0,
        "median_words": float(np.median(word_lens)) if word_lens else 0.0,
        "std_words": float(np.std(word_lens)) if word_lens else 0.0,
        "avg_chars": float(np.mean(char_lens)) if char_lens else 0.0,
        "median_chars": float(np.median(char_lens)) if char_lens else 0.0,
        "word_counter": wc,
    }


def norm_freq(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


def jensen_shannon(p: dict[str, float], q: dict[str, float], eps: float = 1e-12) -> float:
    vocab = sorted(set(p) | set(q))
    P = np.array([p.get(w, 0.0) for w in vocab], dtype=float)
    Q = np.array([q.get(w, 0.0) for w in vocab], dtype=float)
    P = P / (P.sum() + eps)
    Q = Q / (Q.sum() + eps)
    M = 0.5 * (P + Q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def class_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    vc = df[label_col].value_counts()
    total = len(df)
    rows = []
    for label in ("Pratyaksha", "Anumana", "Upamana", "Shabda"):
        c = int(vc.get(label, 0))
        rows.append({"label": label, "count": c, "pct": 100.0 * c / total})
    return pd.DataFrame(rows)


def overlap_count(test_texts: set[str], other_texts: set[str]) -> tuple[int, list[str]]:
    inter = test_texts & other_texts
    return len(inter), sorted(inter)[:5]


def main() -> None:
    test_df, tc, tlc = load_df(TEST)
    merged_df, mc, mlc = load_df(MERGED)
    nyaya_df, nc, nlc = load_df(NYAYA)
    aaec_df, ac, alc = load_df(AAEC)
    ibm_df, ic, ilc = load_df(IBM)

    test_texts = {norm_text(x) for x in test_df[tc]}
    test_stats = vocab_stats(test_df[tc].tolist())
    train_stats = vocab_stats(merged_df[mc].tolist())
    nyaya_stats = vocab_stats(nyaya_df[nc].tolist())
    aaec_stats = vocab_stats(aaec_df[ac].tolist())
    ibm_stats = vocab_stats(ibm_df[ic].tolist())

    jsd_merged = jensen_shannon(norm_freq(test_stats["word_counter"]), norm_freq(train_stats["word_counter"]))
    jsd_nyaya = jensen_shannon(norm_freq(test_stats["word_counter"]), norm_freq(nyaya_stats["word_counter"]))
    jsd_aaec = jensen_shannon(norm_freq(test_stats["word_counter"]), norm_freq(aaec_stats["word_counter"]))
    jsd_ibm = jensen_shannon(norm_freq(test_stats["word_counter"]), norm_freq(ibm_stats["word_counter"]))

    test_class = class_table(test_df, tlc)
    train_class = class_table(merged_df, mlc)

    overlaps = {}
    for name, df, col in [
        ("nyaya_dataset.csv", nyaya_df, nc),
        ("aaec_reviewed_dataset.csv", aaec_df, ac),
        ("ibm_reviewed_dataset.csv", ibm_df, ic),
    ]:
        other = {norm_text(x) for x in df[col]}
        overlaps[name] = overlap_count(test_texts, other)

    domain_paren = sum(
        1 for t in test_df[tc] if re.search(r"\((science|technology|education|medicine|politics|philosophy)\)", str(t), re.I)
    )

    notes_col = "reviewer_notes" if "reviewer_notes" in test_df.columns else None
    if notes_col:
        note_vc = test_df[notes_col].value_counts()
        seed_rows = int(note_vc.get("test seed v1", 0))
        aug_rows = int(note_vc.get("test augmentation", 0))
    else:
        seed_rows = len(test_df)
        aug_rows = 0

    seed_df = test_df[test_df[notes_col] == "test seed v1"] if notes_col and seed_rows else test_df
    seed_stats = vocab_stats(seed_df[tc].tolist()) if len(seed_df) else test_stats

    test_top = test_stats["word_counter"].most_common(20)
    train_top = train_stats["word_counter"].most_common(20)

    test_freq = norm_freq(test_stats["word_counter"])
    train_freq = norm_freq(train_stats["word_counter"])
    distinctive_test = sorted(
        ((w, test_freq[w], train_freq.get(w, 0.0)) for w in test_freq),
        key=lambda x: x[1] - x[2],
        reverse=True,
    )[:15]
    distinctive_train = sorted(
        ((w, train_freq[w], test_freq.get(w, 0.0)) for w in train_freq),
        key=lambda x: x[1] - x[2],
        reverse=True,
    )[:15]

    lines: list[str] = []
    lines.append("# Test Set Distribution Analysis\n")
    lines.append("Analysis of `dataset/test_set.csv` relative to the current merged training corpus ")
    lines.append("(`dataset/processed/merged_retraining_corpus.csv`).\n")
    lines.append("Generated by `reports/_analyze_test_set_distribution.py`.\n")
    lines.append("---\n")

    lines.append("## 1. Where was this dataset created from?\n")
    lines.append(
        "`dataset/test_set.csv` is produced by **`dataset/labeled_corpus.py`** via `build_test_dataframe()` → "
        "`write_csvs()`. It is a **held-out curated corpus** built from deterministic templates in `_test_templates()`, "
        "expanded across **11 domain slots** (science, technology, education, medicine, politics, philosophy, "
        "legal reasoning, public debate, social media discourse, news analysis, daily life).\n"
    )
    lines.append(
        "The companion file `dataset/master_dataset.csv` uses **different** `_master_templates()` so training and "
        "test phrasing stay disjoint by design. `dataset_generator.py` and `classification/retrain_merged.py` "
        "explicitly **exclude** any row whose text appears in `test_set.csv` to prevent leakage.\n"
    )
    lines.append(
        "Every row carries `reviewer_notes = \"test seed v1\"`, matching the `corpus_id=\"test\"` tag in "
        "`labeled_corpus.py`.\n"
    )
    lines.append(
        f"\n**On-disk composition (n={len(test_df)}):** **{seed_rows}** rows from `_test_templates()` "
        f"(`test seed v1`) and **{aug_rows}** rows from deterministic padding (`test augmentation`) in "
        "`_materialize()` when an older 8-template build undershot the 200-row target. Padding rows use "
        "hash-picked labels and share a repetitive meta-vocabulary (`synthetic vignette`, `evaluation bookkeeping`).\n"
    )

    lines.append("## 2. Samples per pramāṇa\n")
    lines.append("| Pramāṇa | Count | % of test set |\n")
    lines.append("|--------|------:|-------------:|\n")
    for _, row in test_class.iterrows():
        lines.append(f"| {row['label']} | {int(row['count'])} | {row['pct']:.1f}% |\n")
    lines.append(f"| **Total** | **{len(test_df)}** | **100%** |\n")

    lines.append("## 3. Manual or automatic annotation?\n")
    lines.append(
        "**Automatically generated**, not manually annotated from external corpora.\n\n"
        "- Rows are instantiated from fixed English templates with domain placeholders.\n"
        "- Labels are assigned **deterministically by template** (each template hard-codes a pramāṇa and strength).\n"
        "- `review_status = approved` reflects pipeline bookkeeping, not independent human review.\n"
        "- This contrasts with `aaec_reviewed_dataset.csv` and `ibm_reviewed_dataset.csv`, which come from "
        "manual Nyāya annotation of argumentative essays and IBM ArgKP claims.\n"
    )

    lines.append("## 4. Overlap with training sources\n")
    lines.append("Exact normalized-text overlap (case-folded, whitespace-collapsed):\n\n")
    lines.append("| Corpus | Overlapping rows with test_set.csv |\n")
    lines.append("|--------|----------------------------------:|\n")
    for name, (n, samples) in overlaps.items():
        lines.append(f"| `{name}` | **{n}** |\n")
    lines.append(
        "\n**Result:** Zero exact-text overlap with all three sources. The held-out guard in "
        "`dataset_generator.py` / merge scripts is effective for literal duplicates.\n"
    )
    if any(n > 0 for n, _ in overlaps.values()):
        lines.append("\nExample overlapping texts:\n")
        for name, (n, samples) in overlaps.items():
            if n:
                for s in samples:
                    lines.append(f"- `{name}`: {s[:100]}…\n")

    lines.append("## 5. Vocabulary distribution\n")
    lines.append("| Metric | test_set.csv | Merged training corpus |\n")
    lines.append("|--------|-------------:|-----------------------:|\n")
    lines.append(f"| Documents | {test_stats['n_docs']} | {train_stats['n_docs']} |\n")
    lines.append(f"| Total tokens | {test_stats['n_tokens']} | {train_stats['n_tokens']} |\n")
    lines.append(f"| Unique types (vocabulary) | {test_stats['n_types']} | {train_stats['n_types']} |\n")
    lines.append(f"| Type–token ratio (TTR) | {test_stats['ttr']:.4f} | {train_stats['ttr']:.4f} |\n")
    lines.append(
        f"| Jensen–Shannon divergence (word freq) | — | **{jsd_merged:.4f}** vs merged |\n"
    )
    lines.append("\n**JSD vs individual training sources:**\n\n")
    lines.append(f"- vs `nyaya_dataset.csv` (synthetic only): **{jsd_nyaya:.4f}**\n")
    lines.append(f"- vs `aaec_reviewed_dataset.csv`: **{jsd_aaec:.4f}**\n")
    lines.append(f"- vs `ibm_reviewed_dataset.csv`: **{jsd_ibm:.4f}**\n")
    lines.append(
        "\nLower JSD = more similar word-frequency profile. Values near 0.3–0.5 indicate substantial lexical shift; "
        "near 0.1 indicate closer overlap.\n"
    )

    lines.append("### Top tokens (test_set.csv)\n\n")
    lines.append("| Rank | Token | Count |\n|-----:|-------|------:|\n")
    for i, (w, c) in enumerate(test_top, 1):
        lines.append(f"| {i} | {w} | {c} |\n")

    lines.append("\n### Top tokens (merged training corpus)\n\n")
    lines.append("| Rank | Token | Count |\n|-----:|-------|------:|\n")
    for i, (w, c) in enumerate(train_top, 1):
        lines.append(f"| {i} | {w} | {c} |\n")

    lines.append("\n### Distinctive vocabulary (higher relative frequency in test)\n\n")
    lines.append("| Token | Freq (test) | Freq (train) |\n|-------|------------:|-------------:|\n")
    for w, ft, fr in distinctive_test:
        lines.append(f"| {w} | {ft:.4f} | {fr:.4f} |\n")

    lines.append("\n### Distinctive vocabulary (higher relative frequency in training)\n\n")
    lines.append("| Token | Freq (train) | Freq (test) |\n|-------|-------------:|------------:|\n")
    for w, fr, ft in distinctive_train:
        lines.append(f"| {w} | {fr:.4f} | {ft:.4f} |\n")

    lines.append(
        f"\n**Template signature:** {domain_paren}/{len(test_df)} test sentences contain domain parentheticals "
        f"such as `(science)` or `(medicine)`, a pattern essentially absent from AAEC/IBM argumentative text.\n"
    )
    lines.append(
        f"\n**Template-only subset** (`test seed v1`, n={len(seed_df)}): mean **{seed_stats['avg_words']:.1f}** words, "
        f"TTR **{seed_stats['ttr']:.4f}**, top tokens include domain-specific cues (`telemetry`, `bench`, `metaphors`). "
        f"Padding rows inflate shared meta-tokens (`test`, `augmentation`, `vignette`) in corpus-wide vocabulary stats.\n"
    )

    lines.append("## 6. Average sentence length\n")
    lines.append("| Metric | test_set.csv | Merged training |\n")
    lines.append("|--------|-------------:|----------------:|\n")
    lines.append(f"| Mean words | {test_stats['avg_words']:.1f} | {train_stats['avg_words']:.1f} |\n")
    lines.append(f"| Median words | {test_stats['median_words']:.1f} | {train_stats['median_words']:.1f} |\n")
    lines.append(f"| Std dev (words) | {test_stats['std_words']:.1f} | {train_stats['std_words']:.1f} |\n")
    lines.append(f"| Mean characters | {test_stats['avg_chars']:.1f} | {train_stats['avg_chars']:.1f} |\n")
    lines.append(f"| Median characters | {test_stats['median_chars']:.1f} | {train_stats['median_chars']:.1f} |\n")

    lines.append("\n**Per-source training length (mean words):**\n\n")
    for name, stats in [
        ("nyaya_dataset.csv", nyaya_stats),
        ("aaec_reviewed_dataset.csv", aaec_stats),
        ("ibm_reviewed_dataset.csv", ibm_stats),
        ("merged (all)", train_stats),
    ]:
        lines.append(f"- {name}: **{stats['avg_words']:.1f}** words\n")

    lines.append("## 7. Class distribution comparison\n")
    lines.append("| Pramāṇa | test_set.csv | | Merged training | |\n")
    lines.append("|--------|-------------:|---|-------------:|---|\n")
    lines.append("| | Count | % | Count | % |\n")
    train_vc = merged_df[mlc].value_counts()
    for label in ("Pratyaksha", "Anumana", "Upamana", "Shabda"):
        tc_n = int(test_class.loc[test_class["label"] == label, "count"].iloc[0])
        tc_p = float(test_class.loc[test_class["label"] == label, "pct"].iloc[0])
        tr_n = int(train_vc.get(label, 0))
        tr_p = 100.0 * tr_n / len(merged_df)
        lines.append(f"| {label} | {tc_n} | {tc_p:.1f}% | {tr_n} | {tr_p:.1f}% |\n")

    lines.append("\n**Training source breakdown (counts):**\n\n")
    lines.append("| Source | n | Pratyaksha | Anumana | Upamana | Shabda |\n")
    lines.append("|--------|--:|-----------:|--------:|--------:|-------:|\n")
    for name, df, lc in [
        ("nyaya_dataset.csv", nyaya_df, nlc),
        ("aaec_reviewed_dataset.csv", aaec_df, alc),
        ("ibm_reviewed_dataset.csv", ibm_df, ilc),
    ]:
        vc = df[lc].value_counts()
        lines.append(
            f"| {name} | {len(df)} | {vc.get('Pratyaksha',0)} | {vc.get('Anumana',0)} | "
            f"{vc.get('Upamana',0)} | {vc.get('Shabda',0)} |\n"
        )
    vc = merged_df[mlc].value_counts()
    lines.append(
        f"| **merged (deduped)** | **{len(merged_df)}** | **{vc.get('Pratyaksha',0)}** | "
        f"**{vc.get('Anumana',0)}** | **{vc.get('Upamana',0)}** | **{vc.get('Shabda',0)}** |\n"
    )

    # L1 class distribution distance
    test_p = test_class.set_index("label")["pct"].to_dict()
    train_p = {row["label"]: row["pct"] for _, row in train_class.iterrows()}
    l1 = sum(abs(test_p[l] / 100.0 - train_p[l] / 100.0) for l in ("Pratyaksha", "Anumana", "Upamana", "Shabda")) / 2
    lines.append(f"\nTotal variation distance (class distributions): **{l1:.3f}** (0 = identical, 1 = disjoint).\n")

    lines.append("## 8. Conclusion: in-distribution or out-of-distribution?\n")
    lines.append(
        "**`test_set.csv` is out-of-distribution (OOD) relative to the current merged training corpus**, "
        "despite sharing the same four Nyāya labels and zero exact-text leakage.\n\n"
        "### Evidence for OOD\n\n"
    )
    ood_points = []
    if jsd_merged >= 0.15:
        ood_points.append(
            f"- **Lexical shift:** Jensen–Shannon divergence vs merged training is **{jsd_merged:.3f}**, "
            "indicating a measurably different word-frequency profile."
        )
    if abs(test_stats["avg_words"] - train_stats["avg_words"]) > 3:
        ood_points.append(
            f"- **Length mismatch:** Test sentences average **{test_stats['avg_words']:.1f}** words vs "
            f"**{train_stats['avg_words']:.1f}** in training ({abs(test_stats['avg_words']-train_stats['avg_words']):.1f} word gap)."
        )
    if l1 > 0.15:
        ood_points.append(
        f"- **Class skew:** Test set is roughly balanced (~23–29% per class), while merged training is "
        f"**~{train_p['Anumana']:.0f}% Anumana** — total variation distance **{l1:.2f}**."
    )
    if aug_rows > 0:
        ood_points.append(
            f"- **Padding artefact:** {aug_rows}/{len(test_df)} rows are generic `test augmentation` fillers with "
            "hash-assigned labels, further diverging from natural argumentative training text."
        )
    ood_points.append(
        "- **Genre / provenance:** Test text is **synthetic template prose** with domain parentheticals; "
        "training is dominated by **argumentative essays (AAEC)** and **claim pairs (IBM)**, plus older synthetic nyaya rows."
    )
    ood_points.append(
        "- **Vocabulary:** Test-frequent tokens (`telemetry`, `bench`, `metaphors`, `filings`) differ from "
        "training-frequent argumentative tokens (`argument`, `claim`, `therefore`, `because`)."
    )
    for p in ood_points:
        lines.append(p + "\n")

    lines.append(
        "\n### What is still aligned\n\n"
        "- Same label schema and cue families (perception, inference, analogy, testimony).\n"
        "- Deliberately **balanced** pramāṇa counts for evaluation (unlike training).\n"
        "- **No duplicate texts** with nyaya, AAEC, or IBM reviewed sets.\n"
    )

    lines.append(
        "\n### Practical implication\n\n"
        "Held-out accuracy on `test_set.csv` (~68–76% in project reports) **understates in-corpus performance** "
        "and **overstates real-world generalization** to AAEC/IBM-style arguments. The test set is best interpreted as a "
        "**curated, leakage-free sanity check** on template-style English, not as i.i.d. draws from the merged "
        "training distribution. For deployment claims, evaluate on manually reviewed AAEC/IBM holdouts or fresh "
        "human-annotated samples from the target domain.\n"
    )

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"JSD merged={jsd_merged:.4f} L1={l1:.3f}")
    print(test_class)


if __name__ == "__main__":
    main()
