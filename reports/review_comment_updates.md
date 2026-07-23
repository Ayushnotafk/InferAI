# Professor Review — Comment Updates

Paste the sections below into the professor review response document.

---

**Date:** 7 July 2026

## Comment 1 — Dataset provenance and transparency

**Status:** ✅ Addressed

**Work Completed**

We implemented full dataset provenance and summary documentation drawn from on-disk corpora and existing extraction reports. Each source (synthetic Nyaya, AAEC, IBM ArgKP) is now documented with raw, cleaned, reviewed, and final-used counts. The retraining merge pipeline (`classification/retrain_merged.py`) is described with deduplication statistics (3,589 duplicates removed; 1,861 unique training rows). Train/validation/test partitioning is documented with per-split class counts and explicit confirmation that `test_set.csv` is excluded from training.

**Files created**

- `docs/dataset_provenance.md`
- `reports/dataset_summary.md`
- `reports/train_test_split.md`
- `reports/generate_professor_documentation.py` (reproducible generator)

**Existing reports referenced**

- `reports/dataset_statistics.md`
- `reports/final_retraining_report.md`
- `reports/test_set_distribution_analysis.md`
- `reports/data_leakage_report.md`

---

## Comment 6 — Lexical diversity metrics

**Status:** ✅ Addressed

**Work Completed**

We generated a lexical diversity analysis comparing the synthetic corpus (`nyaya_dataset.csv`, n = 4,200) and the curated real-world corpus (`real_world_dataset.csv`, n = 200). Raw Type–Token Ratio (TTR) alone was shown to be misleading across corpora of different sizes (synthetic TTR = 0.003367 vs real-world TTR = 0.518248). Following professor feedback, we added **Guiraud's R** (0.782816 vs 18.696500) and **MATTR-50** (0.797221 vs 0.834957) as length-aware complementary indices. Figures and interpretation are documented in the lexical diversity report.

**Metrics reported**

| Metric | Synthetic | Real-world |
|--------|----------:|-----------:|
| TTR | 0.003367 | 0.518248 |
| Guiraud's R | 0.782816 | 18.696500 |
| MATTR (w=50) | 0.797221 | 0.834957 |

**Files**

- `reports/lexical_diversity_report.md`
- `reports/figures/lexical_diversity_metrics.png`
- `reports/figures/lexical_diversity_token_vocab.png`
- `reports/generate_lexical_diversity.py`

---

## Comment 7 — Annotation agreement and disagreement analysis

**Status:** ✅ Addressed

**Work Completed**

We extended the inter-annotator agreement report without altering the original summary statistics. The existing results remain: **91.50%** agreement, Cohen's κ = **0.8804**, bootstrap 95% CI **[0.8212, 0.9295]** on 200 matched examples. We appended an **Extended Annotation Analysis** section with per-class agreement, per-class disagreement rates, confusion-matrix interpretation, and exported all **17** disagreement rows to a structured CSV with heuristic boundary explanations.

**Files created / updated**

- `reports/kappa_report.md` (appended `## Extended Annotation Analysis`)
- `reports/disagreement_examples.csv`
- `calculate_kappa.py` (extended; regenerates CSV and appendix)

---

## Additional documentation (cross-reference)

| Topic | Report |
|-------|--------|
| Held-out evaluation | `reports/heldout_evaluation.md` |
| Calibration | `reports/calibration_report.md` |
| Real-world CV | `reports/real_world_cross_validation.md` |
| Symbolic rules | `reports/symbolic_rule_improvements.md` |
| SHAP scope | `reports/shap_limitations.md` |

---

## Research paper revision log

**Status:** ⏸ Not applicable — no research paper source file is present in the repository at this time. When a paper draft is added, append a `## Revision Log` section per supervisor instructions.

---

*Regenerate documentation: `python reports/generate_professor_documentation.py`*  
*Regenerate kappa appendix: `python calculate_kappa.py`*
