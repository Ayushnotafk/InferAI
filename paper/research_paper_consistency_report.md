# Research Paper Consistency Report

**Paper:** `paper/InferAI_Research_Paper.md`  
**Title:** PramāṇaNet: A Nyāya-Inspired Neuro-Symbolic Framework for Explainable Epistemic Reasoning Classification  
**Date:** 2026-07-18  
**Scope:** Verification that every numerical claim in the paper is grounded in InferAI repository reports/code. No invented experiments, datasets, or metrics.

---

## 1. Canonical dataset counts

| Statistic | Paper value | Source | Status |
|-----------|------------:|--------|--------|
| Synthetic raw | 4,200 | `docs/dataset_provenance.md`, `dataset_canonical.py` | ✓ |
| Synthetic used in merge | 611 | provenance, `final_retraining_report.md` | ✓ |
| AAEC essays | 402 | `aaec_statistics.md` | ✓ |
| AAEC extracted components | 6,050 | `aaec_statistics.md` | ✓ |
| AAEC reviewed | 650 | `dataset_canonical.py`, provenance | ✓ |
| IBM raw | 36,800 | `ibm_argkp_statistics.md` | ✓ |
| IBM cleaned unique | 8,639 | `ibm_argkp_statistics.md` | ✓ |
| IBM reviewed | 600 | `ibm_reviewed_dataset_stats.md` | ✓ |
| Merged corpus | 1,861 (=611+650+600) | `dataset_canonical.py` | ✓ |
| Dupes removed at merge | 3,589 | provenance, final retraining | ✓ |
| Raw before dedup | 5,450 | `MERGE_RAW_BEFORE_DEDUP` | ✓ |
| Manually reviewed real-world | 1,250 | 650+600 | ✓ |
| Train / Val | 1,488 / 373 | `train_test_split.md` | ✓ |
| Held-out test | 200 | `test_set.csv`, train_test_split | ✓ |
| Test excluded from merge | Yes (0 used) | provenance | ✓ |

**Historical notes (explicitly labeled as historical in paper):** early merge 811 rows; initial AAEC review batch 250; CV execution footnote on 850-row export — consistent with `repository_consistency_audit.md`.

---

## 2. Evaluation metrics

| Statistic | Paper value | Source | Status |
|-----------|------------:|--------|--------|
| In-dist holdout accuracy | 0.9287 | `final_retraining_report.md` | ✓ |
| In-dist holdout macro F1 | 0.9326 | same | ✓ |
| In-dist macro P / R | 0.9484 / 0.9198 | same | ✓ |
| In-dist train accuracy | 0.9573 | same | ✓ |
| Holdout support (reported) | 533 | classification report in final_retraining | ✓ |
| Template early accuracy / macro F1 | 0.9550 / 0.9552 | `heldout_evaluation.md` | ✓ |
| Template original (no resample) | Acc 0.6850, macro F1 0.6831 | `error_analysis.md`, oversampling comparison | ✓ |
| Template class-weight | Acc 0.6600, macro F1 0.6556 | `class_weight_comparison.md` | ✓ |
| Template oversampled | Acc 0.6000, macro F1 0.6007 | `final_retraining_report.md` §3 | ✓ |
| Real-world CV accuracy (pooled) | 0.9118 | `real_world_cross_validation.md` | ✓ |
| Real-world CV macro F1 (pooled) | 0.3414 | same | ✓ |
| CV fold mean macro F1 | 0.3359 ± 0.0411 | same | ✓ |
| CV Anumana / Pratyaksha / Upamana / Shabda F1 | 0.9531 / 0.0 / 0.0 / 0.4127 | same | ✓ |
| ECE | 0.457 | `calibration_report.md` | ✓ |
| Brier (mean OvR) | 0.0949 | same | ✓ |
| Exact leakage | 0 | `data_leakage_report.md` | ✓ |
| Near-duplicates | 506 | same | ✓ |
| JSD (test vs merged) | 0.5065 | `test_set_distribution_analysis.md` | ✓ |
| Class TV distance | 0.576 | same | ✓ |
| Early synthetic holdout (~0.996) | Contextual only | `final_evaluation_report.md` | ✓ labeled as early |
| Early curated-test 0.76 snapshot | Contextual | `final_evaluation_report.md` | ✓ labeled as snapshot |

---

## 3. Annotation / kappa

| Statistic | Paper value | Source | Status |
|-----------|------------:|--------|--------|
| Matched examples | 200 | `kappa_report.md` | ✓ |
| Agreements / disagreements | 183 / 17 | same | ✓ |
| Agreement % | 91.50% | same | ✓ |
| Cohen’s κ | 0.8804 | same | ✓ |
| Bootstrap 95% CI | [0.8212, 0.9295] | same | ✓ |
| Assisted-annotator caveat | Stated | kappa_report Interpretation | ✓ |

---

## 4. Lexical diversity

| Statistic | Paper value | Source | Status |
|-----------|------------:|--------|--------|
| Synthetic TTR / Guiraud / MATTR-50 | 0.003367 / 0.782816 / 0.797221 | `lexical_diversity_report.md` | ✓ |
| Real-world (200) TTR / Guiraud / MATTR | 0.518248 / 18.6965 / 0.834957 | same | ✓ |

---

## 5. Symbolic engine / hybrid fusion

| Statistic | Paper value | Source | Status |
|-----------|------------:|--------|--------|
| Total rules | 77 | `_RULE_SPECS`, `rules_v2.yaml` | ✓ |
| Rules per pramāṇa | 23 / 17 / 13 / 24 | `rule_statistics.md` | ✓ |
| Avg weights (P/A/U/S) | 1.77 / 2.02 / 2.31 / 2.23 | computed from `_RULE_SPECS` | ✓ |
| Overall avg weight | 2.06 | same | ✓ |
| Min / max weight | 0.7 / 3.4 | same | ✓ |
| Prior ~28 binary rules | Stated | `symbolic_rule_improvements.md` | ✓ |
| ML / rule fusion weights | 0.8 / 0.2 | `hybrid_reasoning.py`, README | ✓ |
| Agreement boost / add | 1.10 / +4.0 | same | ✓ |
| Disagreement mult | 0.90 | same | ✓ |
| Multi-cue bonus / cap | 0.12 / 0.36 | same | ✓ |
| Soft floor exponent | 1.35 | same | ✓ |
| Mix penalty | ×0.88 if ≥3 families | same | ✓ |

---

## 6. Architecture / explainability

| Claim | Source | Status |
|-------|--------|--------|
| Encoder `all-MiniLM-L6-v2`, 384-d, frozen | README, embedder, shap_limitations | ✓ |
| LogisticRegression `max_iter=1000` | train_model.py, CV report | ✓ |
| SHAP = LinearExplainer on embedding dims | shap_limitations.md | ✓ |
| SHAP does not explain tokens/rules/structure | same | ✓ |
| No counterfactual generator | Codebase audit (paper §7.5) | ✓ |
| Nyāya-inspired / not full computational Nyāya | README, progress report, labeling guidelines | ✓ |

---

## 7. Figures referenced

| Figure | Path | Status |
|--------|------|--------|
| Fig 1 Lexical metrics | `reports/figures/lexical_diversity_metrics.png` | ✓ exists |
| Fig 2 Token/vocab | `reports/figures/lexical_diversity_token_vocab.png` | ✓ exists |
| Fig 3 Held-out CM | `reports/figures/heldout_confusion_matrix.png` | ✓ exists |
| Fig 4 Real-world CV CM | `reports/figures/real_world_cv_confusion_matrix.png` | ✓ exists |
| Fig 5 Calibration | `reports/figures/calibration_reliability_diagram.png` | ✓ exists |
| Fig 6 Error analysis CM | `reports/figures/error_analysis_confusion_matrix.png` | ✓ exists |

---

## 8. References policy

References [1]–[15] are standard works corresponding to methods/corpora actually used (SHAP, AAEC lineage, Nyāya classical sources, Sentence-BERT, Cohen’s κ, Guiraud, MATTR, IBM ArgKP, project artifacts). No fabricated experimental papers. IBM ArgKP and project artifacts are cited as corpus/project sources rather than inventing peer-review venues for them.

**Fixed during paper polish:** Guiraud attribution corrected to P. Guiraud (1954); removed incorrect “L. Einav” placeholder.

---

## 9. Known tensions (documented, not hidden)

| Tension | How paper handles it |
|---------|----------------------|
| Multiple accuracies (0.60–0.996) | Protocol/snapshot table in §15.1 and §6 |
| Canonical 1,250 vs CV executed on 850-row export | Explicit footnote in §6.5 and §25.4 |
| AAEC reviewed stats table summing to 250 class counts while total labeled 650 | Paper uses canonical 650 for size; class % cited from reviewed exports with imbalance narrative |
| Holdout support 533 vs 20% of 1861 (=373) | Paper quotes report’s classification-report support (533) as written in `final_retraining_report.md` without inventing a reconciliation |
| Production oversampled worse than original on template test | Stated plainly; original preferred for template macro F1 |
| κ partly rule-assisted | Caveat in abstract-adjacent claims and §5.6 |

---

## 10. Validation checklist

| Check | Result |
|-------|--------|
| No invented experiments | ✓ |
| No invented datasets | ✓ |
| No invented evaluation numbers | ✓ |
| Dataset counts match `dataset_canonical.py` | ✓ |
| Contradictory metrics labeled by protocol | ✓ |
| “Full computational Nyāya” claim avoided | ✓ |
| Symbolic rules = 77, matches YAML and code | ✓ |
| Remaining undocumented inventions | **0** |

---

## 11. Word count

Word count of `paper/InferAI_Research_Paper.md`: **≈10,000+** (IEEE-style draft including appendices and extended sections 13–27). Target band: 10,000–12,000 words.

---

*End of consistency report.*
