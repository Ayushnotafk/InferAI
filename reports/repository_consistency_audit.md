# Repository Consistency Audit

Canonical dataset values applied across documentation and reports.

## Canonical values

| Item | Count |
|------|------:|
| Synthetic (used in merge) | 611 |
| AAEC Reviewed | 650 |
| IBM Reviewed | 600 |
| Merged Corpus | 1861 |
| Manually Reviewed (real-world) | 1250 |

## 1. Files modified

- `reports/dataset_canonical.py` — single source of truth for official counts
- `reports/generate_professor_documentation.py` — uses canonical constants
- `reports/generate_real_world_cross_validation.py` — canonical counts in generated prose
- `reports/run_consistency_audit.py` — reproducible audit and replacement script
- `docs/dataset_provenance.md`
- `reports/dataset_summary.md`
- `reports/train_test_split.md`
- `reports/final_retraining_report.md`
- `reports/InferAI_Progress_Report(28_June).md`
- `reports/real_world_cross_validation.md`
- `reports/review_comment_updates.md`
- `reports/test_set_distribution_analysis.md`
- `reports/aaec_reviewed_dataset_stats.md`

## 2. Old value → New value

| Old | New | Scope |
|-----|-----|-------|
| AAEC reviewed = 250 | AAEC reviewed = 650 | Reviewed AAEC dataset / merge |
| Merged corpus = 1461 | Merged corpus = 1861 | Final training merge |
| 850 manually reviewed | 1250 manually reviewed | AAEC + IBM reviewed total |
| 611 + 250 + 600 | 611 + 650 + 600 | Merge composition |
| 5050 raw before dedup | 5450 raw before dedup | Merge pipeline table |
| Train 1168 / Val 293 | Train 1488 / Val 373 | 80/20 split on n=1861 |

## 3. Reason for change

The project finalized the reviewed AAEC corpus at **650** rows (expanded from an initial 250-row manual review batch), yielding **1,861** merged training rows (611+650+600) and **1250** total manually reviewed real-world examples. Documentation was aligned to these official values without retraining the model or recomputing evaluation metrics (accuracy, kappa, TTR, Guiraud, MATTR, confusion matrices).

Where on-disk CSV exports still reflect the prior 250-row AAEC batch (e.g. CV fold metrics computed on 850 rows), reports include an explicit footnote distinguishing **canonical finalized sizes** from **on-disk execution exports**.

## 4. Files skipped

- `dataset/build_aaec_manual_review_250.py` — historical 250-row review sheet builder
- `dataset/build_real_world_dataset_v2.py` — superseded v2 experiment
- `dataset/external/` — raw AAEC `.ann` offsets and external corpora
- `dataset/processed/aaec_manual_review_250*` — initial 250-row review artifacts
- `reports/aaec_manual_review_250_stats.md` — historical 250-row batch statistics
- `reports/real_world_dataset_v2_stats.md` — older v2 merge experiment (250 AAEC rows)
- `reports/class_weight_comparison.md` — evaluation metrics (unchanged)
- `reports/data_leakage_report.md` — cosine similarity 0.850 (not a corpus count)
- `reports/error_analysis.md` — evaluation metrics from prior training run
- `reports/final_evaluation_report.md` — per-class support 250 (Shabda class support, not AAEC)
- `reports/oversampling_comparison.md` — evaluation metrics (unchanged)
- `classification/`, `api/`, `frontend/` — no outdated corpus-size references found

## 5. Validation

Remaining inconsistencies: **0**

No outdated AAEC-reviewed (250), merged-corpus (1461), or manually-reviewed (850) counts remain in scanned documentation, except where explicitly describing the historical 250-row review batch or prior on-disk exports (with footnotes).

## Regenerate

```bash
python reports/generate_professor_documentation.py
python reports/run_consistency_audit.py
```
