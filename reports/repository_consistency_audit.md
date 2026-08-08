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

- `docs/dataset_provenance.md`
- `reports/dataset_canonical.py`
- `reports/dataset_summary.md`
- `reports/generate_professor_documentation.py`
- `reports/generate_real_world_cross_validation.py`
- `reports/train_test_split.md`

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

The project finalized the reviewed AAEC corpus at **650** rows (expanded from an initial 250-row manual review batch), yielding **1,861** merged training rows (611+650+600) and **1250** total manually reviewed real-world examples. Documentation was aligned to these official values without retraining the model or recomputing evaluation metrics.

## 4. Files skipped

- `dataset/build_aaec_manual_review_250.py` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `dataset/build_real_world_dataset_v2.py` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `dataset/external` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `dataset/processed/aaec_manual_review_250` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `reports/aaec_manual_review_250_stats.md` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `reports/class_weight_comparison.md` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `reports/data_leakage_report.md` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `reports/error_analysis.md` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `reports/final_evaluation_report.md` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `reports/oversampling_comparison.md` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)
- `reports/real_world_dataset_v2_stats.md` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)

## 5. Validation

Remaining inconsistencies: **0**

No outdated AAEC/merged/manual-review counts found in scanned files.

## Regenerate

```bash
python reports/generate_professor_documentation.py
python reports/run_consistency_audit.py
```
