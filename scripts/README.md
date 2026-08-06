# InferAI Scripts — Reviewer Fix Pipeline

Run these scripts **in order** to rebuild the evaluation from scratch.

## Execution Order

```
Step 1:  python scripts/add_group_keys.py
Step 2:  python scripts/build_clean_splits.py
Step 3:  python scripts/select_alpha_and_calibrate.py
Step 4:  python scripts/run_baseline_comparison.py
Step 5:  python scripts/run_group_cv.py
Step 6:  python scripts/evaluate_rule_coverage.py
Step 7:  python scripts/run_integrity_audit.py
Step 8:  python scripts/build_blind_annotation_sheet.py
Step 9:  python scripts/evaluate_final_model.py      ← RUN EXACTLY ONCE
```

## What each script fixes

| Script | Reviewer Issue | What it does |
|---|---|---|
| `add_group_keys.py` | C-03, M-15, MISS-2 | Adds `essay_id`, `topic_group`, `template_family` to all rows |
| `build_clean_splits.py` | C-01, C-03, MISS-2 | Group-disjoint 3-way split; frozen manifests; isolates near-dup pads |
| `select_alpha_and_calibrate.py` | C-01, C-04, C-05 | Alpha sweep on VAL only; temperature scaling + isotonic calibration |
| `run_baseline_comparison.py` | M-09 | Majority, TF-IDF+SVM, TF-IDF+LR, SBERT+LR, rules-only, hybrid |
| `run_group_cv.py` | M-15 | StratifiedGroupKFold CV; verifies zero group overlap per fold |
| `evaluate_rule_coverage.py` | MISS-4 | Per-class rule coverage; explainability rating sheet |
| `run_integrity_audit.py` | C-01, C-02 | Single source-of-truth for all split sizes and alpha selection |
| `build_blind_annotation_sheet.py` | C-06 | 200-example blind annotation sheet, no labels, no rule output |
| `evaluate_final_model.py` | C-01, C-05, M-09 | ONE-SHOT final test evaluation; saves frozen run record |

## Key results (from VAL evaluation)

| System | Macro-F1 (VAL) |
|---|---:|
| Majority class | 0.223 |
| TF-IDF + LinearSVC | 0.866 |
| TF-IDF + LogReg | 0.633 |
| SBERT + LogReg (ML-only) | **0.935** |
| Symbolic rules only | 0.508 |
| Hybrid (α=0.9) | **0.935** |

**Group-disjoint real-world CV** (M-15 fix): macro-F1 = 0.247 (was 0.341 with row-level CV — confirms leakage was real).

**Calibration** (C-05 fix): Isotonic regression ECE = 0.018 (was 0.458 uncalibrated).

## Files produced

| File | Purpose |
|---|---|
| `dataset/processed/aaec_reviewed_with_groups.csv` | AAEC + essay_id |
| `dataset/processed/ibm_reviewed_with_groups.csv` | IBM + topic_group |
| `dataset/processed/merged_corpus_with_groups.csv` | Merged group-aware |
| `dataset/processed/split_train.csv` | Training partition |
| `dataset/processed/split_val.csv` | Validation (alpha/calib selection) |
| `dataset/clean_final_test.csv` | **UNTOUCHED** final test partition |
| `dataset/split_manifests/experiment_registry.json` | Frozen run record |
| `dataset/split_manifests/full_split_manifest.csv` | Row-level split assignments |
| `dataset/annotations/blind_annotation_sheet_v1.csv` | For human annotators |
| `models/calibration/temperature.json` | T=0.393 fitted on VAL |
| `models/calibration/isotonic_*.pkl` | Per-class isotonic regressors |
| `reports/alpha_selection_report.md` | Alpha sweep results |
| `reports/calibration_comparison_report.md` | ECE/Brier/NLL comparison |
| `reports/baseline_comparison.md` | Full baseline table |
| `reports/group_cv_report.md` | Group-disjoint CV results |
| `reports/rule_coverage_report.md` | Symbolic rule coverage |
| `reports/explainability_rating_sheet.csv` | For human XAI raters |
| `reports/experiment_integrity_audit.md` | Split arithmetic reconciliation |
| `reports/final_test_evaluation.md` | **Canonical paper result table** |
