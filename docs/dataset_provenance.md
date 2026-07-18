# Dataset Provenance

Document describing the origin, processing, and final use of every corpus in InferAI. Canonical counts are defined in `reports/dataset_canonical.py`. Regenerate with `python reports/generate_professor_documentation.py`.

## Synthetic Nyaya Dataset

- **Source:** Generated in-project by `dataset_generator.py` and `dataset/labeled_corpus.py`.
- **Why it was created:** To provide a large, balanced seed corpus of English arguments labelled with Nyāya pramāṇa categories when insufficient manually annotated data was available.
- **Generation methodology:** Template-based expansion across eleven domain slots with deterministic label assignment per template (`dataset/labeled_corpus.py`).
- **Number of samples:** **4200** rows in `dataset/raw/nyaya_dataset.csv`.
- **Preprocessing:** Text normalised (whitespace collapse); schema standardised to `text`, `pramana_label` (or `label`).
- **Deduplication:** Exact-text deduplication applied during merge with reviewed corpora (`classification/retrain_merged.py`).
- **Final samples used in retraining merge:** **611** of 4200 synthetic rows retained in `merged_retraining_corpus.csv` after cross-corpus deduplication.

## AAEC

- **Full dataset name:** Argument Annotated Essays Corpus (AAEC) — brat standoff annotations of persuasive essays.
- **Original source:** `dataset/external/AAEC/` (402 essay `.txt` / `.ann` pairs).
- **License:** No separate license file is bundled in this repository; comply with the terms of the original AAEC distribution from which the files were obtained.
- **Number of essays:** **402** (`reports/aaec_statistics.md`).
- **Number of extracted components:** **6050** MajorClaim, Claim, and Premise spans (`dataset/parse_aaec.py` → `aaec_extracted.csv`).
- **Manual review process:** We built annotation sheets (`build_aaec_nyaya_annotation_sheet.py`), conducted manual Nyāya review in staged batches (including an initial 250-row sheet), and exported the finalized reviewed set to `aaec_reviewed_dataset.csv` where `reviewed=yes` (`dataset/build_aaec_reviewed_dataset.py`). Guidelines: `dataset/labeling_guidelines.md`.
- **Final reviewed count:** **650** rows in `aaec_reviewed_dataset.csv`.
- **Preprocessing steps:** Parse brat `.ann` files; extract text-bound spans; remove empty and duplicate rows (39 removed at extraction); normalise whitespace; deduplicate on text at export.

## IBM ArgKP

- **Full dataset name:** IBM Argument Knowledge Pairing (ArgKP) combined corpus.
- **Source:** `dataset/external/IBM_ARGKP/ArgKP_combined.csv`.
- **License:** No separate license file is bundled in this repository; comply with the terms of the original IBM ArgKP release.
- **Raw size:** **36800** rows (`reports/ibm_argkp_statistics.md`).
- **Cleaning pipeline:** `dataset/parse_ibm_argkp.py` — normalise text, drop empty rows, deduplicate arguments → **8639** unique rows (28161 duplicates removed).
- **Manual review:** 600-row Nyāya annotation sheet; human review in `nyaya_annotated_1.csv`; export where `reviewed=yes` (`dataset/build_ibm_reviewed_dataset.py`).
- **Final reviewed count:** **600** rows in `ibm_reviewed_dataset.csv`.

## Final Merged Dataset

The production retraining table is `dataset/processed/merged_retraining_corpus.csv`, built by `classification/retrain_merged.py`. The held-out evaluation set `dataset/test_set.csv` is **never** included in this merge.

**Table — Corpus pipeline summary**

| Dataset | Raw | Clean | Reviewed | Used (retraining merge) |
|---------|----:|------:|---------:|------------------------:|
| Synthetic Nyaya | 4200 | 4200 | — | 611 |
| Real-world (curated v1) | 200 | 200 | 200 | 0* |
| AAEC | 6050 | 6050 | 650 | 650 |
| IBM ArgKP | 36800 | 8639 | 600 | 600 |
| **Merged retraining corpus** | 5450 | — | — | **1861** |
| Held-out test (`test_set.csv`) | 200 | 200 | — | **0 (excluded)** |

\*The early merge `nyaya_dataset_merged.csv` (811 rows) combined synthetic + real-world v1 for initial experiments; the current retraining merge uses synthetic + AAEC + IBM reviewed.

**Composition:** 611 + 650 + 600 = **1861** (after 3589 duplicate texts removed during merge).
