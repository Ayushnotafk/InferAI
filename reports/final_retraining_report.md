# InferAI — Final Retraining Report

Model retrained on merged synthetic + AAEC reviewed + IBM reviewed corpora. Architecture unchanged: **Sentence-BERT (`all-MiniLM-L6-v2`) + Logistic Regression**.

## 1. Merged corpus

- Synthetic source: `dataset\raw\nyaya_dataset.csv`
- AAEC reviewed: `dataset/processed/aaec_reviewed_dataset.csv`
- IBM reviewed: `dataset/processed/ibm_reviewed_dataset.csv`
- Merged export: `dataset/processed/merged_retraining_corpus.csv`
- **Total rows:** 1861
- **Duplicates removed:** 3589
- Shuffle seed: `42`

### Source contribution (after dedup)

- Synthetic: 611
- AAEC: 650
- IBM: 600

### Class distribution

- Pratyaksha: 500 (18.8%)
- Anumana: 1163 (43.7%)
- Upamana: 500 (18.8%)
- Shabda: 500 (18.8%)

### Strength distribution (real-world rows)

- weak: 36
- moderate: 958
- strong: 142

## 2. Evaluation (20% random holdout)

- **Accuracy:** 0.9287
- **Macro precision:** 0.9484
- **Macro recall:** 0.9198
- **Macro F1:** 0.9326
- Train accuracy: 0.9573

### Classification report

```text
precision    recall  f1-score   support

     Anumana     0.8863    0.9617    0.9224       235
  Pratyaksha     0.9885    0.9663    0.9773        89
      Shabda     0.9890    0.8738    0.9278       103
     Upamana     0.9300    0.8774    0.9029       106

    accuracy                         0.9287       533
   macro avg     0.9484    0.9198    0.9326       533
weighted avg     0.9319    0.9287    0.9288       533
```

### Confusion matrix

| True \ Pred | Anumana | Pratyaksha | Shabda | Upamana |
| --- | --- | --- | --- | --- |
| Anumana | 226 | 1 | 1 | 7 |
| Pratyaksha | 3 | 86 | 0 | 0 |
| Shabda | 13 | 0 | 90 | 0 |
| Upamana | 13 | 0 | 0 | 93 |

## 3. Held-out curated test set (`dataset/test_set.csv`)

- Accuracy: **0.6**

```text
precision    recall  f1-score   support

     Anumana     0.4889    0.5000    0.4944        44
  Pratyaksha     0.4268    0.7609    0.5469        46
      Shabda     0.6774    0.3684    0.4773        57
     Upamana     1.0000    0.7925    0.8842        53

    accuracy                         0.6000       200
   macro avg     0.6483    0.6054    0.6007       200
weighted avg     0.6638    0.6000    0.6049       200
```

## 4. Saved artifacts

- `models/nyaya_model.pkl`
- `models/label_encoder.pkl`
- `models/shap_background.npy` (200 rows)
- `models/evaluation/metrics.json`
- `models/evaluation/classification_report.txt`
- `models/evaluation/confusion_matrix.png`

## 5. API compatibility check

`api/app.py` imports `predict_pramana_detailed` from `classification/predictor.py`, which loads `models/nyaya_model.pkl` and `models/label_encoder.pkl`. No API code changes required.

- nyaya_model.pkl exists: **PASS**
- label_encoder.pkl exists: **PASS**
- shap_background.npy exists: **PASS**
- predictor-style inference OK: **PASS**

Sample inference: "Smoke is rising…" → **Upamana**
