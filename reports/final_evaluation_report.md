# InferAI — Final Evaluation Report
This report summarizes quantitative metrics produced during training and a small qualitative review on the held-out `dataset/test_set.csv`.
## Quantitative summary
- **Train accuracy:** 0.9572769953051643
- **Test accuracy (20% random holdout from training table):** 0.9287054409005628
- **Macro precision:** 0.9484478114853369
- **Macro recall:** 0.9197847902060158
- **Macro F1:** 0.9326173449425448

## Held-out curated test set
- **Accuracy on `dataset/test_set.csv`:** 0.6

### Classification report (held-out)
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

## Sklearn report (random holdout)
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

## Confusion matrices (artifacts)
- `models/evaluation/confusion_matrix.png` — random holdout split
- `models/evaluation/confusion_matrix_test_set.png` — curated test set (if generated)

## Qualitative notes
The logistic regression head operates on Sentence-BERT embeddings; errors often occur when rhetorical style mixes multiple pramāṇa cues in one short span, or when implicit authority language appears without explicit attribution.

## Example failure cases (first few mislabels on test set)
1. **True:** `Pratyaksha` → **Predicted:** `Shabda`  
   - Text: "Bench notes (science): the sample fluoresced under UV immediately after staining."
2. **True:** `Pratyaksha` → **Predicted:** `Shabda`  
   - Text: "Bench notes (technology): the sample fluoresced under UV immediately after staining."
3. **True:** `Pratyaksha` → **Predicted:** `Shabda`  
   - Text: "Bench notes (education): the sample fluoresced under UV immediately after staining."
4. **True:** `Pratyaksha` → **Predicted:** `Shabda`  
   - Text: "Bench notes (medicine): the sample fluoresced under UV immediately after staining."
5. **True:** `Pratyaksha` → **Predicted:** `Shabda`  
   - Text: "Bench notes (politics): the sample fluoresced under UV immediately after staining."
6. **True:** `Pratyaksha` → **Predicted:** `Shabda`  
   - Text: "Bench notes (philosophy): the sample fluoresced under UV immediately after staining."

## Observations
- Hybrid fusion (`classification/hybrid_reasoning.py`) can stabilize borderline cases at inference time, but the supervised head still reflects whatever distribution is present in `infer_dataset.csv`.
- Composite strength (`reasoning_strength/composite.py`) intentionally decouples rhetorical strength from raw softmax peaks.
