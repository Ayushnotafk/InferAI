# InferAI — Final Evaluation Report
This report summarizes quantitative metrics produced during training and a small qualitative review on the held-out `dataset/test_set.csv`.
## Quantitative summary
- **Train accuracy:** 0.9985119047619048
- **Test accuracy (20% random holdout from training table):** 0.9964285714285714
- **Macro precision:** 0.9958791208791209
- **Macro recall:** 0.9969361702127659
- **Macro F1:** 0.9963853725683613

## Held-out curated test set
- **Accuracy on `dataset/test_set.csv`:** 0.76

### Classification report (held-out)
```text
precision    recall  f1-score   support

     Anumana     1.0000    0.7273    0.8421        44
  Pratyaksha     0.6034    0.7609    0.6731        46
      Shabda     0.7273    0.5614    0.6337        57
     Upamana     0.8030    1.0000    0.8908        53

    accuracy                         0.7600       200
   macro avg     0.7834    0.7624    0.7599       200
weighted avg     0.7789    0.7600    0.7567       200
```

## Sklearn report (random holdout)
```text
precision    recall  f1-score   support

     Anumana     1.0000    0.9957    0.9979       235
  Pratyaksha     1.0000    1.0000    1.0000       176
      Shabda     1.0000    0.9920    0.9960       250
     Upamana     0.9835    1.0000    0.9917       179

    accuracy                         0.9964       840
   macro avg     0.9959    0.9969    0.9964       840
weighted avg     0.9965    0.9964    0.9964       840
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
