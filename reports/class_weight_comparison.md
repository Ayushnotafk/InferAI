# Class Weight Comparison (`class_weight="balanced"`)

Comparison on held-out `dataset/test_set.csv` (n=200) before and after adding `class_weight="balanced"` to LogisticRegression in `classification/retrain_merged.py`.

## Aggregate metrics

| Metric | Previous | New | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 0.6850 | 0.6600 | -0.0250 |
| Macro F1 | 0.6831 | 0.6556 | -0.0275 |

## Per-class recall

| Class | Previous recall | New recall | Change | Improved? |
| --- | ---: | ---: | ---: | --- |
| Anumana | 1.0000 | 0.5000 | -0.5000 | No |
| Pratyaksha | 0.7174 | 0.8043 | +0.0870 | **Yes** |
| Shabda | 0.4386 | 0.4035 | -0.0351 | **No** |
| Upamana | 0.6604 | 0.9434 | +0.2830 | **Yes** |

## Focus classes (Shabda, Upamana, Pratyaksha)

- **Shabda:** 0.4386 → 0.4035 — **Decreased** by 0.0351 (3.5 pp)
- **Upamana:** 0.6604 → 0.9434 — **Improved** by 0.2830 (28.3 pp)
- **Pratyaksha:** 0.7174 → 0.8043 — **Improved** by 0.0870 (8.7 pp)

## Interpretation

Balanced class weights decreased held-out accuracy (68.5% → 66.0%) and decreased macro F1 (0.683 → 0.656). Recall improved for: **Upamana, Pratyaksha**. Recall declined for: **Shabda**. Re-weighting penalizes the dominant Anumana class (~80% of merged training data), shifting the decision boundary toward minority pramana labels.
