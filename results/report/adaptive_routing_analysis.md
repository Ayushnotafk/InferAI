# Adaptive Routing Analysis

- Samples: 200
- Mean α: 0.4308
- Min / Max α: 0.0500 / 0.8000
- Std α: 0.2283
- Accuracy: 0.7650

## Routing buckets

| Bucket | % | Accuracy |
|--------|---|----------|
| ml_dominated | 48.0% | 0.698 |
| rules_dominated | 31.0% | 1.000 |
| balanced | 21.0% | 0.571 |

## Interpretation

Adaptive routing improves over fixed high-ML hybrids by lowering alpha when symbolic cues are present (recovering rule-correct / ML-wrong cases) and raising alpha when cues are absent so ML remains the decision maker under the soft rule floor. Empirical mean alpha=0.431 (base=0.2).
