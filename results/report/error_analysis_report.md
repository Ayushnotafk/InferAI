# Error Analysis Report

- Samples: 200
- Errors: 47
- Accuracy: 0.7650
- Low-confidence errors: 32
- Rule rescues: 33
- Fusion hurt (ML was correct): 0
- Residual ML failures: 47
- Symbolic failures: 0

## Top confused class pairs

| True | Predicted | Count |
|------|-----------|-------|
| Shabda | Pratyaksha | 25 |
| Anumana | Pratyaksha | 22 |

## Failure clusters

| Cluster | Count | Recommendation |
|---------|-------|----------------|
| embedding_failure | 47 | Collect more minority-class training examples; consider class-balanced fine-tuning of the logistic head on hard embedding neighborhoods. |

## Narrative

Most frequent confusion: Shabda → Pratyaksha (25 cases). Rules rescued 33 ML errors; fusion overrode 0 correct ML decisions; 47 residual ML failures; 0 symbolic-aligned residual errors.
