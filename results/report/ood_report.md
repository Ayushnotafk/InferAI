# Out-of-Distribution Evaluation

OOD accuracy=0.550 vs ID=1.000 (drop=0.450). Mean confidence ID=59.1% / OOD=36.7%. Jensen–Shannon divergence between predicted label distributions=0.1037 (0=identical, 1=maximal).

| Split | n | Accuracy | Macro F1 | Mean conf |
|-------|---|----------|----------|-----------|
| ID | 80 | 1.000 | 1.000 | 59.1 |
| OOD | 80 | 0.550 | 0.295 | 36.7 |

JSD (predicted labels): **0.1037**
