# Publication Tables

### Dataset statistics

| Split | N | Vocab | Avg tokens | TTR | Guiraud R |
| --- | --- | --- | --- | --- | --- |
| train | 4200 | 364 | 25.7398 | 0.0034 | 1.1071 |
| test | 200 | 228 | 16.4000 | 0.0695 | 3.9811 |

### Benchmark comparison

| Mode | Accuracy | Acc 95% CI | Macro F1 | F1 95% CI | ECE |
| --- | --- | --- | --- | --- | --- |
| ml | 0.6000 | [0.535, 0.665] | 0.6010 | [0.534, 0.666] | 0.1420 |
| symbolic | 0.5500 | [0.480, 0.620] | 0.5710 | [0.502, 0.640] | 0.1760 |
| hybrid | 0.7650 | [0.710, 0.825] | 0.7620 | [0.699, 0.819] | 0.3460 |
| adaptive | 0.7650 | [0.705, 0.825] | 0.7620 | [0.697, 0.819] | 0.2350 |

### Alpha ablation study

| alpha | accuracy | macro_f1 | ece |
| --- | --- | --- | --- |
| 0.0000 | 0.5500 | 0.5713 | 0.1765 |
| 0.2000 | 0.7650 | 0.7619 | 0.3462 |
| 0.4000 | 0.7400 | 0.7311 | 0.2750 |
| 0.6000 | 0.6650 | 0.6555 | 0.1431 |
| 0.8000 | 0.6100 | 0.6092 | 0.1278 |
| 1.0000 | 0.6000 | 0.6007 | 0.1422 |

### Adaptive routing statistics

| Mean alpha | Min alpha | Max alpha | Std alpha | % ML dominated | % Rules dominated | % Balanced | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4308 | 0.0500 | 0.8000 | 0.2283 | 48.0000 | 31.0000 | 21.0000 | 0.7650 |

### Fallacy detection

| N | Accuracy | Binary P | Binary R | FP | FN |
| --- | --- | --- | --- | --- | --- |
| 12.0000 | 0.7500 | 0.8571 | 0.8571 | 1.0000 | 1.0000 |

### Explainability evaluation (Likert 1–5)

_No data._
