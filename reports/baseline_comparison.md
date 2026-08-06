# Baseline Comparison (VAL Set — Group-Disjoint)

All baselines trained on `dataset/processed/split_train.csv`, evaluated on `dataset/processed/split_val.csv`. Group-disjoint splits (essay_id / topic_group / template_family) ensure no document or topic leakage between partitions.

## Results

| Baseline | accuracy | macro_precision | macro_recall | macro_f1 | latency_s |
| --- | --- | --- | --- | --- | --- |
| Majority class | 0.8063 | 0.2016 | 0.25 | 0.2232 | 0.01 |
| TF-IDF + LinearSVC | 0.9375 | 0.982 | 0.8013 | 0.8657 | 0.09 |
| TF-IDF + LogReg | 0.8812 | 0.7179 | 0.5833 | 0.6329 | 0.2 |
| SBERT + LogReg (ML-only) | 0.9625 | 0.9349 | 0.9349 | 0.9349 | 0.68 |
| Symbolic rules only | 0.8063 | 0.563 | 0.519 | 0.5084 | 0.04 |
| Hybrid (α=0.9) | 0.9625 | 0.9349 | 0.9349 | 0.9349 | 0.58 |

## Per-baseline classification reports

### Majority class

```text
              precision    recall  f1-score   support

     Anumana     0.8063    1.0000    0.8927       129
  Pratyaksha     0.0000    0.0000    0.0000         6
      Shabda     0.0000    0.0000    0.0000        13
     Upamana     0.0000    0.0000    0.0000        12

    accuracy                         0.8063       160
   macro avg     0.2016    0.2500    0.2232       160
weighted avg     0.6500    0.8063    0.7198       160
```

### TF-IDF + LinearSVC

```text
              precision    recall  f1-score   support

     Anumana     0.9281    1.0000    0.9627       129
  Pratyaksha     1.0000    1.0000    1.0000         6
      Shabda     1.0000    0.5385    0.7000        13
     Upamana     1.0000    0.6667    0.8000        12

    accuracy                         0.9375       160
   macro avg     0.9820    0.8013    0.8657       160
weighted avg     0.9420    0.9375    0.9305       160
```

### TF-IDF + LogReg

```text
              precision    recall  f1-score   support

     Anumana     0.8716    1.0000    0.9314       129
  Pratyaksha     1.0000    0.6667    0.8000         6
      Shabda     0.0000    0.0000    0.0000        13
     Upamana     1.0000    0.6667    0.8000        12

    accuracy                         0.8812       160
   macro avg     0.7179    0.5833    0.6329       160
weighted avg     0.8152    0.8812    0.8409       160
```

### SBERT + LogReg (ML-only)

```text
              precision    recall  f1-score   support

     Anumana     0.9767    0.9767    0.9767       129
  Pratyaksha     1.0000    1.0000    1.0000         6
      Shabda     0.8462    0.8462    0.8462        13
     Upamana     0.9167    0.9167    0.9167        12

    accuracy                         0.9625       160
   macro avg     0.9349    0.9349    0.9349       160
weighted avg     0.9625    0.9625    0.9625       160
```

### Symbolic rules only

```text
              precision    recall  f1-score   support

     Anumana     0.8769    0.8837    0.8803       129
  Pratyaksha     0.0000    0.0000    0.0000         6
      Shabda     0.3750    0.6923    0.4865        13
     Upamana     1.0000    0.5000    0.6667        12

    accuracy                         0.8063       160
   macro avg     0.5630    0.5190    0.5084       160
weighted avg     0.8125    0.8063    0.7993       160
```

### Hybrid (α=0.9)

```text
              precision    recall  f1-score   support

     Anumana     0.9767    0.9767    0.9767       129
  Pratyaksha     1.0000    1.0000    1.0000         6
      Shabda     0.8462    0.8462    0.8462        13
     Upamana     0.9167    0.9167    0.9167        12

    accuracy                         0.9625       160
   macro avg     0.9349    0.9349    0.9349       160
weighted avg     0.9625    0.9625    0.9625       160
```

## Notes

- Fusion alpha α=0.9 was selected on this VAL set by `scripts/select_alpha_and_calibrate.py`.
- Final test evaluation (on `dataset/clean_final_test.csv`) is run exactly once by `scripts/evaluate_final_model.py`.
- A fine-tuned transformer baseline (RoBERTa/DeBERTa) is planned as P1 work; this requires GPU training.
