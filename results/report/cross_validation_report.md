# 5-Fold Cross-Validation Report

Corpus: `C:\Users\ayush\OneDrive\Desktop\sem 4\prnew14th\dataset\raw\nyaya_dataset_merged.csv` (n=811).
Protocol: StratifiedKFold on Sentence-BERT embeddings; LogisticRegression re-fit per fold.

- Mean accuracy: **0.9038** ± 0.0068
- Mean macro F1: **0.8868** ± 0.0107

| Fold | n | Accuracy | Macro F1 |
|------|---|----------|----------|
| 1 | 163 | 0.8957 | 0.8941 |
| 2 | 162 | 0.9074 | 0.8958 |
| 3 | 162 | 0.9012 | 0.8761 |
| 4 | 162 | 0.9136 | 0.8939 |
| 5 | 162 | 0.9012 | 0.8742 |
