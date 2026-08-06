# Alpha Selection Report (VAL Set)

Alpha was swept on the **validation partition only** (`dataset/processed/split_val.csv`). The test partition was not used at this step.

## Results

| alpha | accuracy | macro_f1 |
| --- | --- | --- |
| 0.0 | 0.8063 | 0.5084 |
| 0.1 | 0.8688 | 0.8417 |
| 0.2 | 0.8688 | 0.8417 |
| 0.3 | 0.8812 | 0.8508 |
| 0.4 | 0.9 | 0.8602 |
| 0.5 | 0.9437 | 0.9056 |
| 0.6 | 0.95 | 0.9178 |
| 0.7 | 0.9563 | 0.9261 |
| 0.8 | 0.9563 | 0.9261 |
| 0.9 | 0.9625 | 0.9349 |
| 1.0 | 0.9625 | 0.9349 |

**Selected alpha: 0.9** (criterion: macro-F1 = 0.9349).

## System architecture note

The fusion formula is fixed: `fused = α·p_ML + (1−α)·p_rules`. The confidence display applies a heuristic ±10% boost/reduction based on ML↔rule agreement, but this does **not** change the label or the fused probability distribution. There is no per-example adaptive alpha. This is documented exactly in `classification/hybrid_reasoning.py`.
