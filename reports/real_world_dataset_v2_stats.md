# Real-World Dataset v2 — Statistics

Merged corpus from `real_world_dataset.csv` and reviewed AAEC rows (`aaec_manual_review_250_corrected.csv`, `reviewed=yes`).

## Summary

- **Total rows:** 450
- Rows from v1 (`real_world_dataset.csv`): 200
- Rows from reviewed AAEC: 250
- Combined before deduplication: 450
- **Exact duplicates removed:** 0

## Class distribution (`pramana_label`)

| pramana_label   |   count |   percentage |
|:----------------|--------:|-------------:|
| Pratyaksha      |      50 |        11.11 |
| Anumana         |     270 |        60    |
| Upamana         |      77 |        17.11 |
| Shabda          |      53 |        11.78 |

## Strength distribution (`strength_label`)

| strength_label   |   count |   percentage |
|:-----------------|--------:|-------------:|
| weak             |      94 |        20.89 |
| moderate         |     273 |        60.67 |
| strong           |      83 |        18.44 |

## Source breakdown

| source             |   count |
|:-------------------|--------:|
| aaec_manual_review |     250 |
| real_world_v1      |     200 |

## Schema

```text
text, pramana_label, strength_label
```
