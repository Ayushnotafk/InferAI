# AAEC Reviewed Dataset — Statistics

Exported from the finalized AAEC reviewed pipeline (`dataset/build_aaec_reviewed_dataset.py`, `reviewed=yes`). An initial 250-row manual review batch (`aaec_manual_review_250_corrected.csv`) preceded the expanded corpus.

## Summary

- **Total rows (canonical):** 650
- Rows before deduplication: 650
- **Duplicates removed:** 0

## Class distribution (`pramana_label`)

| pramana_label   |   count |   percentage |
|:----------------|--------:|-------------:|
| Pratyaksha      |       0 |          0   |
| Anumana         |     220 |         88   |
| Upamana         |      27 |         10.8 |
| Shabda          |       3 |          1.2 |

## Strength distribution (`strength_label`)

| strength_label   |   count |   percentage |
|:-----------------|--------:|-------------:|
| weak             |      32 |         12.8 |
| moderate         |     210 |         84   |
| strong           |       8 |          3.2 |

## Schema

```text
text, pramana_label, strength_label
```
