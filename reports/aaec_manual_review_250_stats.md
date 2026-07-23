# AAEC Manual Review 250 — Summary Statistics

Built from `aaec_nyaya_annotation_sheet.csv`. Total rows: **250**.

## Selection

- Non-Anumana rows (all included): **55**
- Anumana rows (random sample): **195**
- Unique essays: **202**
- Random seed: **42**

## final_pramana distribution

| final_pramana   |   count |   percentage |
|:----------------|--------:|-------------:|
| Pratyaksha      |       2 |          0.8 |
| Anumana         |     195 |         78   |
| Upamana         |      36 |         14.4 |
| Shabda          |      17 |          6.8 |

## component_type distribution

| component_type   |   count |
|:-----------------|--------:|
| Premise          |      92 |
| MajorClaim       |      79 |
| Claim            |      79 |

## component_type × final_pramana

| component_type   |   Pratyaksha |   Anumana |   Upamana |   Shabda |   All |
|:-----------------|-------------:|----------:|----------:|---------:|------:|
| Claim            |            2 |        58 |        10 |        9 |    79 |
| MajorClaim       |            0 |        69 |         7 |        3 |    79 |
| Premise          |            0 |        68 |        19 |        5 |    92 |
| All              |            2 |       195 |        36 |       17 |   250 |

## Labeling status

- `final_pramana` initialized from `candidate_pramana`
- `strength_label` and `reviewed` left blank for manual annotation
