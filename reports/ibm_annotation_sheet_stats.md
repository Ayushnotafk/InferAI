# IBM ArgKP Nyaya Annotation Sheet — Statistics

Generated from `ibm_argkp_extracted.csv` with stratified sampling (target=600, random_state=42).

## Summary

- **Sampled rows:** 600
- **Unique topics:** 41
- **Average argument length (chars):** 103.0
- **Near-duplicates removed during sampling:** 0

## Stance distribution

| stance | count |
| --- | --- |
| support | 300 |
| oppose | 300 |

## candidate_pramana distribution

| candidate_pramana | count | percentage |
| --- | --- | --- |
| Pratyaksha | 2 | 0.33 |
| Anumana | 547 | 91.17 |
| Upamana | 8 | 1.33 |
| Shabda | 43 | 7.17 |

## Length buckets (sampled)

| length_bucket | count |
| --- | --- |
| short | 208 |
| long | 199 |
| medium | 193 |

## Output columns

```text
source, topic, stance, text, candidate_pramana, candidate_reason, final_pramana, strength_label, reviewed
```
