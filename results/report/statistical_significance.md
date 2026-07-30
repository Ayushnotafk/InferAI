# Statistical Significance Analysis

Paired comparisons on n=200 held-out samples. Primary test: McNemar (alpha=0.05). Permutation p-values are reported as a sensitivity check.

| Comparison | Acc A | Acc B | McNemar p | Significant | Permutation p |
|------------|-------|-------|-----------|-------------|---------------|
| ml vs hybrid | 0.600 | 0.765 | <0.0001 | Yes | 0.0005 |
| ml vs adaptive | 0.600 | 0.765 | <0.0001 | Yes | 0.0005 |
| symbolic vs hybrid | 0.550 | 0.765 | <0.0001 | Yes | 0.0005 |
| hybrid vs adaptive | 0.765 | 0.765 | 1.0000 | No | 1.0000 |

## Interpretation notes

- **McNemar** tests whether discordant prediction errors differ between two systems.
- A significant result (p < 0.05) indicates one system corrects errors the other misses more often than chance.
- Bootstrap CIs for Accuracy / Precision / Recall / Macro F1 are in `bootstrap_confidence_intervals.csv`.
