# IBM ArgKP — Extraction Statistics

Source file: `dataset/external/IBM_ARGKP/ArgKP_combined.csv`

## Summary

- **Total rows (raw):** 36800
- **Rows after cleaning:** 8639
- **Duplicate arguments removed:** 28161
- **Missing / empty topic or argument rows skipped:** 0
- **Whitespace-only rows removed (post-normalize):** 0
- **Unique topics:** 41
- **Average argument length (chars):** 109.1

## Stance distribution

| stance | count |
| --- | ---: |
| support | 4550 |
| oppose | 4089 |

## Top 20 topics

| topic | count |
| --- | ---: |
| Routine child vaccinations should be mandatory | 279 |
| We should subsidize space exploration | 247 |
| Assisted suicide should be a criminal offence | 246 |
| We should legalize cannabis | 246 |
| We should ban human cloning | 245 |
| Homeschooling should be banned | 244 |
| We should ban the use of child actors | 244 |
| We should introduce compulsory voting | 244 |
| We should legalize sex selection | 244 |
| We should close Guantanamo Bay detention camp | 241 |
| We should abandon the use of school uniform | 238 |
| We should adopt libertarianism | 237 |
| We should prohibit women in combat | 237 |
| We should abolish capital punishment | 236 |
| We should abandon marriage | 236 |
| The vow of celibacy should be abandoned | 234 |
| We should adopt an austerity regime | 234 |
| Social media platforms should be regulated by the government | 233 |
| We should abolish the right to keep and bear arms | 233 |
| We should fight urbanization | 233 |

## Output schema

```text
source, topic, text, stance
```
