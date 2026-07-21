# Symbolic Rule Statistics (v2)

Computed from `resources/rules_v2.yaml`, which is an extraction of the current rule base in `classification/hybrid_reasoning.py`.

## Overview

- **Total rules:** 77
- **Pramāṇa families:** 4 (Pratyaksha, Anumana, Upamana, Shabda)

## Rules per pramāṇa

| Pramāṇa | Rules |
|--------|------:|
| Pratyaksha | 23 |
| Anumana | 17 |
| Upamana | 13 |
| Shabda | 24 |

## Weight distribution

Weights are the per-cue contributions used in `score_weighted_rules()` before multi-cue bonus and hybrid fusion.

- **Minimum weight:** 0.7
- **Maximum weight:** 3.4
- **Average weight (overall):** 2.06

### By pramāṇa (average weight)

| Pramāṇa | Avg Weight |
|--------|-----------:|
| Pratyaksha | 1.77 |
| Anumana | 2.02 |
| Upamana | 2.31 |
| Shabda | 2.23 |

## Cue-type distribution

In the implementation, the field is named `cue` and is documented here as `cue_type`.

- **Unique cue types:** 77 (one per rule entry in `_RULE_SPECS`)
- **Most frequent cue-type family patterns:** n/a (cue types are not reused as categorical bins; they are descriptive labels)

## Summary

The current symbolic engine is intentionally compact (77 cues) and uses **weighted** evidence rather than binary rule hits. It includes explicit false-positive prevention mechanisms in-code (e.g., suppressing Upamana false positives for “I like …” and suppressing Shabda false positives for informal “he studies …”).

