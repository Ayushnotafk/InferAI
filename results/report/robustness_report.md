# Robustness Evaluation Report

Clean hybrid accuracy: **1.000** (n=50).
Worst perturbation: **typo** (mean absolute drop=0.004).

| Perturbation | Accuracy | Δ Acc | Relative drop % |
|--------------|----------|-------|-----------------|
| clean | 1.000 | +0.000 | 0.0 |
| typo | 0.980 | -0.020 | 2.0 |
| grammar | 1.000 | +0.000 | -0.0 |
| word_order | 1.000 | +0.000 | -0.0 |
| synonym | 1.000 | +0.000 | -0.0 |
| paraphrase | 1.000 | +0.000 | -0.0 |

## Notes

Perturbations are heuristic and API-free (no LLM paraphraser). They stress lexical cue sensitivity of the symbolic channel and embedding stability of the ML head.
