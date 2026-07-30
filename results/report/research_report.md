# InferAI Research Report

Auto-generated publication bundle. See `reports/alpha_investigation.md` for fusion analysis.

**Default fixed hybrid α:** 0.2

## Benchmark comparison

| Mode | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | ECE | Mean α |
|------|----------|---------|---------|----------|-------------|-----|--------|
| ml | 0.600 | 0.648 | 0.605 | 0.601 | 0.605 | 0.142 | 1.000 |
| symbolic | 0.550 | 0.832 | 0.570 | 0.571 | 0.572 | 0.176 | 0.000 |
| hybrid | 0.765 | 0.874 | 0.765 | 0.762 | 0.769 | 0.346 | 0.200 |
| adaptive | 0.765 | 0.874 | 0.765 | 0.762 | 0.769 | 0.235 | 0.431 |

## Alpha ablation

| α | Accuracy | Macro F1 | ECE |
|---|----------|----------|-----|
| 0.0 | 0.550 | 0.571 | 0.176 |
| 0.2 | 0.765 | 0.762 | 0.346 |
| 0.4 | 0.740 | 0.731 | 0.275 |
| 0.6 | 0.665 | 0.655 | 0.143 |
| 0.8 | 0.610 | 0.609 | 0.128 |
| 1.0 | 0.600 | 0.601 | 0.142 |

## Adaptive routing

- Mean α: 0.43075
- Min / Max: 0.05 / 0.8
- ML dominated: 48.0%
- Rules dominated: 31.0%
- Balanced: 21.0%

Adaptive routing improves over fixed high-ML hybrids by lowering alpha when symbolic cues are present (recovering rule-correct / ML-wrong cases) and raising alpha when cues are absent so ML remains the decision maker under the soft rule floor. Empirical mean alpha=0.431 (base=0.2).

## Error analysis

Most frequent confusion: Shabda → Pratyaksha (25 cases). Rules rescued 33 ML errors; fusion overrode 0 correct ML decisions; 47 residual ML failures; 0 symbolic-aligned residual errors.

## Fallacy detector

- Binary precision: 0.857
- Binary recall: 0.857
- FP / FN: 1 / 1

## Artifacts

All CSV tables, PNG/PDF figures, and supporting Markdown live in this directory.
Combined figure PDF: `figures_bundle.pdf`.

## Reproduce

```bash
python scripts/freeze_model.py --tag research
python scripts/generate_research_report.py
python scripts/generate_research_report.py --include-llm  # requires API keys
```
