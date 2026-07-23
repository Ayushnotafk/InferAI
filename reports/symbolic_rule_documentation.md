# Symbolic Rule Engine

This document describes the InferAI symbolic reasoning layer as implemented in
`classification/hybrid_reasoning.py` and inventoried in `resources/rules_v2.yaml`.
It does **not** modify the engine, rules, or model behaviour; it documents the
existing rule base for professor review and paper writing.

## Scope: Nyāya-inspired, not full Nyāya

The symbolic engine is a **lightweight computational operationalization** of
Nyāya-inspired reasoning. It detects surface cue families associated with the
four pramāṇas (Pratyaksha, Anumana, Upamana, Shabda) and produces a soft
distribution that is fused with machine-learning predictions.

It is **not** a claim of full computational Nyāya: there is no Sanskrit formalization,
no classical *hetvābhāsa* fallacy engine, and no dialectical debate protocol.

## Evolution of the rule base

| Stage | Rule style | Approximate count |
|-------|------------|------------------:|
| Previous implementation | Binary / boolean keyword hits | ~28 |
| Current implementation | Weighted semantic regex cues | **77** |

Source: `reports/symbolic_rule_improvements.md`, `resources/rules_v2.yaml`
(`version: 2.0`, `total_rules: 77`, generated from `_RULE_SPECS`).

## Rules grouped by pramāṇa

| Pramāṇa | Number of Rules | Average Weight |
|---------|----------------:|---------------:|
| Pratyaksha | 23 | 1.77 |
| Anumana | 17 | 2.02 |
| Upamana | 13 | 2.31 |
| Shabda | 24 | 2.23 |
| **Total** | **77** | **2.06** |

Weights range from **0.7** (informal testimony) to **3.4** (official “according to”
institutions). Overall average weight: **2.06** (computed from
`classification/hybrid_reasoning.py::_RULE_SPECS`).

## Weighted scoring

Each matching cue adds its weight to a per-pramāṇa raw score. When multiple distinct
cues fire for the same pramāṇa, a diminishing multi-cue bonus is applied:

\[
\mathrm{bonus}=\min(0.36,\ 0.12\cdot(n-1)),\quad
s' = s\cdot(1+\mathrm{bonus}).
\]

Rule probabilities use a soft floor and sharpening:

\[
\mathrm{score}_c = 1 + (s'_c)^{1.35},
\]

with a mix penalty \(\times 0.88\) if three or more pramāṇa families fire, then
normalization to \(p_{\mathrm{rules}}\).

## Hybrid fusion with ML predictions

Default fusion (README; `hybrid_fuse`):

\[
p_{\mathrm{fused}} = \mathrm{normalize}\bigl(0.8\,p_{\mathrm{ML}} + 0.2\,p_{\mathrm{rules}}\bigr).
\]

Symbolic rules do **not** participate in gradient training; they adjust inference-time
predictions and confidence only.

## Agreement bonus and confidence adjustment

Let \(c_0 = 100\cdot\max_y p_{\mathrm{fused}}(y)\). When symbolic cues are active:

- **Agreement** (ML argmax = rule argmax): \(c=\min(99.5,\ 1.10\,c_0+4.0)\).
- **Disagreement**: \(c=\max(5.0,\ 0.90\,c_0)\).

Constants: `AGREEMENT_CONF_BOOST=1.10`, `AGREEMENT_CONF_ADD=4.0`,
`DISAGREEMENT_CONF_MULT=0.90`.

## False-positive prevention

Implemented guards in `score_weighted_rules()`:

- **Upamana:** phrase-level analogy only; suppress `like a/an` when “I/we like …” patterns appear.
- **Shabda:** suppress informal “he/she studies …” unless research context co-occurs.
- **Shabda:** tiered `according to` so specific institutional patterns preempt the general pattern.
- Longer / more specific patterns are evaluated first (`_SORTED_RULES`).

## Context-aware matching

Matching is case-insensitive regex search over the raw input string. Cue metadata
(cue name, weight, matched span) is returned for interpretability. Multi-family
firing triggers the mix penalty to encode the annotation guideline that short spans
usually have one *dominant* epistemic mode.

## Representative rules (from `resources/rules_v2.yaml`)

| ID | Pramāṇa | Cue | Weight | Pattern (abbrev.) |
|----|---------|-----|-------:|-------------------|
| PRATYAKSHA_002 | Pratyaksha | measured | 1.8 | `\bmeasured\b` |
| PRATYAKSHA_020 | Pratyaksha | we measured | 2.0 | `\bwe measured\b` |
| ANUMANA_002 | Anumana | therefore | 2.3 | `\btherefore\b` |
| ANUMANA_010 | Anumana | if...then | 2.4 | `\bif\b.{0,40}\bthen\b` |
| UPAMANA_006 | Upamana | analogous to | 2.6 | `\banalogous to\b` |
| UPAMANA_013 | Upamana | like a/an | 2.0 | `\blike a\b\|\blike an\b` |
| SHABDA_001 | Shabda | according to official body | 3.4 | `\baccording to\s+(?:who\|…)\b` |
| SHABDA_003 | Shabda | according to informal source | 0.7 | `\baccording to\s+(?:my friend\|…)\b` |
| SHABDA_010 | Shabda | WHO | 3.2 | `\bWHO\b` |

Full inventory: `resources/rules_v2.yaml` (77 rules). Statistics: `reports/rule_statistics.md`.

## Validation

| Check | Result |
|-------|--------|
| Rules in `_RULE_SPECS` | 77 |
| Rules in `rules_v2.yaml` | 77 |
| Invented rules | None — YAML is an extraction of the implementation |

---

*Documentation only. Inference logic unchanged.*
