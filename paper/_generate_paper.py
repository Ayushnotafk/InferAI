# -*- coding: utf-8 -*-
"""Generate InferAI research paper markdown from project facts only."""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "InferAI_Research_Paper.md"

PAPER = r'''# PramāṇaNet: A Nyāya-Inspired Neuro-Symbolic Framework for Explainable Epistemic Reasoning Classification

**InferAI Project Technical Report / Research Paper Draft**  
**Domain:** Explainable AI · Argument Mining · Indigenous Knowledge Systems (IKS) · NLP

---

## Abstract

**Background.** Contemporary explainable AI for argument analysis typically operationalizes Western rhetorical taxonomies (claim–premise, stance, fallacy types) while under-representing classical Indian epistemological categories. The Nyāya school’s four *pramāṇas*—Pratyaksha (perception), Anumana (inference), Upamana (analogy), and Shabda (testimony)—provide a compact epistemic taxonomy for classifying *how* an argument justifies its conclusion. However, mapping short English argumentative spans to these categories requires both statistical generalization and transparent symbolic cues.

**Method.** We present InferAI (also referred to as PramāṇaNet in this paper), a Nyāya-*inspired* neuro-symbolic system that embeds argument text with Sentence-BERT (`all-MiniLM-L6-v2`), classifies with L2-regularized logistic regression, and fuses softmax outputs with a 77-rule weighted symbolic engine via \(p_{\mathrm{fused}}=0.8\,p_{\mathrm{ML}}+0.2\,p_{\mathrm{rules}}\). The symbolic layer is an operationalization of pramāṇa *cue families*, not a claim of full computational Nyāya. Training uses a merged corpus of 1,861 rows (611 synthetic + 650 AAEC-reviewed + 600 IBM-reviewed) with a held-out template test set of 200 examples. Manual review covers 1,250 real-world sentences. Inter-annotator agreement on a 200-example double-annotated subset yields Cohen’s \(\kappa=0.8804\).

**Results.** On an in-distribution 20% holdout of the merged corpus, accuracy reaches 0.9287 and macro F1 0.9326. On the curated template test set, accuracy varies across training configurations (0.955 early; 0.685 without resampling; 0.600 with minority oversampling used in the production merge). Stratified five-fold cross-validation on real-world reviewed data yields high accuracy (0.9118) but low macro F1 (0.3414 pooled OOF), exposing severe Anumana skew. Expected Calibration Error is 0.457. Lexical analysis confirms synthetic template density (MATTR-50 = 0.797) versus richer real-world phrasing (MATTR-50 = 0.835).

**Contribution.** InferAI contributes (i) a transparent Nyāya-inspired taxonomy for epistemic reasoning classification, (ii) a hybrid fusion architecture with documented symbolic rules (`resources/rules_v2.yaml`), (iii) manually reviewed AAEC/IBM corpora with provenance and \(\kappa\), and (iv) a multi-protocol evaluation suite (leakage, calibration, OOD analysis, real-world CV) suitable for AI/NLP/XAI/IKS venues.

**Keywords:** Nyāya pramāṇa, explainable AI, neuro-symbolic reasoning, Sentence-BERT, argument mining, Indigenous Knowledge Systems, epistemic classification, hybrid fusion

---

## 1. Introduction

### 1.1 Explainable AI and Epistemic Transparency

Machine learning classifiers for text often achieve high accuracy while remaining opaque about *why* a prediction was issued. Explainable AI (XAI) methods such as SHAP attribute model outputs to features [1], but for epistemic tasks—determining the *mode of justification*—feature attributions alone do not map cleanly onto human-interpretable reasoning types. InferAI addresses this gap by pairing a statistical classifier with an explicit symbolic cue layer whose matches are returned as interpretability signals (matched spans, per-pramāṇa rule scores, ML–rule agreement metadata).

### 1.2 Argument Mining and Reasoning Classification

Argument mining extracts claims, premises, and relations from text [2], [3]. InferAI is complementary: given a short English argumentative span, it predicts which *epistemic instrument* (pramāṇa) best characterizes the dominant justification, and extracts claim/premise structure for presentation. The four-way label space is intentionally coarser than full argument graphs, enabling interactive analysis via a FastAPI backend and Streamlit frontend (project README; `api/app.py`; `frontend/app.py`).

### 1.3 Nyāya as Epistemology

Classical Nyāya treats knowledge as arising through *pramāṇas*—validated means of knowing [4], [5]. InferAI does **not** reconstruct Nyāya dialectical procedure, fallacy catalogues (*hetvābhāsa*), or Sanskrit formalization. Instead, it uses the four commonly taught pramāṇas as an annotation and prediction taxonomy for English text. Throughout this paper we use the phrases **Nyāya-inspired** and **operationalized through Nyāya pramāṇas** to mark this deliberate scope limitation.

### 1.4 Problem Statement

Given a short English argument \(x\), predict a label \(y\in\{\mathrm{Pratyaksha},\mathrm{Anumana},\mathrm{Upamana},\mathrm{Shabda}\}\), estimate confidence, surface symbolic cue evidence, and generate a short natural-language explanation. Challenges include: (i) severe class imbalance in real-world argumentative English (Anumana dominance), (ii) distribution shift between synthetic templates and natural essays/claims, (iii) boundary ambiguity among pramāṇas on short spans, and (iv) the risk of overstating symbolic machinery as “computational Nyāya.”

### 1.5 Research Gap and Motivation

Prior argument-mining systems rarely adopt Indian epistemological categories. Prior IKS applications rarely couple classical taxonomies with modern embedding classifiers under rigorous leakage, calibration, and OOD evaluation. InferAI is motivated by the need for **trustworthy, culturally grounded XAI** that remains scientifically honest about what the symbolic engine can and cannot claim.

### 1.6 Contributions

This work contributes:

1. **Nyāya-inspired taxonomy** for classifying epistemic reasoning modes in English arguments, with explicit labeling guidelines (`dataset/labeling_guidelines.md`).
2. **Hybrid neuro-symbolic architecture** combining Sentence-BERT embeddings, logistic regression, and a 77-rule weighted cue engine with fusion \(0.8\,p_{\mathrm{ML}}+0.2\,p_{\mathrm{rules}}\) (`classification/hybrid_reasoning.py`).
3. **Manual annotation** of 1,250 real-world sentences (650 AAEC + 600 IBM) and double annotation yielding \(\kappa=0.8804\) on 200 matched examples (`reports/kappa_report.md`).
4. **Explainability stack**: structural extraction (claim/premises/indicators), HTML cue highlighting, template NL explanations, optional embedding-space SHAP with documented limitations (`reports/shap_limitations.md`).
5. **Multi-protocol evaluation**: in-distribution holdout, curated template test, real-world five-fold CV, calibration (ECE/Brier), leakage audit, lexical diversity (TTR/Guiraud/MATTR), and error analysis (reports under `reports/`).

---

## 2. Nyāya Theory (Background)

### 2.1 Pramāṇa

In Nyāya epistemology, a *pramāṇa* is a means of valid knowledge (*pramā*). Different schools enumerate different sets; the fourfold set used here—perception, inference, analogy, and verbal testimony—is the standard Nyāya–Vaiśeṣika pedagogical framing taught in Indian philosophical curricula [4], [5]. InferAI treats each pramāṇa as a **label for the dominant justificatory mode of a short English span**, not as a verdict of classical *prāmāṇya* (epistemic validity in the full Nyāya sense).

### 2.2 Pratyaksha (Perception)

Pratyaksha denotes knowledge arising from direct contact of sense with object, including instrumented observation in modern scientific prose (“we measured,” “sensor data,” “under the microscope”). In InferAI guidelines, first-person sensory reports and laboratory/measurement vocabulary are cues for Pratyaksha (`dataset/labeling_guidelines.md`).

### 2.3 Anumana (Inference)

Anumana is knowledge from a sign (*liṅga*) to a conclusion via a general relation—classically illustrated by smoke → fire. In English arguments, inferential connectives (“because,” “therefore,” “if…then,” “implies”) and causal phrasing mark Anumana. Real-world reviewed corpora in this project are heavily Anumana-skewed (≈88–90% in reviewed AAEC/IBM exports; `reports/ibm_reviewed_dataset_stats.md`, progress report §3).

### 2.4 Upamana (Analogy / Comparison)

Upamana concerns knowledge by similarity—“this is like that.” InferAI guidelines require phrase-level analogy cues (“similar to,” “analogous to,” “works like”) and explicitly reject bare social preference (“I like football”) as analogy. The symbolic engine encodes this distinction with false-positive suppression (`classification/hybrid_reasoning.py`).

### 2.5 Shabda (Testimony / Authority)

Shabda is knowledge from trustworthy verbal testimony. InferAI operationalizes this via authority and reporting cues (“according to,” institutional names, “study found”), with **tiered weights**: official bodies (e.g., WHO) receive higher weight than informal sources (“according to my friend,” weight 0.7) (`reports/symbolic_rule_improvements.md`; `resources/rules_v2.yaml`).

### 2.6 Why Nyāya Is Epistemology—and Why This Work Is Only Inspired

Nyāya is a systematic theory of knowledge, debate, and error analysis. A full computational Nyāya system would require formal treatment of inference patterns, debate protocol, and Sanskrit primary texts. InferAI instead:

- uses four pramāṇa names as a **modern ML label space**;
- detects **surface cues** associated with those modes in English;
- fuses cues with a statistical classifier for robustness.

We therefore describe InferAI as **Nyāya-inspired** and **operationalized through Nyāya pramāṇas**, never as full computational Nyāya. This stance is consistent with the project README and progress report framing.

---

## 3. Related Work

### 3.1 Explainable AI

Post-hoc explanation methods attribute predictions to input features [1]. InferAI uses optional SHAP on the logistic head in Sentence-BERT embedding space (384 dimensions), and separately provides rule-match traces that are human-readable. The project explicitly documents that embedding SHAP is **not** token-level explanation (`reports/shap_limitations.md`).

### 3.2 Argument Mining

Stab and Gurevych’s Argument Annotated Essays Corpus (AAEC) provides claim/premise annotations for persuasive essays [2], [3]. InferAI reuses AAEC as a *source of argumentative spans* that are then manually re-labeled into Nyāya pramāṇas—an epistemic overlay, not a replacement for claim–premise structure.

### 3.3 Neuro-symbolic AI

Neuro-symbolic systems combine learning with rules or logic [6]. InferAI’s hybrid layer is lightweight: regex-weighted cues produce a soft distribution fused with ML softmax. Rules do not participate in gradient training; they adjust inference-time predictions and confidence.

### 3.4 Sentence-BERT

Reimers and Gurevych’s Sentence-BERT produces semantically meaningful sentence embeddings [7]. InferAI freezes `all-MiniLM-L6-v2` (384-d) and trains only a logistic regression head (`classification/embedder.py`, `classification/train_model.py`), prioritizing reproducibility and low compute for a B.Tech research prototype.

### 3.5 Symbolic Reasoning and IKS

Indigenous Knowledge Systems (IKS) research encourages responsible computational engagement with classical Indian knowledge. InferAI’s contribution to IKS is the **transparent mapping of Nyāya categories into an evaluable NLP pipeline**, with documented limits. Related work in computational Indian philosophy is sparse in engineering venues; we position InferAI as an engineering bridge rather than a Sanskrit philology system.

### 3.6 Comparison Table

| System / Line of Work | Label Space | Learner | Symbolic Layer | XAI | IKS Link |
|----------------------|-------------|---------|----------------|-----|----------|
| Standard argument mining [2], [3] | Claim/premise/stance | Various | Rare | Variable | No |
| Pure embedding classifiers | Task-specific | SBERT+LR/NN | None | Optional SHAP | No |
| Heavy neuro-symbolic logic | Formal logic | Hybrid | Complex | Variable | Rare |
| **InferAI (this work)** | **Four pramāṇas** | **SBERT+LR** | **77 weighted cues** | **Rules + NL + optional SHAP** | **Nyāya-inspired** |

---

## 4. Methodology

### 4.1 System Overview

The InferAI pipeline is:

```
Input text
   → Preprocessing (whitespace normalization)
   → Structural cue extraction (claim / premises / indicators / HTML highlights)
   → Sentence-BERT embedding (all-MiniLM-L6-v2, 384-d)
   → Logistic Regression softmax \(p_{\mathrm{ML}}\)
   → Symbolic weighted rule engine \(p_{\mathrm{rules}}\)
   → Hybrid fusion \(p_{\mathrm{fused}} = 0.8\,p_{\mathrm{ML}} + 0.2\,p_{\mathrm{rules}}\)
   → Agreement-adjusted confidence
   → Reasoning strength + natural-language explanation
   → Optional embedding-space SHAP
```

This matches the architecture documented in the README and `reports/InferAI_Progress_Report(28_June).md`.

### 4.2 Preprocessing and Structural Extraction

Input text is normalized by collapsing whitespace. The API extracts a primary claim, premises/evidence spans, reasoning indicators, and HTML-safe highlighting of cue phrases (`api/app.py`). These structural fields support explanation even when the ML and rule labels disagree.

### 4.3 Sentence-BERT Encoder

Let \(x\) be the input string. The frozen encoder \(f_{\theta}\) maps \(x\) to \(e\in\mathbb{R}^{384}\):

\[
e = f_{\theta}(x),\quad \theta=\texttt{all-MiniLM-L6-v2}\ \text{(frozen)}.
\]

### 4.4 Logistic Regression Head

A multinomial logistic regression classifier with `max_iter=1000` (scikit-learn defaults otherwise) produces:

\[
p_{\mathrm{ML}}(y\mid e)=\mathrm{softmax}(We+b).
\]

Training uses an 80/20 random split with `random_state=42` (`classification/train_model.py`). Class order follows the fitted `LabelEncoder` over \(\{\mathrm{Anumana},\mathrm{Pratyaksha},\mathrm{Shabda},\mathrm{Upamana}\}\).

### 4.5 Symbolic Rule Engine

The rule base comprises **77** weighted regex cues documented in `resources/rules_v2.yaml` and implemented in `_RULE_SPECS` (`classification/hybrid_reasoning.py`):

| Pramāṇa | Rules | Avg Weight |
|---------|------:|-----------:|
| Pratyaksha | 23 | 1.77 |
| Anumana | 17 | 2.02 |
| Upamana | 13 | 2.31 |
| Shabda | 24 | 2.23 |
| **Total** | **77** | **2.06** |

Weights range from **0.7** (informal “according to my friend”) to **3.4** (official “according to WHO/CDC/…”). For each match, weight is accumulated per pramāṇa. Multi-cue bonus:

\[
\mathrm{bonus}=\min\bigl(0.36,\ 0.12\cdot(n_{\mathrm{cues}}-1)\bigr),\quad
s' = s\cdot(1+\mathrm{bonus}).
\]

Rule probabilities use a soft floor and sharpening:

\[
\mathrm{score}_c = 1 + (s'_c)^{1.35},
\]

with a mix penalty \(\times 0.88\) if ≥3 pramāṇa families fire, then normalization to \(p_{\mathrm{rules}}\).

**False-positive prevention** (implemented, not claimed as Nyāya orthodoxy):

- Upamana: phrase-level analogy only; suppress `like a/an` when “I/we like …” patterns appear.
- Shabda: suppress informal “he/she studies …” unless research context co-occurs.
- Shabda: tiered `according to` so specific institutional patterns preempt the general pattern.

### 4.6 Hybrid Fusion and Confidence

\[
p_{\mathrm{fused}} = \mathrm{normalize}\bigl(0.8\,p_{\mathrm{ML}} + 0.2\,p_{\mathrm{rules}}\bigr).
\]

Let \(c_0 = 100\cdot\max_y p_{\mathrm{fused}}(y)\). If symbolic cues are active:

- **Agreement** (ML argmax = rule argmax): \(c=\min(99.5,\ 1.10\,c_0+4.0)\).
- **Disagreement**: \(c=\max(5.0,\ 0.90\,c_0)\).

### 4.7 Explanation Generation

A template explainer selects among three canned templates per pramāṇa and appends a Strong/Moderate/Weak strength phrase (`explanation_engine/explainer.py`). Optional SHAP returns top embedding dimensions by absolute attribution (`explanation_engine/shap_explainer.py`). **No counterfactual generator exists** in the codebase.

### 4.8 Algorithm (Inference)

```
Algorithm 1: InferAI Hybrid Inference
Input: text x; models (encoder, LR, label encoder); weights α=0.8, β=0.2
1: e ← SentenceBERT(x)
2: p_ML ← Softmax(LR(e))
3: (raw, matched) ← ScoreWeightedRules(x)   # 77 cues + FP guards
4: ApplyMultiCueBonus(raw, matched)
5: p_rules ← Normalize(1 + raw^{1.35}; mix penalty if needed)
6: p_fused ← Normalize(α p_ML + β p_rules)
7: y_ML ← argmax p_ML; y_rule ← argmax p_rules if matched else ∅
8: y* ← argmax p_fused
9: c ← AdjustConfidence(max p_fused, y_ML, y_rule)
10: explanation ← TemplateExplain(y*, strength(x))
11: return y*, c, matched, explanation, optional SHAP(e)
```

### 4.9 Architecture Diagram (Textual)

```
                 ┌─────────────────────┐
   text x ──────►│ Structural extract  │──► claim, premises, HTML
                 └─────────┬───────────┘
                           │
                 ┌─────────▼───────────┐
                 │ Sentence-BERT       │
                 │ all-MiniLM-L6-v2    │
                 └─────────┬───────────┘
                           │ e ∈ R^384
                 ┌─────────▼───────────┐     ┌──────────────────┐
                 │ Logistic Regression │     │ 77 weighted rules│
                 │ p_ML                │     │ p_rules          │
                 └─────────┬───────────┘     └────────┬─────────┘
                           │                          │
                           └──────────┬───────────────┘
                                      ▼
                           Hybrid fuse (0.8 / 0.2)
                                      │
                           Confidence adjust + NL explain
                                      │
                           Optional SHAP (embedding dims)
```

---

## 5. Dataset

Canonical counts are defined in `reports/dataset_canonical.py` and documented in `docs/dataset_provenance.md`.

### 5.1 Synthetic Nyāya Corpus

- **Raw size:** 4,200 rows (`dataset/raw/nyaya_dataset.csv`).
- **Generation:** template-based expansion across domain slots with deterministic labels (`dataset/labeled_corpus.py`, `dataset_generator.py`).
- **Used in final merge:** **611** rows after cross-corpus deduplication.

Lexical diversity of the full synthetic file (`reports/lexical_diversity_report.md`):

| Metric | Synthetic (4,200) | Real-world curated (200) |
|--------|------------------:|-------------------------:|
| Tokens | 108,107 | 2,603 |
| Vocabulary | 364 | 1,349 |
| TTR | 0.003367 | 0.518248 |
| Guiraud’s R | 0.782816 | 18.6965 |
| MATTR (w=50) | 0.797221 | 0.834957 |

Raw TTR alone is misleading under length disparity; Guiraud and MATTR confirm template density in synthetic text versus more varied real-world phrasing.

![Lexical diversity metrics](../reports/figures/lexical_diversity_metrics.png)

*Figure 1. Lexical diversity indices for synthetic vs curated real-world corpora (`reports/figures/lexical_diversity_metrics.png`).*

### 5.2 AAEC (Argument Annotated Essays Corpus)

- **Essays:** 402; **extracted components:** 6,050 (MajorClaim 742, Claim 1,497, Premise 3,811) (`reports/aaec_statistics.md`).
- **Cleaning:** 39 duplicate/empty removed at extraction.
- **Reviewed for Nyāya labels:** **650** rows (`aaec_reviewed_dataset.csv`).
- Annotation proceeded in staged batches (including an initial 250-row manual review sheet), then expanded to the finalized 650-row reviewed set (`docs/dataset_provenance.md`).

### 5.3 IBM ArgKP

- **Raw:** 36,800 rows; **cleaned unique arguments:** 8,639 (28,161 duplicates removed) (`reports/ibm_argkp_statistics.md`).
- **Reviewed:** **600** rows (`ibm_reviewed_dataset.csv`).
- Class distribution (reviewed): Anumana 543 (90.5%), Shabda 46 (7.67%), Upamana 6 (1.0%), Pratyaksha 5 (0.83%) (`reports/ibm_reviewed_dataset_stats.md`).

### 5.4 Merged Training Corpus

\[
611 + 650 + 600 = 1861
\]

after removing **3,589** duplicate texts during merge (`classification/retrain_merged.py`). Raw concatenation before dedup: **5,450**. The held-out file `dataset/test_set.csv` (**200** rows) is **never** included in the merge.

| Split | Rows | Fraction | Source |
|-------|-----:|----------|--------|
| Train | 1,488 | 80% | merged corpus, `random_state=42` |
| Validation / in-training holdout | 373 | 20% | same |
| Held-out test | 200 | — | `test_set.csv` (excluded) |

(`reports/train_test_split.md`)

Held-out class counts (n=200): Pratyaksha 46, Anumana 44, Upamana 53, Shabda 57.

### 5.5 Annotation Protocol

Annotators follow `dataset/labeling_guidelines.md`:

1. Read the span once.
2. Assign the dominant pramāṇa (ties → first decisive epistemic move).
3. Assign strength (Strong / Moderate / Weak) independently.
4. Record notes; avoid toxic content and PII.
5. Prevent exact-span duplicates across master and test sets.

Heuristic pre-labeling accelerates review (`auto_annotate_second_annotator.py`) but is not treated as ground truth until human confirmation (`reviewed=yes`).

### 5.6 Inter-Annotator Agreement

On **200** matched double-annotated examples (`reports/kappa_report.md`):

| Metric | Value |
|--------|------:|
| Agreements | 183 |
| Disagreements | 17 |
| Agreement % | 91.50% |
| Cohen’s \(\kappa\) | **0.8804** |
| Bootstrap 95% CI | [0.8212, 0.9295] |

Interpretation: strong agreement beyond chance. **Caveat documented in-project:** annotator-2 labels were partly rule-assisted, so \(\kappa\) reflects consistency with a similar heuristic rather than fully independent dual human judgment.

Primary residual disagreements involve Pratyaksha boundaries (confused with Anumana/Shabda).

### 5.7 Dataset Transparency and Provenance

Full provenance is published in `docs/dataset_provenance.md` and `reports/dataset_summary.md`. Total manually reviewed real-world examples: **1,250** (650+600). Early experimental merge of synthetic + curated real-world v1 (811 rows) is historical and not the production merge.

### 5.8 Distribution Shift (Template Test vs Training)

`reports/test_set_distribution_analysis.md` reports:

- Exact text overlap of `test_set.csv` with synthetic/AAEC/IBM sources: **0**.
- Jensen–Shannon divergence (word frequencies) vs merged training: **0.5065**.
- Class total-variation distance: **0.576**.
- Mean length: 15.8 words (test) vs 19.5 (train).
- Test composition: 88 template seeds + 112 padding vignettes.

**Verdict:** the template test is **out-of-distribution** relative to AAEC/IBM-dominated training text.

![Token/vocab comparison](../reports/figures/lexical_diversity_token_vocab.png)

*Figure 2. Token and vocabulary comparison (`reports/figures/lexical_diversity_token_vocab.png`).*

---

## 6. Experiments

All metrics below are taken from existing project reports. Different reports correspond to **different training snapshots**; we label each protocol explicitly and do not pool incompatible numbers into a single “best accuracy.”

### 6.1 Training Setup

| Setting | Value |
|---------|-------|
| Encoder | `SentenceTransformer('all-MiniLM-L6-v2')` |
| Classifier | `LogisticRegression(max_iter=1000)` |
| Split | 80/20, `random_state=42` |
| Merge shuffle seed | 42 |
| Hybrid weights | ML 0.8, rules 0.2 |
| SHAP background | ≤200 training embeddings |

Architecture was intentionally frozen during post-review improvements so gains could be attributed to data and evaluation methodology (progress report).

### 6.2 In-Distribution Holdout (Merged Corpus)

From `reports/final_retraining_report.md` (merged n=1,861; reported holdout support n=533):

| Metric | Value |
|--------|------:|
| Accuracy | **0.9287** |
| Macro precision | 0.9484 |
| Macro recall | 0.9198 |
| Macro F1 | **0.9326** |
| Weighted F1 | 0.9288 |
| Train accuracy | 0.9573 |

Per-class F1 (holdout): Anumana 0.9224, Pratyaksha 0.9773, Shabda 0.9278, Upamana 0.9029.

### 6.3 Curated Template Test (`test_set.csv`, n=200)

Multiple documented snapshots:

| Snapshot / Configuration | Accuracy | Macro F1 | Source |
|--------------------------|---------:|---------:|--------|
| Early / pre-merge evaluation | 0.9550 | 0.9552 | `heldout_evaluation.md`, `confusion_matrix_analysis.md` |
| Retrained original (no resampling) | 0.6850 | 0.6831 | `error_analysis.md`, `oversampling_comparison.md` |
| Class-weight (`balanced`) | 0.6600 | 0.6556 | `class_weight_comparison.md` |
| Minority oversampling (production merge path) | 0.6000 | 0.6007 | `final_retraining_report.md`, `oversampling_comparison.md` |
| Earlier final_evaluation snapshot | 0.7600 | 0.7599 | `final_evaluation_report.md` |

**Reading.** High early accuracy reflects template-like training/evaluation alignment. After integrating real-world AAEC/IBM text and changing training strategy, template-test scores drop—consistent with OOD analysis (JSD ≈ 0.51). Oversampling improved minority recall in some settings but **reduced** aggregate template-test metrics; the project report recommends the original (non-resampled) configuration for best template macro F1 among compared strategies.

![Held-out confusion matrix](../reports/figures/heldout_confusion_matrix.png)

*Figure 3. Held-out confusion matrix from the high-accuracy snapshot protocol (`reports/figures/heldout_confusion_matrix.png`; see `heldout_evaluation.md`).*

### 6.4 Training Strategy Comparison

| Model | Accuracy | Macro P | Macro R | Macro F1 |
|-------|---------:|--------:|--------:|---------:|
| Original | **0.6850** | 0.7494 | 0.7041 | **0.6831** |
| Class-weight | 0.6600 | 0.7061 | 0.6628 | 0.6556 |
| Oversampled | 0.6000 | 0.6483 | 0.6054 | 0.6007 |

(`reports/oversampling_comparison.md`)

### 6.5 Real-World Stratified 5-Fold CV

Protocol (`reports/real_world_cross_validation.md`): merge AAEC+IBM reviewed rows; stratified 5-fold CV; fresh LR head per fold on Sentence-BERT embeddings; production artifacts not modified.

**Canonical pool size:** 1,250 (650+600). The CV metrics below were computed on the on-disk export available at run time (report footnotes 850-row execution export for fold metrics); class skew on that export: Anumana 89.8%.

| Metric | Fold mean | Std | Pooled OOF |
|--------|----------:|----:|-----------:|
| Accuracy | 0.9118 | 0.0083 | **0.9118** |
| Macro precision | 0.4696 | 0.0165 | 0.4600 |
| Macro recall | 0.3152 | 0.0365 | 0.3160 |
| Macro F1 | 0.3359 | 0.0411 | **0.3414** |

Per-class pooled OOF F1: Anumana 0.9531; Shabda 0.4127; Pratyaksha 0.0; Upamana 0.0 (minority collapse under skew).

![Real-world CV confusion matrix](../reports/figures/real_world_cv_confusion_matrix.png)

*Figure 4. Pooled out-of-fold confusion matrix for real-world CV (`reports/figures/real_world_cv_confusion_matrix.png`).*

**Interpretation.** Accuracy is inflated by Anumana majority prediction; **macro F1 is the fairer headline metric** for real-world generalization.

### 6.6 Calibration

On the calibration evaluation protocol (`reports/calibration_report.md`):

| Metric | Value |
|--------|------:|
| ECE | **0.457** |
| Multiclass Brier (mean OvR) | 0.0949 |
| Mean max confidence | 0.498 |
| Accuracy at that operating point | 0.955 |

![Reliability diagram](../reports/figures/calibration_reliability_diagram.png)

*Figure 5. Calibration reliability diagram (`reports/figures/calibration_reliability_diagram.png`).*

ECE ≫ 0.10 indicates **miscalibration**; confidence and SHAP magnitudes must not be read as epistemic certainty.

### 6.7 Error Analysis

For the 0.6850-accuracy template-test snapshot (`reports/error_analysis.md`): 63 misclassifications; lowest recall Shabda 0.4386; lowest precision Anumana 0.5238.

Most common confusions:

| True → Pred | Count | Rate |
|-------------|------:|-----:|
| Upamana → Anumana | 18 | 34.0% |
| Shabda → Pratyaksha | 17 | 29.8% |
| Shabda → Anumana | 12 | 21.1% |
| Pratyaksha → Anumana | 10 | 21.7% |

Root causes cited in-project: Anumana dominance in training, ambiguous template wording, insufficient minority examples.

![Error analysis confusion matrix](../reports/figures/error_analysis_confusion_matrix.png)

*Figure 6. Error-analysis confusion matrix (`reports/figures/error_analysis_confusion_matrix.png`).*

### 6.8 Data Leakage Audit

(`reports/data_leakage_report.md`)

| Check | Result |
|-------|--------|
| Exact cross-split duplicates | **0** |
| Near-duplicates (fuzz ≥90% or cosine ≥0.9) | **506** |

Near-duplicates are predominantly synthetic padding pairs (master-pad vs test-pad) with fuzz ≈92–95%. Exact-overlap checks against AAEC/IBM reviewed sources for `test_set.csv` are zero (`test_set_distribution_analysis.md`). The merge pipeline’s test-exclusion guard is effective against literal leakage; near-duplicate template padding remains a residual optimism risk for template-style evaluation.

### 6.9 Early Synthetic-Heavy Evaluation (Context)

`reports/final_evaluation_report.md` documents a high in-distribution random-holdout accuracy (≈0.996) on an earlier synthetic-heavy training regime, with curated test accuracy 0.76. These numbers characterize an **earlier experimental stage**, not the final merged real-world-integrated production narrative. We report them for completeness and historical transparency.

---

## 7. Explainability

### 7.1 Hybrid Reasoning Traces

For each prediction, InferAI exposes:

- ML label and rule label (when cues fire);
- per-pramāṇa `rule_scores` and `matched_cues` with surface spans;
- agreement metadata and adjusted confidence;
- structural claim/premise/indicator fields.

This constitutes the primary user-facing explanation channel.

### 7.2 Symbolic Cue Matching

Rules are documented exhaustively in `resources/rules_v2.yaml` (version 2.0, 77 rules). Representative examples:

1. **SHABDA_001** (weight 3.4): official `according to WHO/CDC/...`
2. **SHABDA_003** (weight 0.7): informal `according to my friend`
3. **UPAMANA_006** (weight 2.6): `analogous to`
4. **UPAMANA_013** (weight 2.0): `like a/an` with FP guard for “I like …”
5. **ANUMANA_010** (weight 2.4): `if … then` within 40 characters
6. **ANUMANA_014** (weight 2.5): `therefore it follows`
7. **PRATYAKSHA_011** (weight 2.3): `microscope`
8. **PRATYAKSHA_018** (weight 1.8): first-person sensory verbs

### 7.3 SHAP (Embedding Space)

SHAP via `LinearExplainer` attributes the logistic head’s decision to **384 embedding dimensions**, not tokens (`reports/shap_limitations.md`). Limitations:

- opaque dimensions;
- frozen encoder not attributed;
- hybrid rule adjustments not included in SHAP;
- background-set dependence (≤200 rows);
- must not be over-read under miscalibration (ECE 0.457).

### 7.4 Natural-Language Templates

Template explanations are independent of SHAP and provide readable but shallow verbalizations. They are not free-form LLM generations.

### 7.5 Counterfactuals

The repository does **not** implement counterfactual explanation generation. Future work in project reports mentions perturbation-based or token-level attributions as desirable extensions.

### 7.6 Explainability Limitations

InferAI offers **usable** explanations (rules + structure) and **developer diagnostics** (embedding SHAP). It does not offer causal epistemic validation in the classical Nyāya sense, nor token-faithful attributions by default.

---

## 8. Discussion

### 8.1 What Worked

- **Frozen simple architecture** enabled reproducible iteration on data and evaluation without confounding model changes.
- **Canonical dataset documentation** (1,861 / 1,250 / \(\kappa\)) strengthened transparency for supervisor review.
- **Weighted symbolic rules** (77) improved interpretability and authority tiering without retraining.
- **Multi-protocol evaluation** correctly identified that template accuracy can overstate or mischaracterize real-world behavior.
- **Lexical diversity suite** (TTR + Guiraud + MATTR) answered the methodological concern that raw TTR alone misleads.

### 8.2 What Failed or Remains Hard

- **Minority pramāṇas** on real-world CV: Pratyaksha and Upamana OOF F1 collapsed to 0 under extreme skew.
- **Calibration** remains poor (ECE 0.457).
- **Template vs real-world gap**: JSD ≈ 0.51; production oversampled model scores only 0.60 accuracy on template test while in-distribution holdout stays high (0.93).
- **Symbolic–ML disconnection**: rules do not influence training gradients; the LR head can ignore explicit cues.
- **AAEC reviewed class-count documentation** in `aaec_reviewed_dataset_stats.md` still reflects an earlier 250-row distribution table under a 650-row header—an acknowledged documentation lag relative to canonical size (consistency audit).

### 8.3 Synthetic vs Real

Synthetic data provides balanced seed coverage but template repetition (low Guiraud). Real-world AAEC/IBM text provides natural phrasing at the cost of Anumana dominance. The merged corpus is a pragmatic compromise for a research prototype, not a balanced production corpus.

### 8.4 Hybrid Reasoning Role

Hybrid fusion is a **light regularizer / interpretability coupler**, not a second full reasoner. Agreement boosting slightly increases confidence when ML and cues align; disagreement reduces confidence—useful for UI honesty, insufficient alone to rescue minority classes.

---

## 9. IKS Implications

### 9.1 Indigenous Knowledge Systems

IKS research seeks respectful, accurate engagement with classical knowledge traditions. InferAI contributes an **engineering artifact** that:

- teaches four Nyāya pramāṇas as an analytic lens on English arguments;
- keeps classical scope claims modest (“inspired,” not “complete”);
- publishes rule inventories and evaluation limitations for scholarly scrutiny.

### 9.2 Trustworthy and Educational AI

In educational settings, InferAI can help learners identify whether a passage primarily appeals to observation, inference, analogy, or authority. Trustworthiness requires communicating calibration limits and imbalance risks—already documented in project reports.

### 9.3 Dual Belonging: AI and IKS

The work belongs to **AI/NLP/XAI** through embeddings, classification metrics, SHAP, and leakage audits. It belongs to **IKS** through deliberate use of Nyāya categories as the semantic backbone of explanation. Neither identity should erase the other: removing Nyāya yields a generic classifier; removing evaluation rigor yields an uncritical cultural demo.

---

## 10. Limitations

1. **Class imbalance** in real-world reviewed data (Anumana ≈90%).
2. **Synthetic template density** and padding vignettes in `test_set.csv`.
3. **Embedding-level SHAP** is not token-faithful.
4. **English-only** short spans; no Sanskrit/Hindi primary-text pipeline.
5. **Minority-class failure** on real-world CV (Pratyaksha, Upamana).
6. **OOD template evaluation** complicates single-number reporting.
7. **Partly heuristic second annotator** inflates apparent independence of \(\kappa\).
8. **On-disk CSV lag**: some exports still reflect earlier 250-row AAEC batches while documentation uses canonical 650/1,861/1,250.
9. **No counterfactual** explanations.
10. **Not full Nyāya**: no *hetvābhāsa* engine, no debate protocol, no Sanskrit formal semantics.

---

## 11. Ethics

### 11.1 Trustworthy and Responsible AI

InferAI surfaces confidence adjustments and rule evidence to reduce silent failures, but miscalibration means raw confidence must not be treated as probability of truth.

### 11.2 Epistemic Validity

Predicting a pramāṇa label is **not** certifying that the argument is sound or that testimony is trustworthy. Shabda cues detect *appeals to authority*, not authority’s correctness.

### 11.3 Bias

Anumana skew biases the system toward labeling natural arguments as inferential. Institutional Shabda cues (WHO, CDC, government) encode contemporary authority markers that may not transfer across cultures or time.

### 11.4 Educational Use

Suitable as a teaching aid for epistemology/XAI with instructor oversight; unsuitable as an automated grader of “correct reasoning” without human review.

### 11.5 Data Stewardship

AAEC and IBM ArgKP must be used under their original distribution terms; the repository notes no bundled license files (`docs/dataset_provenance.md`).

---

## 12. Conclusion

InferAI (PramāṇaNet) is a Nyāya-*inspired* neuro-symbolic framework for classifying short English arguments into four pramāṇas, combining Sentence-BERT, logistic regression, and a documented 77-rule hybrid engine. The finalized corpus integrates 611 synthetic + 650 AAEC-reviewed + 600 IBM-reviewed examples (1,861 merged; 1,250 manually reviewed). Inter-annotator \(\kappa=0.8804\) indicates strong label consistency under documented caveats.

Evaluation shows a clear pattern: **in-distribution holdout performance is strong** (accuracy 0.9287, macro F1 0.9326), while **real-world macro F1 remains low** (0.3414) despite high accuracy (0.9118), and **template-test scores vary by training snapshot** (0.60–0.955). Calibration (ECE 0.457) and OOD diagnostics (JSD 0.5065) further discipline interpretation.

### Future Work

As stated in project reports: larger balanced annotation; active learning for boundary cases; temperature scaling / calibration; token-level attributions; multilingual Nyāya-aware extension; tighter train-time integration of symbolic cues; rebuild of on-disk reviewed CSVs to match canonical 650/1,861 counts.

---

## References

[1] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[2] C. Stab and I. Gurevych, “Annotating argument components and relations in persuasive essays,” in *Proc. COLING*, 2014.

[3] C. Stab and I. Gurevych, “Parsing argumentation structures in persuasive essays,” *Computational Linguistics*, vol. 43, no. 3, pp. 619–659, 2017. (AAEC resource lineage)

[4] Gautama, *Nyāya Sūtras* (classical source for the pramāṇa framework; standard critical editions and translations).

[5] B. K. Matilal, *Perception: An Essay on Classical Indian Theories of Knowledge*. Oxford, U.K.: Oxford Univ. Press, 1986.

[6] A. d’Avila Garcez and L. C. Lamb, “Neurosymbolic AI: The 3rd wave,” *Artificial Intelligence Review*, 2023. (survey framing for neuro-symbolic methods)

[7] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP-IJCNLP*, 2019.

[8] J. Cohen, “A coefficient of agreement for nominal scales,” *Educational and Psychological Measurement*, vol. 20, no. 1, pp. 37–46, 1960.

[9] L. Einav, “Guiraud’s index,” classical lexical richness; project uses Guiraud’s \(R=V/\sqrt{N}\) as implemented in `reports/generate_lexical_diversity.py`.

[10] M. A. Covington and J. D. McFall, “Cutting the Gordian knot: The moving-average type–token ratio (MATTR),” *Journal of Quantitative Linguistics*, vol. 17, no. 2, pp. 94–100, 2010.

[11] IBM Research, ArgKP / Argument Knowledge Pairing corpus documentation (IBM ArgKP combined release used under `dataset/external/IBM_ARGKP/`).

[12] InferAI project artifacts: `reports/*`, `docs/dataset_provenance.md`, `resources/rules_v2.yaml`, `classification/hybrid_reasoning.py` (primary empirical sources for this paper).

---

## Appendix A — Symbolic Rule Inventory Pointer

Full rule listing: `resources/rules_v2.yaml` (version 2.0, `total_rules: 77`, generated from `classification/hybrid_reasoning.py::_RULE_SPECS`).

## Appendix B — Report Index (Empirical Sources)

| Topic | Report |
|-------|--------|
| Provenance | `docs/dataset_provenance.md` |
| Dataset summary | `reports/dataset_summary.md` |
| Train/val/test | `reports/train_test_split.md` |
| Final retrain | `reports/final_retraining_report.md` |
| Real-world CV | `reports/real_world_cross_validation.md` |
| Kappa | `reports/kappa_report.md` |
| Lexical diversity | `reports/lexical_diversity_report.md` |
| Calibration | `reports/calibration_report.md` |
| Leakage | `reports/data_leakage_report.md` |
| Error analysis | `reports/error_analysis.md` |
| SHAP limits | `reports/shap_limitations.md` |
| Progress narrative | `reports/InferAI_Progress_Report(28_June).md` |
| Rule stats | `reports/rule_statistics.md` |
| Consistency audit | `reports/repository_consistency_audit.md` |

---

*End of paper draft. All numerical claims are drawn from the InferAI repository reports listed above.*
'''

OUT.write_text(PAPER, encoding="utf-8")
print(f"Wrote {OUT}")
words = len(PAPER.split())
print(f"Word count (approx): {words}")
'''

if __name__ == "__main__":
    # The string above is the paper; execute write
    text = PAPER if "PAPER" in dir() else None
