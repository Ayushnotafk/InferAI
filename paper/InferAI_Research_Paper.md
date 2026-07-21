# PramāṇaNet: A Nyāya-Inspired Neuro-Symbolic Framework for Explainable Epistemic Reasoning Classification

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

### 1.7 Paper Organization

Section 2 reviews Nyāya pramāṇas and the inspired/operationalized scope. Section 3 situates InferAI among XAI, argument mining, neuro-symbolic AI, and IKS. Section 4 details methodology and fusion equations. Section 5 documents datasets, provenance, \(\kappa\), and lexical diversity. Section 6 reports experiments across protocols. Section 7 analyzes explainability. Sections 8–12 discuss results, IKS implications, limitations, ethics, and conclusions. Sections 13–20 provide implementation notes, extended interpretation, threats to validity, and reproducibility guidance.

### 1.8 Scope Boundaries (Non-Claims)

- InferAI does **not** decide whether an argument is true.
- InferAI does **not** implement classical debate victory conditions.
- InferAI does **not** provide token-level causal explanations by default.
- InferAI does **not** solve class imbalance; it measures it.
- A single accuracy number does **not** summarize the system; protocol matters.

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

### 2.7 Operational Definitions Used by Annotators

The project’s labeling guidelines (`dataset/labeling_guidelines.md`) translate classical categories into English annotation practice. We summarize those operational definitions here because they are the true semantic contract of the supervised labels.

**Pratyaksha (operational).** Dominant support is direct sensory evidence, measurement, or an immediately reportable state (“thermometer read…”, “through the microscope…”, “motor was audibly misfiring…”). Edge case: citing WHO is Shabda even if factual; “smoke therefore fire” is Anumana even if vivid.

**Anumana (operational).** Dominant move is from sign/premise to a non-observed conclusion (diagnosis, troubleshooting, legal or scientific inference). Cues include “therefore”, “because”, “implies”, “likely”. Weak hedging still permits Anumana with reduced strength.

**Upamana (operational).** Dominant move is comparison/similarity (“similar to”, “analogous to”, “acts like”). Social “I like this” is explicitly excluded. If analogy is subordinated to “therefore…”, annotators label the dominant move (often Anumana).

**Shabda (operational).** Dominant justification appeals to authority, documentation, testimony, or institutions (“according to”, “FDA label”, “court held”). News-like reporting without attribution is borderline; named institutions favor Shabda.

**Strength (operational).** Strong / Moderate / Weak encodes clarity, connector density, evidence density, and hedging—**not** model confidence and **not** classical *prāmāṇya*. Review statuses include pending / approved / needs_edit / rejected.

These definitions matter experimentally: model errors often occur exactly where guidelines mark edge cases (Upamana inside Anumana essays; Shabda/Pratyaksha report verbs).

### 2.8 Pedagogical Mapping Versus Philosophical Completeness

Indian philosophy curricula often introduce the four pramāṇas early because they are memorable and contrastive. InferAI exploits that pedagogical clarity. Philosophical completeness would additionally require:

- detailed theory of perception (ordinary vs yogic, determinate vs indeterminate);
- the full inferential schema (pakṣa, sādhya, hetu, vyāpti, dṛṣṭānta);
- conditions of trustworthy testimony;
- catalogues of fallacious reasons.

None of these are implemented as formal engines. Mentioning them here clarifies the **gap** that the phrase “Nyāya-inspired” is meant to hold open.

### 2.9 English as a Medium, Not a Replacement for Sanskrit Sources

All training and evaluation text in InferAI is English. This is a deliberate engineering constraint for a B.Tech NLP stack (Sentence-BERT MiniLM, FastAPI, Streamlit). It is also a philosophical limitation: classical Nyāya discourse is primarily Sanskrit (with later vernacular commentaries). InferAI therefore studies **English argumentative rhetoric under Nyāya-inspired labels**, not a digitization of the *Nyāya Sūtras*. Future work listed in the progress report explicitly includes Hindi/Sanskrit extension via multilingual Sentence-BERT variants.

---

## 3. Related Work

### 3.1 Explainable AI

Post-hoc explanation methods attribute predictions to input features [1]. InferAI uses optional SHAP on the logistic head in Sentence-BERT embedding space (384 dimensions), and separately provides rule-match traces that are human-readable. The project explicitly documents that embedding SHAP is **not** token-level explanation (`reports/shap_limitations.md`).

Beyond SHAP, the XAI literature distinguishes intrinsic interpretability (linear models, rule lists) from post-hoc explanation (feature attributions, counterfactuals). InferAI sits between these poles: the logistic head is intrinsically simple, while SHAP and template text are post-hoc. The symbolic engine is intrinsic to the hybrid decision at inference time but extrinsic to training. This hybrid positioning is why we report both ML-only and hybrid labels in the API.

### 3.2 Argument Mining

Stab and Gurevych’s Argument Annotated Essays Corpus (AAEC) provides claim/premise annotations for persuasive essays [2], [3]. InferAI reuses AAEC as a *source of argumentative spans* that are then manually re-labeled into Nyāya pramāṇas—an epistemic overlay, not a replacement for claim–premise structure.

Argument mining typically asks: What is the claim? What supports it? What attacks it? InferAI asks a different question: **By what epistemic instrument is the support offered?** A premise can be perceptual, inferential, analogical, or testimonial. That orthogonal axis is InferAI’s niche.

### 3.3 IBM ArgKP and Claim-Oriented Corpora

IBM ArgKP pairs arguments with key points for matching and stance-related research [11]. InferAI uses ArgKP as a pool of short natural arguments for Nyāya annotation after cleaning 36,800 rows to 8,639 unique texts. We do not evaluate key-point matching; we only reuse argument strings as real-world linguistic material.

### 3.4 Neuro-symbolic AI

Neuro-symbolic systems combine learning with rules or logic [6]. InferAI’s hybrid layer is lightweight: regex-weighted cues produce a soft distribution fused with ML softmax. Rules do not participate in gradient training; they adjust inference-time predictions and confidence. Compared with systems that compile logic into neural architectures or perform differentiable theorem proving, InferAI is intentionally shallow—closer to a **feature-rich post-processor** than to a unified neuro-symbolic learner. The benefit is auditability (`resources/rules_v2.yaml`); the cost is limited ability to correct systematic minority-class failures.

### 3.5 Sentence-BERT and Lightweight Classifiers

Reimers and Gurevych’s Sentence-BERT produces semantically meaningful sentence embeddings [7]. InferAI freezes `all-MiniLM-L6-v2` (384-d) and trains only a logistic regression head (`classification/embedder.py`, `classification/train_model.py`), prioritizing reproducibility and low compute for a B.Tech research prototype. This design echoes a common industrial pattern: strong pretrained embeddings + simple head. Our contribution is not a new encoder but the **pramāṇa label space + hybrid cues + evaluation suite** around that pattern.

### 3.6 Lexical Diversity Methodology

Professor review required length-aware lexical indices. Guiraud’s \(R=V/\sqrt{N}\) and MATTR [9], [10] are used in `reports/lexical_diversity_report.md` to show that synthetic TTR collapse is partly a length artefact. This methodological point is related work for corpus linguistics inside an AI project: without it, synthetic data can be mis-sold as “diverse” based on raw type counts alone.

### 3.7 Symbolic Reasoning and IKS

Indigenous Knowledge Systems (IKS) research encourages responsible computational engagement with classical Indian knowledge. InferAI’s contribution to IKS is the **transparent mapping of Nyāya categories into an evaluable NLP pipeline**, with documented limits. Related computational Indian philosophy work exists in digital humanities and Sanskrit NLP; InferAI is closer to applied XAI engineering than to philological encoding.

### 3.8 Inter-Annotator Agreement Practice

Cohen’s \(\kappa\) [8] remains standard for nominal annotation quality. InferAI reports \(\kappa=0.8804\) with bootstrap CI and per-class agreement tables (`reports/kappa_report.md`). The assisted-annotator caveat distinguishes this work from papers that claim fully independent dual annotation without disclosure.

### 3.9 Comparison Table

| System / Line of Work | Label Space | Learner | Symbolic Layer | XAI | IKS Link |
|----------------------|-------------|---------|----------------|-----|----------|
| Standard argument mining [2], [3] | Claim/premise/stance | Various | Rare | Variable | No |
| Pure embedding classifiers | Task-specific | SBERT+LR/NN | None | Optional SHAP | No |
| Heavy neuro-symbolic logic | Formal logic | Hybrid | Complex | Variable | Rare |
| **InferAI (this work)** | **Four pramāṇas** | **SBERT+LR** | **77 weighted cues** | **Rules + NL + optional SHAP** | **Nyāya-inspired** |

### 3.10 Summary of Gaps Filled

Relative to argument mining, InferAI adds an epistemic-mode axis. Relative to generic XAI classifiers, it adds culturally grounded labels and auditable rules. Relative to IKS demos, it adds leakage, calibration, OOD, and CV discipline. Relative to heavy neuro-symbolic stacks, it adds simplicity and reproducibility at the cost of limited logical depth. That trade-off is explicit.

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

*Averages computed from `_RULE_SPECS` in `classification/hybrid_reasoning.py`; see also `reports/rule_statistics.md` and `reports/symbolic_rule_documentation.md`.*

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

**Representative rules** (full list in `resources/rules_v2.yaml`; see `reports/symbolic_rule_documentation.md`):

| ID | Pramāṇa | Cue type | Weight | Example trigger |
|----|---------|----------|-------:|-----------------|
| PRATYAKSHA_002 | Pratyaksha | measured | 1.8 | “The voltage was measured at 3.3V.” |
| PRATYAKSHA_020 | Pratyaksha | we measured | 2.0 | “We measured the concentration at 2 mg/L.” |
| ANUMANA_002 | Anumana | therefore | 2.3 | “The premises are true; therefore the claim follows.” |
| ANUMANA_010 | Anumana | if…then | 2.4 | “If the alarm is armed then the door cannot open.” |
| UPAMANA_006 | Upamana | analogous to | 2.6 | “This is analogous to balancing a budget.” |
| UPAMANA_013 | Upamana | like a/an | 2.0 | “The mind is like a mirror.” / non-trigger: “I like football.” |
| SHABDA_001 | Shabda | according to official body | 3.4 | “According to WHO, vaccines are effective.” |
| SHABDA_003 | Shabda | according to informal source | 0.7 | “According to my friend, this works every time.” |
| SHABDA_010 | Shabda | WHO | 3.2 | “WHO issued updated guidance.” |

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

Confusion matrix (rows = true label; `reports/final_retraining_report.md`):

| True \ Pred | Anumana | Pratyaksha | Shabda | Upamana |
|-------------|---------|------------|--------|---------|
| Anumana | 226 | 1 | 1 | 7 |
| Pratyaksha | 3 | 86 | 0 | 0 |
| Shabda | 13 | 0 | 90 | 0 |
| Upamana | 13 | 0 | 0 | 93 |

The dominant residual errors on this in-distribution holdout are Shabda→Anumana and Upamana→Anumana (13 each), consistent with Anumana prevalence in merged training text.

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

### 9.4 Curriculum Pattern for IKS–AI Projects

Indian engineering programmes increasingly require IKS-linked major projects. InferAI illustrates a defensible pattern: select a classical taxonomy with contrastive categories; operationalize with written guidelines; keep symbolic claims modest; evaluate with leakage, calibration, OOD, and agreement metrics; publish negative results (e.g., real-world macro F1 0.3414) alongside positive ones (in-distribution macro F1 0.9326). Stronger future IKS integration would add scholar-annotators, bilingual corpora, and human evaluation of explanation faithfulness against classical criteria—none of which are claimed complete in the current repository.

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
11. **Snapshot fragmentation** across evaluation reports requires protocol-aware reading.
12. **Rules do not train the ML head**, limiting correction of systematic errors.

---

## 11. Ethics

### 11.1 Trustworthy and Responsible AI

InferAI surfaces confidence adjustments and rule evidence to reduce silent failures, but miscalibration means raw confidence must not be treated as probability of truth. Responsible UIs should show ML and hybrid labels when they disagree.

### 11.2 Epistemic Validity

Predicting a pramāṇa label is **not** certifying that the argument is sound or that testimony is trustworthy. Shabda cues detect *appeals to authority*, not authority’s correctness. Conflating testimony detection with content endorsement would be an ethical failure in educational or journalistic misuse.

### 11.3 Bias

Anumana skew biases the system toward labeling natural arguments as inferential. Institutional Shabda cues (WHO, CDC, government) encode contemporary authority markers that may not transfer across cultures or time. Synthetic templates may encode authors’ rhetorical habits. Near-duplicate padding can bias evaluation optimism.

### 11.4 Educational Use

Suitable as a teaching aid for epistemology/XAI with instructor oversight; unsuitable as an automated grader of “correct reasoning” without human review. Learners should be taught the inspired/operationalized distinction explicitly.

### 11.5 Data Stewardship

AAEC and IBM ArgKP must be used under their original distribution terms; the repository notes no bundled license files (`docs/dataset_provenance.md`). Derived Nyāya labels are project-created annotations.

### 11.6 Overclaim Risk

Marketing InferAI as “computational Nyāya” or as robust epistemic understanding would be unethical given minority-class collapse on real-world CV and ECE 0.457. Branding must remain subordinate to evaluation.

---

## 12. Conclusion

InferAI (PramāṇaNet) is a Nyāya-*inspired* neuro-symbolic framework for classifying short English arguments into four pramāṇas, combining Sentence-BERT, logistic regression, and a documented 77-rule hybrid engine. The finalized corpus integrates 611 synthetic + 650 AAEC-reviewed + 600 IBM-reviewed examples (1,861 merged; 1,250 manually reviewed). Inter-annotator \(\kappa=0.8804\) indicates strong label consistency under documented caveats.

Evaluation shows a clear pattern: **in-distribution holdout performance is strong** (accuracy 0.9287, macro F1 0.9326), while **real-world macro F1 remains low** (0.3414) despite high accuracy (0.9118), and **template-test scores vary by training snapshot** (0.60–0.955). Calibration (ECE 0.457) and OOD diagnostics (JSD 0.5065) further discipline interpretation.

### Future Work

As stated in project reports: larger balanced annotation; active learning for boundary cases; temperature scaling / calibration; token-level attributions; multilingual Nyāya-aware extension; tighter train-time integration of symbolic cues; rebuild of on-disk reviewed CSVs to match canonical 650/1,861 counts; and independent dual annotation to replace assisted second-pass \(\kappa\).

---

## 13. Extended Methodology Notes (Implementation Faithfulness)

This section records implementation details that reviewers typically request, taken directly from the InferAI codebase and reports. It does not introduce new experiments.

### 13.1 Embedding and Training Mechanics

The encoder is loaded once as `SentenceTransformer('all-MiniLM-L6-v2')` in `classification/embedder.py`. Training script `classification/train_model.py` reads a CSV with either `pramana_label` or `label`, embeds all texts, fits `LabelEncoder`, splits with `test_size=0.2` and `random_state=42`, and trains `LogisticRegression(max_iter=1000)`. Artifacts written:

- `models/nyaya_model.pkl`
- `models/label_encoder.pkl`
- `models/shap_background.npy` (up to 200 training embeddings)
- `models/evaluation/metrics.json`, classification report, and confusion-matrix plots when available

The merge trainer `classification/retrain_merged.py` concatenates synthetic + AAEC reviewed + IBM reviewed, normalizes text, drops exact duplicates, shuffles with seed 42, and exports `dataset/processed/merged_retraining_corpus.csv`. Canonical documentation records 3,589 duplicates removed and 1,861 retained rows.

### 13.2 API Contract

`POST /analyze` (`api/app.py`) accepts `{ "text": ..., "include_shap": false }` and returns at least:

`input_text`, `claim`, `evidence`, `premises`, `reasoning_indicators`, `highlighted_html`, `predicted_pramana` (ML-only), `hybrid_predicted_pramana`, `confidence`, `adjusted_confidence`, `reasoning_strength`, `reasoning_strength_debug`, `explanation`, and nested `hybrid` (full fusion dictionary). Optional `shap` is included when requested and when background embeddings exist.

This split between ML label and hybrid label is intentional: users can see when symbolic cues overturn or reinforce the logistic head.

### 13.3 Frontend Demonstration

`frontend/app.py` provides Streamlit demos for Observation / Inference / Analogy / Authority examples, confidence bars (combined vs model), strength badges, highlighted HTML, template explanation, and optional SHAP bar charts for top embedding dimensions. The UI is a research demonstrator, not a production deployment interface.

### 13.4 Reasoning Strength

Composite strength estimation consumes pattern signals from `detect_pattern_signals()` (authority/analogy/inference/observation hit counts and weighted scores). Strength is reported as weak / moderate / strong and is **independent** of the pramāṇa label in the annotation guidelines; the runtime composite is a heuristic overlay for UI communication (`reasoning_strength/`).

### 13.5 Softmax Soft Floor Motivation

Rule distribution uses `score = 1 + raw^{1.35}` rather than a hard one-hot of the winning cue family. The soft floor prevents the symbolic channel from assigning exact zero probability to unused pramāṇas, which would otherwise make fusion brittle when only one weak cue fires. The exponent 1.35 modestly sharpens strong evidence. The ×0.88 mix penalty when three or more families fire encodes the annotation guideline that short spans usually have one *dominant* mode.

### 13.6 Why Logistic Regression Rather Than a Fine-Tuned Transformer

Project reports emphasize that the architecture was frozen after professor review so that improvements could be attributed to data and evaluation methodology. A linear head on frozen MiniLM embeddings is:

- cheap to retrain on 1,861 rows;
- compatible with `LinearExplainer` SHAP;
- transparent relative to a large fine-tuned classifier;
- adequate for a B.Tech research prototype focused on taxonomy, hybrid cues, and evaluation discipline.

We do not claim state-of-the-art classification accuracy against large instruction-tuned models; we claim a **reproducible, explainable, Nyāya-inspired pipeline**.

---

## 14. Extended Dataset Construction Narrative

### 14.1 Why Synthetic Data Was Necessary

At project start, insufficient manually annotated Nyāya-labeled English data existed in-repository. Template generation (`dataset/labeled_corpus.py`) created a balanced seed so that all four labels appeared in training. The cost—template repetition—is quantified by Guiraud’s R = 0.783 versus 18.697 on a small curated real-world set (`reports/lexical_diversity_report.md`).

### 14.2 AAEC Integration Pipeline

1. Parse brat `.ann` / `.txt` pairs under `dataset/external/AAEC/` (402 essays).
2. Extract MajorClaim / Claim / Premise spans → 6,050 components (`reports/aaec_statistics.md`).
3. Build annotation sheets with heuristic `candidate_pramana` labels.
4. Manual review in stages (initial 250-row sheet, then expansion).
5. Export `reviewed=yes` rows → finalized reviewed count **650**.

AAEC contributes essay-genre argumentative English. Reviewed exports remain Anumana-heavy, reflecting persuasive-essay rhetoric rather than laboratory observation prose.

### 14.3 IBM ArgKP Integration Pipeline

1. Load `ArgKP_combined.csv` (36,800 rows).
2. Normalize, drop empties, deduplicate → 8,639 unique arguments.
3. Sample and annotate 600 rows to Nyāya labels + strength.
4. Export reviewed CSV (600 rows; avg length 103.0 characters).

IBM contributes short claim-like arguments with stance metadata in the source corpus; InferAI uses the argument text and human Nyāya labels, not IBM’s original stance labels as pramāṇa substitutes.

### 14.4 Double Annotation and Disagreement Analysis

Cohen’s \(\kappa=0.8804\) on 200 matched pairs indicates strong agreement (`reports/kappa_report.md`). Confusion patterns show:

- Anumana is highly stable (ann1 agreement-when-labeled 98.7%).
- Pratyaksha is the least stable (78.57% agreement-when-ann1), consistent with short-span ambiguity between observation reports and inferential glosses.
- Upamana and Shabda show mid-to-high stability (~89%).

The project caveat remains critical for paper honesty: annotator 2 was partly rule-assisted. Therefore \(\kappa\) should be read as **guideline consistency under assisted second pass**, not as a fully independent dual-expert ceiling.

### 14.5 Strength Label Distributions

IBM reviewed strength (`reports/ibm_reviewed_dataset_stats.md`): moderate 543 (90.5%), strong 55 (9.17%), weak 2 (0.33%). AAEC reviewed strength tables in `aaec_reviewed_dataset_stats.md` report moderate dominance (84% in the tabulated distribution). Strength is used for explanation tone and UI badges; it is not the primary classification target of the logistic head.

### 14.6 Early 811-Row Merge (Historical)

An earlier merge (`nyaya_dataset_merged.csv`, 811 rows) combined synthetic data with curated real-world v1 (200 rows) for initial experiments (`docs/dataset_provenance.md`). Train/validation sizes 648/163 appear in older leakage reports. The **production narrative** of this paper uses the later 1,861-row merge. We retain mention of the 811-row stage to avoid silently rewriting project history.

---

## 15. Extended Experimental Interpretation

### 15.1 Why Multiple Accuracies Appear in the Repository

A careless reader might treat 0.996, 0.955, 0.9287, 0.76, 0.685, and 0.60 as contradictory claims about “the model.” They are not. They are **protocol- and snapshot-indexed**:

| Approx. figure | What it measures |
|----------------|------------------|
| ~0.996 | Early synthetic-heavy random holdout (`final_evaluation_report.md`) |
| 0.955 | Early curated template test under a high-performing snapshot (`heldout_evaluation.md`) |
| 0.9287 | In-distribution 20% holdout after merged retraining (`final_retraining_report.md`) |
| 0.76 | Another curated-test snapshot in `final_evaluation_report.md` |
| 0.685 | Retrained original (no resampling) on template test (`error_analysis.md`) |
| 0.60 | Oversampled production path on template test (`final_retraining_report.md`) |
| 0.9118 / 0.3414 | Real-world CV accuracy / macro F1 (`real_world_cross_validation.md`) |

Scientific reporting for InferAI therefore requires **naming the protocol**. This paper adopts that discipline.

### 15.2 Macro F1 Versus Accuracy on Real-World CV

With Anumana ≈ 90% of real-world reviewed text, a majority-class classifier can exceed 90% accuracy while failing minorities. Pooled OOF results confirm this: Anumana F1 0.9531 versus Pratyaksha/Upamana F1 0.0. InferAI’s progress report therefore elevates **macro F1** as the headline metric for real-world claims. We reaffirm that choice here.

### 15.3 Oversampling Trade-off

Oversampling minority classes to 500 each (Anumana left at 1,163) was tested (`reports/oversampling_comparison.md`). Template-test macro F1 fell from 0.6831 (original) to 0.6007 (oversampled). Class weighting similarly hurt aggregate metrics while shifting recall. The empirical lesson: **naïve rebalancing on a template OOD test is not automatically beneficial**. Future rebalancing should be validated on real-world CV macro F1, not only on template accuracy.

### 15.4 Calibration and User Trust

ECE = 0.457 means confidence bars in the UI can be systematically wrong. Hybrid agreement adjustments (+10%/+4 or −10%) are heuristic UX signals, not calibrated probabilities. Educational deployments should display qualitative caution (“confidence is uncalibrated”)—a recommendation aligned with `reports/calibration_report.md` and the progress report’s calibration section.

### 15.5 Leakage Versus Near-Duplication

Zero exact leakage is necessary but not sufficient. 506 near-duplicate pairs, mostly padding vignettes, can still inflate template-style scores. The OOD analysis (JSD 0.5065, TV 0.576) shows that even without exact overlap, the template test differs lexically and distributionally from AAEC/IBM training text. Hence InferAI’s evaluation story is: **literal leakage controlled; distributional shift acknowledged; near-duplicate padding documented**.

### 15.6 Confusion Pair Semantics

Error analysis pairs map onto epistemic boundary difficulty:

- **Upamana → Anumana:** analogies often appear inside inferential essays; surface “like” may be weak or absent while “because/therefore” dominates.
- **Shabda → Pratyaksha:** “reported/observed” language can look observational when the epistemic move is actually testimony.
- **Shabda → Anumana:** studies are cited as premises inside inferential arguments.
- **Pratyaksha → Anumana:** measurements are immediately interpreted.

These pairs motivate both guideline clarifications and future active learning around boundary spans.

---

## 16. Symbolic Engine as Operationalization (Deeper Discussion)

### 16.1 From ~28 Binary Rules to 77 Weighted Cues

`reports/symbolic_rule_improvements.md` records the expansion:

| Aspect | Before | After |
|--------|--------|-------|
| Count | ~28 boolean | 77 weighted |
| Scoring | hit × 1.2–1.4 | weights 0.7–3.4 + multi-cue bonus |
| Shabda | flat `according to` | tiered official / expert / informal |
| Upamana | bare `like` risk | phrase-level + FP suppression |
| Confidence | fused max only | agreement boost / disagreement reduce |

This expansion answers a professor-review demand for richer symbolic documentation without claiming a new learning algorithm.

### 16.2 Authority Tiering as Epistemic Sensitivity

Assigning weight 3.4 to “according to WHO” and 0.7 to “according to my friend” is not a moral ranking of humans; it is an **operational proxy** for how English argumentative prose typically marks institutional versus informal testimony. Classical Nyāya discusses the reliability of speakers (*āpta*); InferAI only detects surface cues correlated with those discussions.

### 16.3 What the Rules Are Not

Rules are not:

- a theorem prover;
- a Sanskrit parser;
- a fallacy detector;
- a truth validator.

They are **weighted detectors of justificatory phrasing** used to build \(p_{\mathrm{rules}}\) and explanation traces.

### 16.4 Documentation Artifact

`resources/rules_v2.yaml` lists every rule with id, pramāṇa, cue_type, weight, pattern, description, and examples. `reports/rule_statistics.md` summarizes distributions. These artifacts make the symbolic layer auditable—an IKS-friendly transparency practice.

---

## 17. Positioning Relative to Conference Venues

### 17.1 NLP / Argument Mining Venues

InferAI offers a new label inventory (pramāṇas) over reused argumentative sources (AAEC, ArgKP), with careful OOD and leakage analysis. The modest classifier is acceptable when the contribution is taxonomic and evaluative.

### 17.2 XAI Venues

The combination of rule traces, agreement metadata, and explicitly limited SHAP is a case study in **honest explanation stacking**: multiple channels with different fidelity claims.

### 17.3 IKS Venues

The paper’s primary IKS contribution is methodological humility: classical categories guide design; evaluation prevents romantic overclaim. This is a template for other IKS–AI bridges (e.g., other *darśana* taxonomies) that wish to remain academically defensible.

### 17.4 Hybrid AI Venues

Fusion weight 0.8/0.2, multi-cue bonuses, and FP guards are simple but fully specified. Reproducibility is prioritized over architectural novelty.

---

## 18. Threats to Validity

**Internal validity.** Different reports correspond to different model files; without a single frozen evaluation harness re-run end-to-end on one checkpoint, cross-report metric comparison requires careful labeling (as done here).

**External validity.** English short spans from essays and ArgKP claims do not represent spoken debate, Sanskrit commentary, or long documents.

**Construct validity.** Pramāṇa labels on English spans are an operational construct; they may diverge from classical determinations by trained Nyāya scholars.

**Statistical conclusion validity.** Real-world CV minority supports are tiny (e.g., 5 Pratyaksha in the CV export class table), so zero F1 is expected and unstable.

**Annotation validity.** Assisted second annotator inflates \(\kappa\) relative to independent dual annotation.

---

## 19. Reproducibility Checklist

From the project README and report generators:

1. `pip install -r requirements.txt`
2. Build/generate datasets as needed (`dataset_generator.py`, review export scripts).
3. `python classification/train_model.py` or `classification/retrain_merged.py`
4. `python reports/run_all_evaluation_reports.py` (or individual generators)
5. `python calculate_kappa.py`
6. `python reports/generate_professor_documentation.py`
7. Serve API: `uvicorn api.app:app --reload`
8. UI: `streamlit run frontend/app.py`

Canonical counts live in `reports/dataset_canonical.py`. Symbolic rules live in `classification/hybrid_reasoning.py` and are mirrored in `resources/rules_v2.yaml`.

---

## 20. Closing Synthesis

InferAI demonstrates that a **small, well-documented neuro-symbolic system** can host an Indigenous epistemological taxonomy inside a modern NLP stack without pretending to replace classical scholarship. The empirical picture is mixed by design: strong in-distribution metrics, sobering real-world macro F1, transparent calibration failure, and explicit OOD warnings. That mixture is the scientific result.

If the project’s early prototype showed that Nyāya labels *can* be predicted from English templates, the matured system shows *when those predictions fail*—on minority pramāṇas, on natural argumentative skew, and under miscalibrated confidence. For AI, that is progress. For IKS, that is respect.

---

## 21. Snapshot-Specific Confusion Analysis (High-Accuracy Template Protocol)

For completeness, we include the automated confusion analysis associated with the high-accuracy curated-test snapshot (`reports/confusion_matrix_analysis.md`; aligned with `heldout_evaluation.md` accuracy 0.955).

Confusion counts:

| True \ Pred | Anumana | Pratyaksha | Shabda | Upamana |
|-------------|---------|------------|--------|---------|
| Anumana | 42 | 2 | 0 | 0 |
| Pratyaksha | 0 | 44 | 2 | 0 |
| Shabda | 0 | 5 | 52 | 0 |
| Upamana | 0 | 0 | 0 | 53 |

Key findings from that report:

- Most confused pair: **Shabda → Pratyaksha** (5 errors; 8.8% of true Shabda).
- Highest recall: **Upamana 100%** (n=53).
- Lowest recall among classes in this snapshot: **Shabda 91.2%**.

Per-class F1 values match the held-out evaluation table (Anumana 0.9767, Pratyaksha 0.9072, Shabda 0.9369, Upamana 1.0000). This snapshot demonstrates that **under template-aligned conditions**, all four pramāṇas are recoverable at high F1. It must not be generalized to real-world CV without the imbalance caveats in Section 6.5.

The contrast with error analysis at accuracy 0.685—where Upamana→Anumana dominates—shows that **confusion structure is snapshot-dependent**. Papers that cite only one confusion matrix without naming the checkpoint would mislead.

---

## 22. Explainability Stack in Depth (From Project SHAP Documentation)

### 22.1 What Embedding SHAP Answers

Per `reports/shap_limitations.md`, each SHAP value answers: how much embedding dimension \(d\) pushed log-odds toward class \(c\) relative to the background distribution. Indices 0–383 are latent MiniLM axes; they do **not** map one-to-one to cue words such as “therefore.”

### 22.2 What SHAP Does Not Answer

- Token importance for end users.
- Nyāya rule-trace contributions (rules fuse after the ML head).
- Causal claim/premise structure (handled by separate extractors).
- Strength estimation (heuristic composite outside SHAP).

Template NL explanations are independent of SHAP and must not be conflated with SHAP attributions.

### 22.3 Why Embedding SHAP Is Harder to Read Than Token SHAP

| Aspect | Embedding-level (current) | Token-level (ideal / future) |
|--------|---------------------------|------------------------------|
| Unit | 384 latent dims | Words / subwords |
| Human readability | Low | High |
| Exact LinearExplainer fit | Yes (linear head) | Needs full-pipeline method |
| Alignment with Nyāya cues | Indirect | Can highlight cue phrases |
| Compute | Low | Higher |

The architecture’s decoupling of frozen MiniLM from the linear head improves training simplicity but sacrifices token fidelity. Professor review correctly pushed the project to document this rather than overclaim “explainability.”

### 22.4 Complementary User-Facing Channels

Users still receive structured extraction, HTML highlights, hybrid diagnostics, and template NL text. SHAP remains a **developer diagnostic**. This multi-channel design is InferAI’s practical answer to XAI expectations under architectural constraints.

---

## 23. Worked Qualitative Scenarios (Consistent With System Behavior)

The following scenarios illustrate intended behavior using the implemented cue families and fusion logic. They are explanatory, not new labeled benchmark items.

**Scenario A — Perception.** “We measured the concentration at 2 mg/L using laboratory sensors.” Pratyaksha cues (`we measured`, `laboratory`, `sensor`) accumulate high raw score; ML head typically agrees on scientific measurement prose; hybrid confidence may receive an agreement boost.

**Scenario B — Inference.** “Packet loss spikes at peak hours, therefore congestion is more plausible than hardware failure.” Anumana cues (`therefore`) fire; this matches labeling-guideline examples.

**Scenario C — Analogy.** “Gradient descent is like rolling downhill on a noisy loss surface.” Upamana cue (`is like`) fires; social “I like football” would be suppressed by FP guards.

**Scenario D — Tiered testimony.** “According to WHO, vaccines are effective” triggers high-weight Shabda; “According to my friend, vaccines are effective” triggers low-weight Shabda (0.7). The tiering is an operational proxy for institutional versus informal testimony marking in English.

**Scenario E — Disagreement case.** A span with strong “because/therefore” (Anumana) that the ML head labels Shabda due to a study mention may surface ML≠rule labels with reduced adjusted confidence—exactly the transparency the hybrid layer is designed to provide.

---

## 24. Professor-Review Alignment Summary

`reports/professor_review_response.md` and `reports/review_comment_updates.md` record that core architecture remained unchanged while dataset provenance, lexical indices, \(\kappa\) reporting, leakage, calibration, and SHAP limitation docs were strengthened. This paper inherits that posture: **architecture simple, evaluation honest, Nyāya claims bounded**.

Key quantitative milestones retained from the progress report’s summary table include: 1,250 reviewed real-world sentences; 1,861 merged training rows; \(\kappa=0.8804\); 77 symbolic rules; real-world CV macro F1 0.3414; ECE 0.457; JSD 0.5065.

---

## 25. Evaluation Protocol Discipline (Additional Discussion)

### 25.1 Why This Paper Refuses a Single Headline Accuracy

InferAI’s repository contains multiple accurate-but-incompatible accuracy numbers because evaluation was expanded after professor review without erasing historical snapshots. A single headline would be scientifically dishonest. Instead, we require every numeric claim to name:

1. the **corpus** (synthetic, merged 1,861, AAEC/IBM reviewed, template test);
2. the **protocol** (random holdout, template test, real-world CV, calibration set);
3. the **training configuration** (original, class-weight, oversampled);
4. the **metric** (accuracy vs macro F1 vs ECE vs \(\kappa\)).

This discipline is itself a contribution relative to prototype reports that quote only the most flattering figure.

### 25.2 Recommended Reporting Set for Future Citations of InferAI

When citing InferAI results, we recommend reporting at minimum:

- Merged corpus size **1,861** (611+650+600) and reviewed real-world **1,250**.
- In-distribution holdout: accuracy **0.9287**, macro F1 **0.9326**.
- Real-world CV: accuracy **0.9118**, macro F1 **0.3414** (with imbalance caveat).
- Template-test range across snapshots: **0.60–0.955**, with OOD warning (JSD **0.5065**).
- \(\kappa=0.8804\) with assisted-annotator caveat.
- ECE **0.457** (uncalibrated).
- Symbolic layer: **77** weighted rules, fusion **0.8/0.2**.

### 25.3 Relationship Between Hybrid Rules and Reported Metrics

Importantly, the quantitative classification metrics in Section 6 primarily evaluate the **logistic regression head on Sentence-BERT embeddings**. The hybrid rule engine adjusts inference-time labels and confidence in the API, but the tabulated CV and holdout reports retrain or evaluate LR heads without claiming that every reported number is a hybrid-only score. Readers should therefore treat hybrid fusion as an **interpretability and confidence layer** whose quantitative impact is exposed through agreement metadata rather than as a separately benchmarked end-to-end learner in every table.

### 25.4 Canonical Count Lag (Transparency Note)

`reports/repository_consistency_audit.md` records that documentation was updated to canonical sizes (AAEC 650, merged 1,861, reviewed 1,250) while some evaluation artifacts were generated on earlier on-disk exports (e.g., CV fold metrics on an 850-row AAEC250+IBM600 merge). This paper states both facts rather than silently rewriting history. Rebuild of reviewed CSVs and re-running CV on the full 1,250-row pool is listed as future work.

### 25.5 Educational Deployment Checklist

For classroom use of the Streamlit demo:

1. Prefer examples with clear single-mode cues (measurement / therefore / analogous to / according to WHO).
2. Display both ML and hybrid labels when they disagree.
3. Do not treat confidence bars as calibrated probabilities (ECE 0.457).
4. Do not present embedding SHAP indices as token rationales.
5. Frame the taxonomy as Nyāya-inspired operational labels, not classical verdicts.

---

## 26. Notation Summary

| Symbol | Meaning |
|--------|---------|
| \(x\) | Input English argument span |
| \(e\in\mathbb{R}^{384}\) | Frozen MiniLM embedding |
| \(p_{\mathrm{ML}}\) | Softmax from logistic regression |
| \(p_{\mathrm{rules}}\) | Soft distribution from weighted cues |
| \(p_{\mathrm{fused}}\) | \(0.8\,p_{\mathrm{ML}}+0.2\,p_{\mathrm{rules}}\) (renormalized) |
| \(s_c\) | Raw weighted score for pramāṇa \(c\) |
| \(\kappa\) | Cohen’s kappa (annotation agreement) |
| ECE | Expected Calibration Error |
| JSD | Jensen–Shannon divergence (lexical OOD) |
| MATTR | Moving-Average Type–Token Ratio (window 50) |
| Guiraud’s \(R\) | \(V/\sqrt{N}\) lexical richness |

---

## 27. Per-Class Deep Dive (From Project Reports)

### 27.1 Anumana Dominance as Both Feature and Bug

Across IBM reviewed data (90.5% Anumana) and AAEC reviewed exports (≈88% Anumana in tabulated class distributions), argumentative English in these sources is predominantly inferential. That is linguistically unsurprising for persuasive essays and claim texts. For InferAI it means:

- majority-class accuracy is easy to inflate;
- minority pramāṇas require deliberate sampling or cost-sensitive learning that was tested but not adopted for production after template-test degradation;
- symbolic cues for Anumana (`because`, `therefore`, `if…then`) are frequent and strong, reinforcing the skew at fusion time when ML already favors Anumana.

### 27.2 Pratyaksha Scarcity

Reviewed IBM data contains only **5** Pratyaksha rows; AAEC reviewed tables report **0** in the exported class-count snapshot used for distribution reporting. Real-world CV therefore cannot learn a stable Pratyaksha decision boundary (pooled F1 0.0). Synthetic data remains the primary source of perceptual examples in the merged train set. This is a **data problem**, not a Nyāya definition problem: laboratory and sensor prose is rare in AAEC/ArgKP claim pools.

### 27.3 Upamana and Shabda Boundary Difficulty

Error analysis and \(\kappa\) confusion both highlight Upamana↔Shabda and Shabda↔Pratyaksha borders. Guidelines already warn that “like” is ambiguous and that report verbs can look observational. The symbolic FP guards reduce some Upamana false positives, but they cannot resolve cases where a span genuinely mixes modes. Hybrid disagreement metadata is the intended human-facing signal for those cases.

### 27.4 Strength Labels Versus Classifier Target

Strength (weak/moderate/strong) is annotated and used in UI/explanation tone. The logistic head’s primary target is pramāṇa, not strength. IBM reviewed strength is heavily moderate (90.5%). Conflating strength with confidence would be a category error; the paper and UI keep them separate.

### 27.5 What “Success” Means for This Project

Success for InferAI is not SOTA leaderboard accuracy. Success is:

1. a documented Nyāya-inspired label contract;
2. a reproducible hybrid pipeline;
3. evaluation that reveals failure modes (imbalance, OOD, miscalibration);
4. explainability claims that match implementation limits.

By those criteria, the post-review system is stronger than the early template-only prototype—even when some accuracy numbers are lower.

### 27.6 Artifact Paths for Reviewers

Primary paper-supporting artifacts: `docs/dataset_provenance.md`, `reports/dataset_canonical.py`, `reports/final_retraining_report.md`, `reports/real_world_cross_validation.md`, `reports/kappa_report.md`, `reports/calibration_report.md`, `reports/test_set_distribution_analysis.md`, `reports/shap_limitations.md`, `reports/symbolic_rule_documentation.md`, `resources/rules_v2.yaml`, and figures under `reports/figures/`. Consistency verification: `paper/research_paper_consistency_report.md`.

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

[9] P. Guiraud, *Les caractères statistiques du vocabulaire*. Paris, France: Presses Universitaires de France, 1954. (Guiraud’s \(R=V/\sqrt{N}\) as implemented in `reports/generate_lexical_diversity.py`)

[10] M. A. Covington and J. D. McFall, “Cutting the Gordian knot: The moving-average type–token ratio (MATTR),” *Journal of Quantitative Linguistics*, vol. 17, no. 2, pp. 94–100, 2010.

[11] IBM Research, ArgKP / Argument Knowledge Pairing corpus (combined release used under `dataset/external/IBM_ARGKP/`; project statistics in `reports/ibm_argkp_statistics.md`).

[12] InferAI project artifacts: `reports/*`, `docs/dataset_provenance.md`, `resources/rules_v2.yaml`, `classification/hybrid_reasoning.py`, `reports/symbolic_rule_documentation.md` (primary empirical sources for this paper).

[13] L. S. Shapley, “A value for n-person games,” in *Contributions to the Theory of Games II*, Princeton Univ. Press, 1953. (foundational cooperative-game value underlying SHAP attributions)

[14] scikit-learn developers, “sklearn.linear_model.LogisticRegression,” scikit-learn documentation (API used with `max_iter=1000` in `classification/train_model.py`).

[15] N. Reimers et al., sentence-transformers / `all-MiniLM-L6-v2` model card (encoder used frozen in InferAI; see Reimers and Gurevych [7] for Sentence-BERT methodology).

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
