# InferAI — SHAP Explainability: Scope and Limitations

This document clarifies what the current SHAP integration explains, why embedding-level attributions are hard to interpret, and recommended future work. It addresses professor feedback on explainability claims in the InferAI (Nyāya-inspired) pipeline.

## What SHAP is actually explaining

InferAI's classifier is a **logistic regression head** on top of **Sentence-BERT** (`all-MiniLM-L6-v2`) embeddings (`classification/embedder.py`, `explanation_engine/shap_explainer.py`).

SHAP is applied via `shap.LinearExplainer`, which is exact for linear models. The explainer attributes the model's output to **input features** — in this case, the **384-dimensional dense embedding vector**, not individual words or tokens.

Each SHAP value answers:

> *How much did embedding dimension **d** push the log-odds toward class **c** relative to the background distribution?*

The API optionally returns the top-|SHAP| embedding dimensions (`top_embedding_contributions`). These indices (0–383) are **latent semantic axes** learned by MiniLM during pre-training; they do not map one-to-one to human-readable concepts like "therefore" or "I saw."

## What SHAP is **not** explaining

- **Not token-level importance:** Users do not see which words drove the prediction.
- **Not Nyāya rule traces:** Hybrid rule cues (`classification/hybrid_reasoning.py`) are fused **after** the ML head; SHAP does not attribute rule-based adjustments.
- **Not causal reasoning structure:** Claim/premise extraction (`preprocessing/argument_structure.py`) is separate from SHAP.
- **Not strength estimation:** Composite strength (`reasoning_strength/composite.py`) is heuristic and outside the SHAP model.

The template-based natural-language explanation (`explanation_engine/explainer.py`) is **independent** of SHAP and should not be conflated with SHAP attributions.

## Limitations of SHAP on dense embedding dimensions

1. **Opacity of dimensions:** Embedding coordinates are not intrinsically interpretable. Reporting "dimension 142 contributed +0.31" does not tell an end user *why* the text was classified as Anumana.

2. **Two-stage model:** Sentence-BERT is frozen at inference; SHAP explains the linear head only. Important lexical cues may be **entangled** across many dimensions, so attributions are distributed rather than sparse.

3. **Background dependence:** `LinearExplainer` uses a background matrix (`models/shap_background.npy`, up to 200 training embeddings). Attributions are relative to that background; changing the background changes SHAP values.

4. **Multiclass attribution:** For multi-class logistic regression, SHAP returns per-class attributions. The reported vector corresponds to the **predicted class** (or a selected class); other classes have different attribution profiles.

5. **Correlation among dimensions:** Embedding dimensions are correlated; SHAP for linear models assumes feature independence in the cooperative game formulation — a known approximation limitation even when the explainer is "exact" for the linear wrapper.

## Why embedding-level SHAP is less interpretable than token-level SHAP

| Aspect | Embedding-level (current) | Token-level (ideal) |
|--------|-------------------------|---------------------|
| Unit of explanation | 384 latent dimensions | Words / subwords |
| Human readability | Low | High |
| Fits linear head only | Yes (exact LinearExplainer) | Requires model that accepts token inputs or perturbation-based SHAP |
| Aligns with Nyāya cues | Indirect | Can highlight "therefore", "like", "according to" |
| Compute cost | Low | Higher for transformer end-to-end models |

Token-level SHAP typically requires explaining the **full pipeline** (BERT + classifier) with a masker over input tokens, or using a model with explicit attention weights. Our architecture decouples embedding generation from classification, which improves training simplicity but ** sacrifices token fidelity** in explanations.

## Current complementary explanations in InferAI

Users still receive:

- **Structured extraction:** claim, premises, reasoning indicators
- **Highlighted HTML** for cue phrases
- **Hybrid fusion diagnostics** (ML vs rule distribution)
- **Template NL explanation** tied to predicted pramāṇa

These address usability; SHAP provides a **research-oriented diagnostic** for developers, not a primary user-facing rationale.

## Future work

1. **Token-level attributions:** Integrate `shap.Explainer` with a masking pipeline over input text, or use Integrated Gradients on the full Sentence-BERT + logistic stack.

2. **Layer-wise relevance propagation (LRP) or attention rollout** on MiniLM tokens for pramāṇa-specific cue alignment.

3. **Post-hoc concept activation vectors (CAVs)** linking embedding directions to Nyāya indicator families (perception, inference, analogy, testimony).

4. **Human evaluation** of explanations: rate helpfulness, faithfulness, and alignment with annotator rationales from `dataset/labeling_guidelines.md`.

5. **Joint reporting:** Present SHAP embedding mass **alongside** rule hits and indicator spans so quantitative and qualitative evidence agree.

6. **Calibration + SHAP:** Miscalibrated probabilities (see `reports/calibration_report.md`) should not be over-interpreted as epistemic confidence when reading SHAP magnitudes.

---

*Generated as part of the InferAI professor review response package. Implementation: `explanation_engine/shap_explainer.py`.*
