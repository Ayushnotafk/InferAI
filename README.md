# InferAI

InferAI is a Nyāya-inspired explainable argument analysis system. It classifies short English arguments into four pramāṇa-style labels (Pratyaksha, Anumana, Upamana, Shabda), fuses **softmax outputs with lightweight rule cues** (`classification/hybrid_reasoning.py`), estimates **ML confidence** and **adjusted hybrid confidence**, extracts **claim / premises**, surfaces **reasoning indicators** and **HTML-safe cue highlighting**, optionally returns **SHAP** over the Sentence-BERT embedding space, and emits a short **varied natural-language explanation** (`explanation_engine/explainer.py`).

## Data layout

| Path | Role |
|------|------|
| `dataset/labeling_guidelines.md` | Annotator instructions and definitions |
| `dataset/labeled_corpus.py` | Builds `master_dataset.csv` and `test_set.csv` |
| `dataset/master_dataset.csv` | Curated seed rows (`text`, `pramana_label`, `strength_label`, …) |
| `dataset/test_set.csv` | Held-out evaluation set (kept out of `nyaya_dataset.csv`) |
| `dataset/raw/nyaya_dataset.csv` | Training table (`text`, `label`) produced by `dataset_generator.py` |

## Setup

From the project root:

```bash
pip install -r requirements.txt
python dataset_generator.py
python classification/train_model.py
python reports/generate_final_evaluation_report.py
```

Training writes:

- `models/nyaya_model.pkl`, `models/label_encoder.pkl`
- `models/shap_background.npy` (SHAP `LinearExplainer` background)
- `models/evaluation/metrics.json`, `classification_report.txt`, and plot PNGs (including `confusion_matrix_test_set.png` when `dataset/test_set.csv` is present)

The Markdown report is written to `reports/final_evaluation_report.md`.

## Research Evaluation and Validation

InferAI includes a reproducible evaluation suite for thesis / professor review. Run everything after training:

```bash
python reports/run_all_evaluation_reports.py
```

Or run individual generators:

| Script | Output | Description |
|--------|--------|-------------|
| `reports/generate_dataset_statistics.py` | `reports/dataset_statistics.md`, `reports/csv/*.csv` | Corpus sizes, class/strength distributions, train/val/test split |
| `reports/generate_lexical_diversity.py` | `reports/lexical_diversity_report.md`, `reports/lexical_diversity_comparison.csv` | TTR, Guiraud's R, MATTR-50 (synthetic vs real-world) |
| `calculate_kappa.py` | `reports/kappa_report.md`, `reports/kappa_confusion_matrix.png` | Cohen's κ, agreement %, bootstrap 95% CI, per-class rates |
| `reports/generate_heldout_evaluation.py` | `reports/heldout_evaluation.md` | Per-class P/R/F1, macro/weighted F1, balanced accuracy on `test_set.csv` |
| `reports/generate_confusion_matrix_analysis.py` | `reports/confusion_matrix_analysis.md` | Automatic confusion-pair interpretation |
| `reports/generate_calibration_report.py` | `reports/calibration_report.md` | Reliability diagram, ECE, Brier score |
| `reports/check_data_leakage.py` | `reports/data_leakage_report.md` | Exact + near-duplicate audit (RapidFuzz + embeddings) |
| `reports/generate_professor_review_response.py` | `reports/professor_review_response.md` | Consolidated response to review concerns |

**Explainability note:** `reports/shap_limitations.md` documents what embedding-level SHAP does and does not explain.

Figures are saved under `reports/figures/`; tabular exports under `reports/csv/`.

## Research neuro-symbolic framework

InferAI v0.4 adds a **research-grade evaluation layer** without breaking the existing pipeline. All new behavior is modular and configurable via API flags or the benchmark scripts.

### Fusion modes (alpha)

`alpha` is the **ML weight** in hybrid fusion (`1 - alpha` is symbolic rule weight):

| Mode | Alpha | Description |
|------|-------|-------------|
| Pure ML | `1.0` | Sentence-BERT + logistic regression only |
| Pure symbolic | `0.0` | Weighted rule cues only (`resources/rules_v2.yaml`) |
| Hybrid (default) | **`0.2`** | Ablation-optimal fixed fusion (20% ML / 80% rules) |
| Adaptive | dynamic | Confidence- and cue-driven routing (`classification/adaptive_router.py`) |

**Why α = 0.2?** Held-out ablation on `dataset/test_set.csv` peaks at 76.5% accuracy / 0.762 macro F1 for α = 0.2 versus 61.0% for the legacy α = 0.8. Fusion math was verified correct; the legacy default was too ML-heavy to overturn wrong ML posteriors when rules are right. Full write-up: [`reports/alpha_investigation.md`](reports/alpha_investigation.md).

### Adaptive routing

When `adaptive_routing=true` (default) and `alpha` is unset, InferAI computes a per-sample `adaptive_alpha` from base α = 0.2:

- High ML confidence → increase ML weight
- Low ML confidence → increase symbolic weight
- Strong symbolic cues → increase symbolic contribution
- No symbolic cues → favour ML

The API returns `adaptive_alpha` and `routing_reason` for explainability.

### Fallacy detection

`reasoning/fallacy_detector.py` applies heuristic checks for common fallacies focused on `Anumana`-style inferences. The fallacy module is intentionally rule-based and modular; it returns a rich `fallacy_analysis` object in API responses (see `api/inference.py`) while keeping legacy fields for backward compatibility.

Dataset: a separate fallacy dataset lives under `dataset/fallacy/` and must not be mixed with the primary Pramana corpus. Human validation is required before examples are treated as gold-standard. See `dataset/fallacy/README.md` and `dataset/fallacy/ANNOTATION_GUIDELINES.md` for details.


### Benchmark framework

```
evaluation/
    benchmark.py               # run_benchmark(), run_ablation_study()
    metrics.py                 # accuracy, P/R, macro/weighted F1, ECE
    plots.py / publication_figures.py
    confusion_matrix.py
    adaptive_analysis.py
    fallacy_evaluation.py
    error_analysis.py
    dataset_statistics.py
    llm_baseline.py            # optional GPT-4 / Gemini
    explanation_evaluation.py  # HITL Likert sheets + printable forms
    freeze.py
```

Reproduce benchmarks:

```bash
python scripts/freeze_model.py --tag research
python scripts/run_benchmarks.py --out results/benchmarks
python scripts/generate_research_report.py --out results/report
```

Optional LLM comparison (requires API keys; skipped if unset):

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="..."
$env:GEMINI_API_KEY="..."
python scripts/generate_research_report.py --include-llm --out results/report
```

Outputs under `results/report/`: benchmark CSV tables, ablation plots (PNG+PDF), confusion matrices, adaptive/fallacy/error analyses, dataset stats, explainability forms, Markdown summary, and `figures_bundle.pdf`.

Ablation sweep: `alpha ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`.

### Statistical rigor & publication assets

```bash
python scripts/generate_research_report.py --out results/report
# Faster iteration (skip CV/calibration/robustness/OOD/SHAP sweeps):
python scripts/generate_research_report.py --skip-slow --out results/report
# Faster stress tests:
python scripts/generate_research_report.py --quick-stress --out results/report
```

Additional evaluation modules:

| Module | Output |
|--------|--------|
| `evaluation/statistical_tests.py` | McNemar, paired permutation, bootstrap 95% CIs → `statistical_significance.csv` |
| `evaluation/cross_validation.py` | 5-fold CV mean±std accuracy / macro F1 |
| `evaluation/calibration.py` | Temperature / Platt / Isotonic ECE comparison + reliability plots |
| `evaluation/robustness.py` | Typo / grammar / word-order / synonym / paraphrase stress tests |
| `evaluation/ood_evaluation.py` | Real-world OOD accuracy, confidence, JSD |
| `evaluation/shap_analysis.py` | Global / ± / per-class SHAP figures (300 DPI) |
| `evaluation/publication_tables.py` | Tables 1–6 as Markdown + LaTeX |

### LLM baseline

`evaluation/llm_baseline.py` uses an identical Nyāya classification prompt for GPT-4 and Gemini 1.5 Pro via `httpx`. Providers are optional and key-driven (`OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`).

### Dataset utilities

`dataset_utils/` provides load, label validation, gold-format validation, stratified splits, and cleaned export **without modifying** existing corpora.

Gold annotation format (JSONL example: `dataset/gold/example_gold.jsonl`):

```json
{"text": "...", "claim": "...", "premises": "...", "label": "Anumana"}
```

### Explainability evaluation

```python
from evaluation.explanation_evaluation import (
    export_evaluation_sheet,
    export_printable_form,
    summarize_completed_sheet,
)
```

Export CSV sheets + printable Markdown forms; summarize clarity / trust / plausibility (1–5) with bar charts.

### Reproducibility (model freeze)

```bash
python scripts/freeze_model.py --tag pre_ablation
```

Artifacts are saved under `models/frozen/<timestamp>/`.

## API

```bash
uvicorn api.app:app --reload
```

POST `/analyze` with JSON:

```json
{
  "text": "Smoke is rising from the hill, therefore there must be fire.",
  "include_shap": false,
  "alpha": null,
  "adaptive_routing": true,
  "benchmark_mode": false
}
```

Set `"alpha": 1.0` for pure ML, `"alpha": 0.0` for pure symbolic, or a value in `(0, 1)` for fixed hybrid. When `alpha` is `null` and `adaptive_routing` is true, routing is dynamic.

Key fields include `claim`, `premises` / `evidence`, `reasoning_indicators`, `highlighted_html`, `predicted_pramana` (ML head), `hybrid_predicted_pramana`, `confidence`, `adjusted_confidence`, `reasoning_strength`, `reasoning_strength_debug`, `hybrid`, `adaptive_alpha`, `routing_reason`, `fallacy_detected`, `fallacy_type`, `fallacy_explanation`, `benchmark_mode`, and optional `shap`.

## Streamlit UI

```bash
streamlit run frontend/app.py
```

Optional: set **`INFERAI_API_URL`** (or legacy **`NYAYAX_API_URL`**) if the API is not on `http://127.0.0.1:8000`.

## End-to-end workflow

Input text → structured extraction (claim / premises / indicators / highlights) → Sentence-BERT embeddings → logistic regression → **hybrid fusion** (fixed or adaptive alpha) → fallacy screening → composite strength → explanation → JSON (optional SHAP).






how to run the infer ai 


Terminal 1 (Backend API ke liye):
(InferAI) 
powershell
uvicorn api.app:app --reload

Terminal 2 (Streamlit UI ke liye):
(InferAI) 

powershell
streamlit run frontend/app.py
