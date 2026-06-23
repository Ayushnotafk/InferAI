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

## API

```bash
uvicorn api.app:app --reload
```

POST `/analyze` with JSON:

```json
{
  "text": "Smoke is rising from the hill, therefore there must be fire.",
  "include_shap": false
}
```

Key fields include `claim`, `premises` / `evidence`, `reasoning_indicators`, `highlighted_html`, `predicted_pramana` (ML head), `hybrid_predicted_pramana`, `confidence`, `adjusted_confidence`, `reasoning_strength`, `reasoning_strength_debug`, `hybrid`, and optional `shap`.

## Streamlit UI

```bash
streamlit run frontend/app.py
```

Optional: set **`INFERAI_API_URL`** (or legacy **`NYAYAX_API_URL`**) if the API is not on `http://127.0.0.1:8000`.

## End-to-end workflow

Input text → structured extraction (claim / premises / indicators / highlights) → Sentence-BERT embeddings → logistic regression → **hybrid fusion (0.8 ML + 0.2 rules)** → composite strength → explanation → JSON (optional SHAP).
