# InferAI

InferAI is an explainable argument analysis system. It classifies short English arguments into four pramāṇa-style labels (**Pratyaksha**, **Anumana**, **Upamana**, **Shabda**), fuses **softmax outputs with lightweight rule cues** (`classification/hybrid_reasoning.py`), estimates **ML confidence** and **adjusted hybrid confidence**, extracts **claim / premises**, surfaces **reasoning indicators** and **HTML-safe cue highlighting**, optionally returns **SHAP** over the Sentence-BERT embedding space, and emits a short **varied natural-language explanation** (`explanation_engine/explainer.py`).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start (Run the Entire Project)](#quick-start-run-the-entire-project)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Generate the Dataset](#1-generate-the-dataset)
  - [2. Train the Model](#2-train-the-model)
  - [3. Generate Evaluation Report](#3-generate-evaluation-report)
  - [4. Start the Backend API](#4-start-the-backend-api)
  - [5. Start the Frontend UI](#5-start-the-frontend-ui)
- [API Reference](#api-reference)
- [Data Layout](#data-layout)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
InferAI/
├── api/
│   └── app.py                  # FastAPI backend (/analyze endpoint)
├── classification/
│   ├── embedder.py             # Sentence-BERT encoder (all-MiniLM-L6-v2)
│   ├── hybrid_reasoning.py     # ML + rule-based fusion logic
│   ├── predictor.py            # Prediction wrapper
│   └── train_model.py          # Model training & evaluation artifacts
├── confidence_engine/
│   └── confidence.py           # Confidence formatting
├── dataset/
│   ├── labeled_corpus.py       # Builds master_dataset.csv and test_set.csv
│   ├── labeling_guidelines.md  # Annotator instructions and definitions
│   ├── master_dataset.csv      # Curated seed rows
│   ├── test_set.csv            # Held-out evaluation set
│   ├── raw/
│   │   └── infer_dataset.csv   # Training table (text, label)
│   └── processed/
├── explanation_engine/
│   ├── explainer.py            # Natural-language explanation generator
│   └── shap_explainer.py       # SHAP over Sentence-BERT embeddings
├── frontend/
│   ├── app.py                  # Streamlit UI (main application)
│   ├── components.py           # Reusable UI components
│   └── theme.py                # Dashboard CSS theme (glassmorphism)
├── models/
│   ├── infer_model.pkl         # Trained logistic regression model
│   ├── label_encoder.pkl       # Label encoder
│   ├── shap_background.npy     # SHAP background data
│   └── evaluation/
│       ├── metrics.json        # Scalar evaluation metrics
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       ├── confusion_matrix_test_set.png
│       ├── class_distribution.png
│       ├── per_class_metrics.png
│       └── train_test_accuracy.png
├── preprocessing/
│   ├── argument_structure.py   # Claim/premise/indicator extraction
│   ├── claim_extractor.py
│   ├── cleaner.py
│   └── evidence_extractor.py
├── reasoning_strength/
│   ├── composite.py            # Composite reasoning strength scorer
│   └── strength_checker.py
├── reports/
│   ├── generate_final_evaluation_report.py
│   └── final_evaluation_report.md
├── dataset_generator.py        # Synthetic data generation pipeline
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Prerequisites

- **Python 3.10+** (tested with 3.13)
- **pip** (Python package manager)
- **Git** (to clone the repository)
- ~1 GB disk space (for Sentence-BERT model download on first run)
- Internet connection (first run only — to download the `all-MiniLM-L6-v2` model from HuggingFace)

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd InferAI
```

### 2. (Recommended) Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling |
| `scikit-learn` | Logistic regression classifier |
| `sentence-transformers` | Sentence-BERT embeddings (`all-MiniLM-L6-v2`) |
| `joblib` | Model serialization |
| `fastapi`, `uvicorn` | Backend REST API |
| `httpx` | HTTP client for frontend → API communication |
| `shap` | Explainability (SHAP values) |
| `streamlit` | Frontend web UI |
| `matplotlib`, `seaborn` | Evaluation plots |

---

## Quick Start (Run the Entire Project)

> **If the model is already trained** (i.e., `models/infer_model.pkl` exists), you can skip Steps 1-3 and jump directly to running the API + Frontend.

### All-in-one (Windows PowerShell):

```powershell
# Terminal 1 — Start the Backend API
cd InferAI
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2 — Start the Frontend UI
cd InferAI
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

Then open your browser:
- **API Docs (Swagger):** http://127.0.0.1:8000/docs
- **Frontend UI:** http://127.0.0.1:8501

### All-in-one (macOS / Linux):

```bash
# Terminal 1 — Start the Backend API
cd InferAI
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2 — Start the Frontend UI
cd InferAI
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

---

## Step-by-Step Guide

### 1. Generate the Dataset

This script materializes the master dataset, synthesizes additional training rows (~4200 total), and writes the training table to `dataset/raw/infer_dataset.csv`.

```bash
cd InferAI
python dataset_generator.py
```

**Expected output:**
```
Wrote ...\dataset\raw\infer_dataset.csv with 4200 rows (master=..., synthetic=...)
```

### 2. Train the Model

Train the Sentence-BERT + Logistic Regression classifier:

```bash
python classification/train_model.py
```

**Expected output:**
```
Train accuracy: 0.99xx
Test accuracy:  0.98xx
              precision    recall  f1-score   support
    Anumana     ...
 Pratyaksha     ...
     Shabda     ...
    Upamana     ...

SHAP background saved (200 rows)
Evaluation artifacts saved under models\evaluation
Model saved successfully
```

**Artifacts generated:**
- `models/infer_model.pkl` — Trained model
- `models/label_encoder.pkl` — Label encoder
- `models/shap_background.npy` — SHAP background matrix
- `models/evaluation/metrics.json` — All metrics in JSON
- `models/evaluation/classification_report.txt` — Text classification report
- `models/evaluation/*.png` — Confusion matrix, class distribution, accuracy, per-class metric plots

### 3. Generate Evaluation Report

```bash
python reports/generate_final_evaluation_report.py
```

**Output:** `reports/final_evaluation_report.md` — A comprehensive Markdown report with quantitative metrics, held-out test results, and qualitative failure analysis.

### 4. Start the Backend API

```bash
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

- The API will be available at **http://127.0.0.1:8000**
- Interactive Swagger docs at **http://127.0.0.1:8000/docs**
- On first startup, the Sentence-BERT model will be loaded (takes a few seconds)

**You should see:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

**Quick test (in a separate terminal):**

```bash
# Using curl
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Smoke rises from the hill, therefore there must be fire.\", \"include_shap\": false}"
```

```powershell
# Using PowerShell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/analyze" -Method POST -ContentType "application/json" -Body '{"text": "Smoke rises from the hill, therefore there must be fire.", "include_shap": false}'
```

```python
# Using Python
import httpx, json
r = httpx.post("http://127.0.0.1:8000/analyze", json={"text": "Smoke rises from the hill, therefore there must be fire.", "include_shap": False})
print(json.dumps(r.json(), indent=2))
```

### 5. Start the Frontend UI

> **Important:** The backend API (Step 4) must be running before starting the frontend.

```bash
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

- The frontend will be available at **http://127.0.0.1:8501**
- By default, it connects to the API at `http://127.0.0.1:8000`
- To use a custom API URL, set the environment variable:

```powershell
# Windows PowerShell
$env:INFERAI_API_URL="http://your-api-host:8000"
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

```bash
# macOS / Linux
INFERAI_API_URL=http://your-api-host:8000 streamlit run frontend/app.py --server.port 8501 --server.headless true
```

---

## Vercel Deployment

This repository already includes Vercel configuration for the backend API and a static frontend prototype.

- `vercel.json` at the repository root deploys the FastAPI backend from `api/app.py`.
- `web_frontend/vercel.json` can deploy the static HTML/JS frontend from `web_frontend/`.

To deploy the API on Vercel:

1. Connect your Git repository to Vercel.
2. Use the existing `vercel.json` at the repo root.
3. Deploy the project.

Once deployed, the API will be available at a Vercel URL, and you can point the Streamlit client to that URL using `INFERAI_API_URL`.

If you want a purely static browser demo, deploy `web_frontend/` as a separate Vercel project and configure its `API URL` field to the deployed FastAPI endpoint.

---

## API Reference

### `POST /analyze`

Analyze an argument text and return structured classification results.

**Request Body:**
```json
{
  "text": "Smoke is rising from the hill, therefore there must be fire.",
  "include_shap": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `string` | *(required)* | The argument text to analyze |
| `include_shap` | `boolean` | `false` | Include SHAP values (slower) |

**Response Fields:**

| Field | Description |
|---|---|
| `input_text` | Echo of the input text |
| `claim` | Extracted claim from the argument |
| `premises` / `evidence` | Extracted supporting premises |
| `reasoning_indicators` | Detected reasoning cues (e.g., `inferential_connectives`) |
| `highlighted_html` | HTML with `<mark>` tags on cue words |
| `predicted_pramana` | ML-only classification label |
| `hybrid_predicted_pramana` | Final label after hybrid fusion (ML + rules) |
| `confidence` | ML-only confidence (%) |
| `adjusted_confidence` | Hybrid-adjusted confidence (%) |
| `reasoning_strength` | Composite strength: `Strong`, `Moderate`, or `Weak` |
| `reasoning_strength_debug` | Breakdown of strength components |
| `explanation` | Natural-language explanation |
| `hybrid` | Full hybrid fusion details (probabilities, weights, signals) |
| `shap` | *(optional)* SHAP embedding contributions |

---

## Data Layout

| Path | Role |
|---|---|
| `dataset/labeling_guidelines.md` | Annotator instructions and definitions |
| `dataset/labeled_corpus.py` | Builds `master_dataset.csv` and `test_set.csv` |
| `dataset/master_dataset.csv` | Curated seed rows (`text`, `pramana_label`, `strength_label`, …) |
| `dataset/test_set.csv` | Held-out evaluation set (kept out of `infer_dataset.csv`) |
| `dataset/raw/infer_dataset.csv` | Training table (`text`, `label`) produced by `dataset_generator.py` |

---

## Architecture

```
Input text
    │
    ▼
┌──────────────────────────────┐
│  Structured Extraction       │
│  (claim / premises /         │
│   indicators / highlights)   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Sentence-BERT Embeddings    │
│  (all-MiniLM-L6-v2)         │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Logistic Regression         │
│  (ML classification head)    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Hybrid Fusion               │
│  (0.8 ML + 0.2 Rules)       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Composite Strength Scorer   │
│  (confidence + evidence +    │
│   connector density +        │
│   hedging penalty)           │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Explanation Engine          │
│  (natural-language summary)  │
└──────────┬───────────────────┘
           │
           ▼
      JSON Response
      (optional SHAP)
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Make sure you run commands from the `InferAI/` directory and have installed all dependencies: `pip install -r requirements.txt` |
| `FileNotFoundError: infer_model.pkl` | Train the model first: `python classification/train_model.py` |
| `FileNotFoundError: infer_dataset.csv` | Generate the dataset first: `python dataset_generator.py` |
| Streamlit asks for email | Use `--server.headless true` flag or press Enter to skip |
| Frontend shows "Could not reach the analysis service" | Make sure the backend API is running on port 8000 first |
| Slow first startup | The Sentence-BERT model (`~90 MB`) downloads from HuggingFace on first run. Subsequent runs use the cached model |
| Port already in use | Change the port: `--port 8001` for API, or `--server.port 8502` for Streamlit |
| HF rate limit warning | Set `HF_TOKEN` environment variable with your HuggingFace token (optional, for faster downloads) |

### Verify Everything is Working

```bash
# 1. Check API is running
curl http://127.0.0.1:8000/docs

# 2. Test analysis endpoint
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "According to experts, this is true.", "include_shap": false}'

# 3. Check frontend is running
# Open http://127.0.0.1:8501 in your browser
```

---

## Deployment Guide

### कौन सा फ़ोल्डर deploy करें?

- Backend API: `InferAI/api/app.py`
- Local frontend: `InferAI/frontend/app.py`
- Optional static demo: `InferAI/web_frontend/`

### Vercel पर deploy करने का fastest तरीका

1. GitHub पर repository push करें।
2. Vercel में नया प्रोजेक्ट बनाएं और repository connect करें।
3. Root में मौजूद `vercel.json` का उपयोग करें।
4. Deploy के बाद Vercel URL पर आपका FastAPI backend चालू होगा।

### Local deployment steps

```powershell
cd c:\Users\munta\Desktop\ak-pr\InferAI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

### अगर frontend अलग host करना हो

`INFERAI_API_URL` को backend endpoint पर सेट करें:

```powershell
$env:INFERAI_API_URL="http://127.0.0.1:8000"
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

---

## End-to-End Workflow

Input text → structured extraction (claim / premises / indicators / highlights) → Sentence-BERT embeddings → logistic regression → **hybrid fusion (0.8 ML + 0.2 rules)** → composite strength → explanation → JSON (optional SHAP).

---

## License

See repository root for license information.
