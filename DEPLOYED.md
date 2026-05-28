## InferAI Deployment Guide

This document explains how to run and deploy the InferAI project. There are two primary components:

1. `api/app.py` — FastAPI backend that exposes a POST `/analyze` endpoint.
2. `frontend/app.py` — Streamlit frontend UI that calls the backend API.

If you only want to deploy the backend, deploying from the repository root is sufficient (a `vercel.json` is included for Vercel deployments).

---

## Folders of interest

- `InferAI/` — repository root
- `InferAI/api/` — backend service
- `InferAI/frontend/` — Streamlit UI (local interactive frontend)
- `InferAI/web_frontend/` — optional static web UI (HTML/JS)

## What to deploy where

- Deploy `api/app.py` on a Python-capable host (Vercel, AWS, Azure, etc.).
- Run `frontend/app.py` locally, or deploy it on Streamlit Cloud / a VM / container.
- Deploy `web_frontend/` (static files) to any static host (Vercel, Netlify, GitHub Pages).
- Keep model files (`models/infer_model.pkl`, `models/label_encoder.pkl`, `models/shap_background.npy`) with the backend; the API loads them at runtime.

---

## Local deployment (quick)

1) Change to the project root:

```powershell
cd c:\Users\munta\Desktop\ak-pr\InferAI
```

2) Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

4) Start the backend API:

```powershell
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

5) Start the Streamlit frontend:

```powershell
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

URLs:

- Backend API: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`
- Frontend UI: `http://127.0.0.1:8501`

---

## Deploying the API to Vercel

This repository includes a `vercel.json` that enables deploying the backend directly to Vercel using the `@vercel/python` builder.

Steps to deploy:

1. Push your repository to GitHub.
2. Create a new project in Vercel and connect the GitHub repo.
3. Vercel will detect the Python API and run the build.
4. After deployment you will receive a public Vercel URL for your API.

Example `vercel.json` used by this repo:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "(.*)",
      "dest": "api/app.py"
    }
  ]
}
```

This configuration makes `api/app.py` the entrypoint for the deployment.

---

## Static frontend options

A) Streamlit Cloud

- Deploy `frontend/app.py` to Streamlit Cloud for an interactive GUI.
- The frontend requires a running backend; set `INFERAI_API_URL` to point to your deployed API.

B) Static site hosts

- Deploy the contents of `web_frontend/` (HTML/JS/CSS) to Vercel, Netlify, or GitHub Pages.
- Configure the static frontend to call your deployed backend URL.

---

## Environment variable for API URL

If your frontend needs to point to a custom backend host, set the environment variable before launching Streamlit:

PowerShell:

```powershell
$env:INFERAI_API_URL="http://your-api-host:8000"
streamlit run frontend/app.py --server.port 8501 --server.headless true
```

Linux / macOS:

```bash
INFERAI_API_URL=http://your-api-host:8000 streamlit run frontend/app.py --server.port 8501 --server.headless true
```

---

## Recommended deployment path

1. Deploy `api/app.py` (backend) to Vercel or another Python host.
2. Run or deploy `frontend/app.py` (Streamlit) locally or on Streamlit Cloud, pointing it to the backend URL.
3. Alternatively, deploy a static UI from `web_frontend/` and configure its API base URL.

---

## Quick checklist

- `cd InferAI`
- `pip install -r requirements.txt`
- `python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000`
- `streamlit run frontend/app.py --server.port 8501 --server.headless true`
- Connect repo to Vercel for API deployment
- If deploying a static frontend, use `web_frontend/`
- Ensure `INFERAI_API_URL` points to the deployed backend

