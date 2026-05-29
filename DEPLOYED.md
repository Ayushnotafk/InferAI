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

---

## Static & Interactive Frontend options

### A) Streamlit Cloud (Standalone / Embedded Mode) — RECOMMENDED ⭐
- You can deploy `frontend/app.py` directly to Streamlit Cloud.
- Since we have embedded the backend logic inside the frontend code:
  - If the `INFERAI_API_URL` environment variable is **not set**, the app automatically runs in **Embedded In-Process Mode** (using local Python modules directly).
  - This avoids Vercel limits and requires **no separate backend API deployment**!
- To deploy:
  1. Push your repository to GitHub.
  2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
  3. Create a new app, select your repo, branch `main`, and set the file path to `frontend/app.py`.
  4. Click **Deploy**. That's it!

### B) Streamlit Cloud (De-coupled API Mode)
- If you deploy your backend API separately (e.g. on Render, Koyeb, or a VM), you can point your Streamlit app to it by setting the environment variable in Streamlit Cloud's **Advanced Settings** -> **Secrets**:
  ```toml
  INFERAI_API_URL = "https://your-api-host.com"
  ```

### C) Static Web UI
- Deploy the contents of `web_frontend/` (HTML/JS/CSS) to Vercel, Netlify, or GitHub Pages.
- This static frontend requires a running backend API. Set the API endpoint inside the JS configuration.

---

## Local execution (Quick Commands)

Run standalone (frontend + backend in one command):
```powershell
python -m streamlit run frontend/app.py
```

Run with separate backend server:
1. Start API:
   ```powershell
   python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
   ```
2. Start Frontend pointing to the API:
   ```powershell
   $env:INFERAI_API_URL="http://127.0.0.1:8000"
   python -m streamlit run frontend/app.py
   ```

---

## Quick checklist

- `cd InferAI`
- `pip install -r requirements.txt`
- Deploy `frontend/app.py` to Streamlit Cloud for the full diagnostic tool (it will automatically run in-process!).


