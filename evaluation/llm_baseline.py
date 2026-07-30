"""
Optional LLM baseline for InferAI benchmarks (GPT-4 / Gemini).

Requires environment variables:
  - OPENAI_API_KEY   for GPT-4
  - GEMINI_API_KEY or GOOGLE_API_KEY for Gemini 1.5 Pro

Uses httpx only (no hard dependency on vendor SDKs). Skips providers
whose keys are missing.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import httpx
import pandas as pd

from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import compute_classification_metrics

VALID_LABELS = ("Anumana", "Pratyaksha", "Shabda", "Upamana")

LLMProvider = Literal["gpt-4", "gemini-1.5-pro"]

SHARED_PROMPT = """You are an expert in Nyāya epistemology applied to short English arguments.
Classify the argument into exactly one of these labels:
- Pratyaksha (direct perception / observation / measurement)
- Anumana (inference / causal reasoning)
- Upamana (analogy / comparison)
- Shabda (testimony / authority / citation)

Respond with ONLY the label name. No explanation.

Argument:
{text}
"""


def _normalize_label(raw: str) -> str | None:
    cleaned = (raw or "").strip().split()[0].strip(".,:;\"'`")
    for lab in VALID_LABELS:
        if cleaned.lower() == lab.lower():
            return lab
    for lab in VALID_LABELS:
        if lab.lower() in (raw or "").lower():
            return lab
    return None


def _call_openai(text: str, *, model: str = "gpt-4", timeout: float = 60.0) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": SHARED_PROMPT.format(text=text)},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _call_gemini(text: str, *, model: str = "gemini-1.5-pro", timeout: float = 60.0) -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    payload = {
        "contents": [{"parts": [{"text": SHARED_PROMPT.format(text=text)}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 16},
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


def predict_llm(
    text: str,
    provider: LLMProvider,
    *,
    retries: int = 2,
    sleep_s: float = 0.5,
) -> str:
    """Return a normalized pramāṇa label from the selected LLM provider."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if provider == "gpt-4":
                raw = _call_openai(text)
            else:
                raw = _call_gemini(text)
            label = _normalize_label(raw)
            if label is None:
                raise ValueError(f"Unparseable LLM label: {raw!r}")
            return label
        except Exception as exc:  # noqa: BLE001 — collect and retry
            last_err = exc
            time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"LLM prediction failed for {provider}: {last_err}")


def available_providers() -> list[LLMProvider]:
    """Return providers whose API keys are present."""
    out: list[LLMProvider] = []
    if os.environ.get("OPENAI_API_KEY"):
        out.append("gpt-4")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        out.append("gemini-1.5-pro")
    return out


def run_llm_benchmark(
    test_csv: str | Path,
    *,
    provider: LLMProvider,
    out_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate one LLM on a labeled CSV using the shared classification prompt.

    Returns metrics dict; optionally writes CSV + confusion matrix PNG.
    """
    df = pd.read_csv(test_csv)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    texts = df["text"].astype(str).tolist()
    y_true = df[label_col].astype(str).tolist()
    if max_samples is not None:
        texts, y_true = texts[:max_samples], y_true[:max_samples]

    y_pred: list[str] = []
    errors: list[dict[str, str]] = []
    for i, text in enumerate(texts):
        try:
            y_pred.append(predict_llm(text, provider))
        except Exception as exc:  # noqa: BLE001
            y_pred.append("Anumana")  # safe fallback; recorded as error
            errors.append({"index": str(i), "error": str(exc)})

    labels = list(VALID_LABELS)
    metrics = compute_classification_metrics(y_true, y_pred, labels=labels)
    result: dict[str, Any] = {
        "mode": f"llm_{provider}",
        "provider": provider,
        "n_samples": len(texts),
        "n_api_errors": len(errors),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": labels,
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = re.sub(r"[^a-z0-9]+", "_", provider)
        pd.DataFrame([result]).to_csv(out / f"benchmark_llm_{tag}.csv", index=False)
        plot_confusion_matrix(
            y_true,
            y_pred,
            labels,
            str(out / f"confusion_llm_{tag}.png"),
            title=f"LLM — {provider}",
        )
        pred_df = pd.DataFrame({"text": texts, "y_true": y_true, "y_pred": y_pred})
        pred_df.to_csv(out / f"predictions_llm_{tag}.csv", index=False)
        if errors:
            (out / f"llm_errors_{tag}.json").write_text(
                json.dumps(errors, indent=2), encoding="utf-8"
            )
    return result
