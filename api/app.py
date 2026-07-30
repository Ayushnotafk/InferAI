"""
InferAI FastAPI service.

The ``/analyze`` endpoint preserves legacy keys while exposing richer
structure (premises, hybrid fusion, composite strength, highlights,
adaptive routing, and fallacy detection).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.inference import run_analysis

app = FastAPI(title="InferAI", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputText(BaseModel):
    text: str
    include_shap: bool = Field(
        default=False,
        description="If true, include SHAP summary for embedding dimensions (slower).",
    )
<<<<<<< HEAD
    alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fixed ML fusion weight (1.0=pure ML, 0.0=pure symbolic). "
        "Overrides adaptive routing when set.",
    )
    adaptive_routing: bool = Field(
        default=True,
        description="When true and alpha is unset, use dynamic neuro-symbolic routing.",
    )
    benchmark_mode: bool = Field(
        default=False,
        description="Benchmark flag; disables adaptive routing unless alpha is set.",
=======
    alpha: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Weight of the ML component in the hybrid fusion (1 - alpha is the weight of the symbolic rules).",
>>>>>>> 7fa6014e8b1ce70575677496d1136adac2916b14
    )


@app.post("/analyze")
def analyze_text(data: InputText):
<<<<<<< HEAD
    return run_analysis(
        data.text,
        include_shap=data.include_shap,
        alpha=data.alpha,
        adaptive_routing=data.adaptive_routing,
        benchmark_mode=data.benchmark_mode,
=======
    text = data.text

    structure = extract_argument_structure(text)
    claim = structure["claim"]
    premises = structure["premises"]
    reasoning_indicators = structure["reasoning_indicators"]
    highlighted_html = structure["highlighted_html"]

    detail = predict_pramana_detailed(text)
    ml_label = detail["ml_label"]
    ml_confidence = float(detail["ml_confidence"])
    embedding = detail["embedding"]
    proba = detail["probabilities"]
    classes = detail["classes"]

    hybrid = hybrid_fuse(proba, text, class_order=classes, ml_weight=data.alpha, rule_weight=1.0 - data.alpha)
    adjusted_confidence = float(hybrid["adjusted_confidence"])

    strength, strength_debug = composite_reasoning_strength(
        text,
        adjusted_confidence,
        claim,
        premises,
>>>>>>> 7fa6014e8b1ce70575677496d1136adac2916b14
    )
