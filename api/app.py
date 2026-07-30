"""
InferAI FastAPI service.

The ``/analyze`` endpoint preserves legacy keys while exposing richer
structure (premises, hybrid fusion, composite strength, highlights,
adaptive routing, and fallacy detection).
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.inference import run_analysis

app = FastAPI(title="InferAI", version="0.4.0")


class InputText(BaseModel):
    text: str
    include_shap: bool = Field(
        default=False,
        description="If true, include SHAP summary for embedding dimensions (slower).",
    )
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
    )


@app.post("/analyze")
def analyze_text(data: InputText):
    return run_analysis(
        data.text,
        include_shap=data.include_shap,
        alpha=data.alpha,
        adaptive_routing=data.adaptive_routing,
        benchmark_mode=data.benchmark_mode,
    )
