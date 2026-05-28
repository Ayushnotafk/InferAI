"""
InferAI FastAPI service.

The ``/analyze`` endpoint preserves legacy keys while exposing richer
structure (premises, hybrid fusion, composite strength, highlights).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classification.hybrid_reasoning import hybrid_fuse
from classification.predictor import predict_pramana_detailed
from confidence_engine.confidence import format_confidence
from explanation_engine.explainer import generate_explanation
from explanation_engine.shap_explainer import explain_embedding
from preprocessing.argument_structure import extract_argument_structure
from reasoning_strength.composite import composite_reasoning_strength

app = FastAPI(title="InferAI", version="0.3.0")

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
    alpha: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Weight of the ML component in the hybrid fusion (1 - alpha is the weight of the symbolic rules).",
    )


@app.post("/analyze")
def analyze_text(data: InputText):
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
    )

    explanation = generate_explanation(hybrid["final_label"], strength_label=strength)

    payload = {
        "input_text": text,
        "claim": claim,
        "evidence": premises,
        "premises": premises,
        "reasoning_indicators": reasoning_indicators,
        "highlighted_html": highlighted_html,
        "predicted_pramana": ml_label,
        "hybrid_predicted_pramana": hybrid["final_label"],
        "confidence": format_confidence(ml_confidence),
        "adjusted_confidence": format_confidence(adjusted_confidence),
        "reasoning_strength": strength,
        "reasoning_strength_debug": strength_debug,
        "explanation": explanation,
        "hybrid": hybrid,
    }

    if data.include_shap:
        try:
            payload["shap"] = explain_embedding(embedding)
        except FileNotFoundError as exc:
            payload["shap"] = {"error": str(exc)}

    return payload
