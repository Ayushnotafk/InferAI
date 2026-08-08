"""Centralized inference orchestration for the InferAI API."""

from __future__ import annotations

from typing import Any

from classification.adaptive_router import DEFAULT_ALPHA, resolve_alpha
from classification.hybrid_reasoning import hybrid_fuse, score_weighted_rules
from classification.predictor import predict_pramana_detailed
from confidence_engine.confidence import format_confidence, get_random_confidence
from explanation_engine.explainer import generate_explanation
from explanation_engine.shap_explainer import explain_embedding
from preprocessing.argument_structure import extract_argument_structure
from reasoning.fallacy_detector import detect_fallacies
from reasoning_strength.composite import composite_reasoning_strength


def run_analysis(
    text: str,
    *,
    include_shap: bool = False,
    alpha: float | None = None,
    adaptive_routing: bool = True,
    benchmark_mode: bool = False,
) -> dict[str, Any]:
    """
    Full InferAI analysis pipeline (structure → ML → hybrid → explanation).

  Parameters
  ----------
  alpha
      Fixed ML weight when ``adaptive_routing=False``. Pure ML = 1.0, pure rules = 0.0.
  adaptive_routing
      When True (default), compute dynamic alpha from ML confidence and symbolic cues.
  benchmark_mode
      When True, disables adaptive routing unless ``alpha`` is explicitly set.
    """
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

    raw_scores, matched = score_weighted_rules(text)
    use_adaptive = adaptive_routing and not benchmark_mode and alpha is None
    resolved_alpha, routing_mode, routing_reason = resolve_alpha(
        ml_confidence_pct=ml_confidence,
        matched_cues=matched,
        rule_scores=raw_scores,
        fixed_alpha=alpha,
        adaptive=use_adaptive,
        base_alpha=DEFAULT_ALPHA,
    )

    hybrid = hybrid_fuse(
        proba,
        text,
        class_order=classes,
        alpha=resolved_alpha,
        routing_mode=routing_mode,
        routing_reason=routing_reason,
    )

    random_conf = get_random_confidence()

    strength, strength_debug = composite_reasoning_strength(
        text,
        random_conf,
        claim,
        premises,
    )

    explanation = generate_explanation(hybrid["final_label"], strength_label=strength)
    fallacy = detect_fallacies(
        text,
        claim,
        premises,
        reasoning_indicators=reasoning_indicators,
    )

    payload: dict[str, Any] = {
        "input_text": text,
        "claim": claim,
        "evidence": premises,
        "premises": premises,
        "reasoning_indicators": reasoning_indicators,
        "highlighted_html": highlighted_html,
        "predicted_pramana": ml_label,
        "hybrid_predicted_pramana": hybrid["final_label"],
        "confidence": format_confidence(random_conf),
        "adjusted_confidence": format_confidence(random_conf),
        "reasoning_strength": strength,
        "reasoning_strength_debug": strength_debug,
        "explanation": explanation,
        "hybrid": hybrid,
        "adaptive_alpha": hybrid["alpha"],
        "routing_reason": hybrid["routing_reason"],
        "routing_mode": hybrid["routing_mode"],
        "fallacy_detected": fallacy["fallacy_detected"],
        "fallacy_type": fallacy["fallacy_type"],
        "fallacy_explanation": fallacy["fallacy_explanation"],
        "benchmark_mode": benchmark_mode,
    }

    if include_shap:
        try:
            payload["shap"] = explain_embedding(embedding)
        except FileNotFoundError as exc:
            payload["shap"] = {"error": str(exc)}

    return payload
