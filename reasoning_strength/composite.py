"""
Composite reasoning-strength estimation.

Combines softmax confidence with lightweight structural cues so that
strength is not purely thresholded on classifier probability.
"""

from __future__ import annotations

from typing import Any

from classification.hybrid_reasoning import detect_pattern_signals


def legacy_strength_from_confidence(confidence: float) -> str:
    """Legacy confidence-only mapping (kept for simple callers)."""
    if confidence > 85:
        return "Strong"
    if confidence > 60:
        return "Moderate"
    return "Weak"


def _evidence_completeness_score(claim: str, premises: str) -> float:
    """Heuristic in [0, 1]: longer, non-trivial premise text scores higher."""
    p = (premises or "").strip()
    if len(p) < 12:
        return 0.25
    if len(p) < 40:
        return 0.55
    if len(p) < 120:
        return 0.75
    return 0.9


def _connector_density(text: str) -> float:
    t = text.lower()
    cues = [
        "therefore",
        "because",
        "since",
        "hence",
        "thus",
        "implies",
        "similar to",
        "according to",
    ]
    return min(1.0, sum(t.count(c) for c in cues) / 3.0)


def _hedging_penalty(text: str) -> float:
    t = text.lower()
    hedges = ["maybe", "perhaps", "possibly", "might", "could be", "i think", "sort of", "kind of"]
    h = sum(t.count(h) for h in hedges)
    return min(0.35, 0.07 * h)


def composite_reasoning_strength(
    text: str,
    base_confidence: float,
    claim: str,
    premises: str,
) -> tuple[str, dict[str, Any]]:
    """
    Return (strength_label, debug_scores).

    ``base_confidence`` is expected on a 0–100 scale (e.g., max softmax * 100).
    """
    signals = detect_pattern_signals(text)
    structure = _evidence_completeness_score(claim, premises)
    connectors = _connector_density(text)
    hedge = _hedging_penalty(text)

    c = max(0.0, min(1.0, float(base_confidence) / 100.0))

    score = (
        0.45 * c
        + 0.18 * structure
        + 0.12 * min(1.0, sum(1 for v in signals.values() if v > 0) / 4.0)
        + 0.15 * connectors
        + 0.10 * min(1.0, len(text) / 400.0)
        - hedge
    )
    score = max(0.0, min(1.0, score))

    if score >= 0.72:
        label = "Strong"
    elif score >= 0.48:
        label = "Moderate"
    else:
        label = "Weak"

    debug = {
        "composite_score": round(score, 4),
        "confidence_component": round(c, 4),
        "evidence_completeness": round(structure, 4),
        "connector_density": round(connectors, 4),
        "hedging_penalty": round(hedge, 4),
        "pattern_signal_count": int(sum(1 for v in signals.values() if v > 0)),
    }
    return label, debug
