"""
Rule-based pramāṇa cues fused with softmax outputs from logistic regression.

The fusion is intentionally simple and inspectable for coursework demos:

    fused = ml_weight * p_ml + rule_weight * p_rules

where ``p_rules`` is a soft distribution derived from substring cues.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Class order must match sklearn ``LabelEncoder`` fitted on
# ["Anumana", "Pratyaksha", "Shabda", "Upamana"] lexicographic — verify at runtime.
DEFAULT_CLASS_ORDER = ("Anumana", "Pratyaksha", "Shabda", "Upamana")


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.clip(v.astype(np.float64), 1e-12, None)
    return v / v.sum()


def detect_pattern_signals(text: str) -> dict[str, Any]:
    """Lightweight boolean and count signals for interpretability."""
    t = text.lower()

    authority_patterns = [
        r"\baccording to\b",
        r"\bexperts?\b",
        r"\bresearchers?\b",
        r"\bstud(?:y|ies)\b",
        r"\breport(ed|s)?\b",
        r"\bwho\b|\bcdc\b|\b(fda|nasa)\b",
        r"\bthe (handbook|manual|court|statute)\b",
        r"\bprofessor\b|\bteacher\b",
    ]
    analogy_patterns = [
        r"\bis like\b",
        r"\bworks like\b",
        r"\bacts like\b",
        r"\bsimilar to\b",
        r"\banalogous to\b",
        r"\bjust as\b",
        r"\blike a\b",
    ]
    inference_patterns = [
        r"\btherefore\b",
        r"\bhence\b",
        r"\bthus\b",
        r"\bconsequently\b",
        r"\bbecause\b",
        r"\bsince\b",
        r"\bimplies\b",
        r"\bwhich suggests\b",
        r"\bmost likely\b",
    ]
    observation_patterns = [
        r"\bi (saw|heard|observed|noticed|smelled|felt)\b",
        r"\bthe (sensor|display|log|image|recording)\b",
        r"\bwe measured\b",
        r"\btelemetry\b",
        r"\bfield notes\b",
        r"\bvisible\b|\baudible\b",
    ]

    def hits(patterns: list[str]) -> int:
        return sum(1 for p in patterns if re.search(p, t))

    return {
        "authority_hits": hits(authority_patterns),
        "analogy_hits": hits(analogy_patterns),
        "inference_hits": hits(inference_patterns),
        "observation_hits": hits(observation_patterns),
    }


def rule_distribution(
    text: str, class_order: tuple[str, ...] = DEFAULT_CLASS_ORDER
) -> np.ndarray:
    """
    Map heuristic hits to a probability vector over ``class_order``.

    This is not a standalone classifier; it nudges the ML posterior.
    """
    s = detect_pattern_signals(text)
    scores = np.ones(len(class_order), dtype=np.float64)

    idx = {name: i for i, name in enumerate(class_order)}

    scores[idx["Pratyaksha"]] += 1.4 * s["observation_hits"]
    scores[idx["Anumana"]] += 1.2 * s["inference_hits"]
    scores[idx["Upamana"]] += 1.3 * s["analogy_hits"]
    scores[idx["Shabda"]] += 1.3 * s["authority_hits"]

    # Mild deference if multiple families fire (mixed style).
    mix_penalty = 0.85 if sum(v > 0 for v in s.values()) >= 3 else 1.0
    scores *= mix_penalty

    return _norm_vec(scores)


def hybrid_fuse(
    ml_probs: np.ndarray,
    text: str,
    *,
    class_order: tuple[str, ...] | list[str] | None = None,
    ml_weight: float = 0.8,
    rule_weight: float = 0.2,
) -> dict[str, Any]:
    """
    Fuse ML softmax with rule-based distribution.

    Parameters
    ----------
    ml_probs
        1D array of class probabilities aligned with ``class_order``.
    text
        Original user text for cue extraction.
    class_order
        Names aligned with ``ml_probs`` indices. If ``None``, use default tuple
        (callers should pass ``label_encoder.classes_`` from disk).
    """
    order = tuple(class_order) if class_order is not None else DEFAULT_CLASS_ORDER
    p_ml = _norm_vec(np.asarray(ml_probs, dtype=np.float64).reshape(-1))
    p_rules = rule_distribution(text, class_order=order)
    fused = ml_weight * p_ml + rule_weight * p_rules
    fused = _norm_vec(fused)
    best_i = int(np.argmax(fused))
    return {
        "class_order": list(order),
        "ml_probs": p_ml.tolist(),
        "rule_probs": p_rules.tolist(),
        "fused_probs": fused.tolist(),
        "final_label": order[best_i],
        "adjusted_confidence": float(fused[best_i] * 100.0),
        "pattern_signals": detect_pattern_signals(text),
        "weights": {"ml": ml_weight, "rules": rule_weight},
    }
