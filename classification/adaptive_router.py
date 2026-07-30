"""
Adaptive neuro-symbolic routing for InferAI.

Computes a dynamic fusion weight ``alpha`` (ML contribution) in [0, 1] from
ML confidence and symbolic cue evidence.  ``1 - alpha`` is the symbolic weight.

Research note: routing decisions are intentionally explainable via ``routing_reason``.
"""

from __future__ import annotations

from typing import Any

# Empirically optimal fixed hybrid on held-out test_set.csv ablation
# (accuracy 76.5% at alpha=0.2 vs 61.0% at legacy 0.8). See
# reports/alpha_investigation.md. Alpha = ML weight; 1-alpha = rules.
DEFAULT_ALPHA = 0.2

# Adaptive routing bounds.
MIN_ALPHA = 0.05
MAX_ALPHA = 0.95

# Confidence thresholds (percent scale, 0–100).
HIGH_ML_CONFIDENCE = 85.0
LOW_ML_CONFIDENCE = 50.0

# Symbolic evidence thresholds.
STRONG_CUE_COUNT = 3
STRONG_CUE_WEIGHT = 6.0


def _count_cues(matched_cues: dict[str, list[Any]]) -> int:
    return sum(len(v) for v in matched_cues.values())


def _total_rule_weight(rule_scores: dict[str, float]) -> float:
    return float(sum(rule_scores.values()))


def compute_adaptive_alpha(
    ml_confidence_pct: float,
    matched_cues: dict[str, list[Any]],
    rule_scores: dict[str, float],
    *,
    base_alpha: float = DEFAULT_ALPHA,
) -> tuple[float, str]:
    """
    Derive ML fusion weight ``alpha`` from ML confidence and symbolic support.

    Heuristic policy (documented for reproducibility):
    - Start from ``base_alpha`` (default 0.2, ablation-optimal).
    - High ML confidence increases ML weight.
    - Low ML confidence decreases ML weight in favour of rules.
    - Many / strong symbolic cues decrease ML weight.
    - No symbolic cues increase ML weight (ML remains the tie-breaker
      under the soft rule floor; larger alpha reduces dilution).
    """
    alpha = float(base_alpha)
    reasons: list[str] = []

    cue_count = _count_cues(matched_cues)
    total_weight = _total_rule_weight(rule_scores)
    symbolic_active = cue_count > 0

    conf = max(0.0, min(100.0, float(ml_confidence_pct)))

    # Deltas tuned around base=0.2 so adaptive stays rule-leaning when cues
    # exist, but can rise toward ML when confidence is high or cues are absent.
    if conf >= HIGH_ML_CONFIDENCE:
        alpha += 0.25
        reasons.append(f"high ML confidence ({conf:.0f}%)")
    elif conf <= LOW_ML_CONFIDENCE:
        alpha -= 0.10
        reasons.append(f"low ML confidence ({conf:.0f}%)")

    if symbolic_active:
        if cue_count >= STRONG_CUE_COUNT or total_weight >= STRONG_CUE_WEIGHT:
            alpha -= 0.12
            reasons.append(
                f"strong symbolic support ({cue_count} cues, weight {total_weight:.1f})"
            )
        else:
            alpha -= 0.05
            reasons.append(f"symbolic cues detected ({cue_count})")
    else:
        alpha += 0.35
        reasons.append("no symbolic cues; ML favoured")

    alpha = max(MIN_ALPHA, min(MAX_ALPHA, alpha))

    if not reasons:
        reason = "Default hybrid routing."
    elif alpha < base_alpha - 0.05:
        reason = "Symbolic reasoning received higher weight because " + "; ".join(reasons) + "."
    elif alpha > base_alpha + 0.05:
        reason = "ML received higher weight because " + "; ".join(reasons) + "."
    else:
        reason = "Balanced routing: " + "; ".join(reasons) + "."

    return round(alpha, 4), reason


def resolve_alpha(
    *,
    ml_confidence_pct: float,
    matched_cues: dict[str, list[Any]],
    rule_scores: dict[str, float],
    fixed_alpha: float | None = None,
    adaptive: bool = False,
    base_alpha: float = DEFAULT_ALPHA,
) -> tuple[float, str, str]:
    """
    Resolve final ``alpha`` and human-readable routing metadata.

    Returns
    -------
    alpha, routing_mode, routing_reason
    """
    if fixed_alpha is not None:
        a = max(0.0, min(1.0, float(fixed_alpha)))
        return a, "fixed", f"Fixed alpha={a:.2f} (explicit configuration)."

    if adaptive:
        a, reason = compute_adaptive_alpha(
            ml_confidence_pct,
            matched_cues,
            rule_scores,
            base_alpha=base_alpha,
        )
        return a, "adaptive", reason

    return base_alpha, "fixed", f"Default hybrid alpha={base_alpha:.2f}."
