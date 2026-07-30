"""
Rule-based pramāṇa cues fused with softmax outputs from logistic regression.

Weighted semantic cues (not binary keyword hits) produce a soft rule distribution:

    fused = ml_weight * p_ml + rule_weight * p_rules

When ML and symbolic argmax agree, hybrid confidence is boosted slightly; on
disagreement it is reduced slightly while both labels remain in the response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# Class order must match sklearn ``LabelEncoder`` fitted on
# ["Anumana", "Pratyaksha", "Shabda", "Upamana"] lexicographic — verify at runtime.
DEFAULT_CLASS_ORDER = ("Anumana", "Pratyaksha", "Shabda", "Upamana")

# Agreement / disagreement multipliers applied to fused max-probability (0–100 scale).
AGREEMENT_CONF_BOOST = 1.10
AGREEMENT_CONF_ADD = 4.0
DISAGREEMENT_CONF_MULT = 0.90

# Extra weight per additional distinct cue within the same pramāṇa (diminishing).
MULTI_CUE_BONUS = 0.12
MULTI_CUE_BONUS_CAP = 0.36


@dataclass(frozen=True)
class WeightedRule:
    pattern: str
    weight: float
    pramana: str
    cue: str
    flags: int = 0


# (pattern, weight, pramana, cue)
_RULE_SPECS: list[tuple[str, float, str, str]] = [
    # --- Pratyaksha (direct perception / measurement) ---
    (r"\bobserved\b", 1.6, "Pratyaksha", "observed"),
    (r"\bmeasured\b", 1.8, "Pratyaksha", "measured"),
    (r"\brecorded\b", 1.5, "Pratyaksha", "recorded"),
    (r"\bdetected\b", 1.6, "Pratyaksha", "detected"),
    (r"\bsensor(?:s| data)?\b", 2.0, "Pratyaksha", "sensor"),
    (r"\btemperature(?:s)?\b", 1.4, "Pratyaksha", "temperature"),
    (r"\bpressure(?:s)?\b", 1.4, "Pratyaksha", "pressure"),
    (r"\blaborator(?:y|ies)\b", 2.2, "Pratyaksha", "laboratory"),
    (r"\bexperiment(?:s|al)?\b", 2.1, "Pratyaksha", "experiment"),
    (r"\bsample(?:s)?\b", 1.7, "Pratyaksha", "sample"),
    (r"\bmicroscope\b", 2.3, "Pratyaksha", "microscope"),
    (r"\bevidence collected\b", 2.0, "Pratyaksha", "evidence collected"),
    (r"\bsurvey results\b", 1.9, "Pratyaksha", "survey results"),
    (r"\bstatistics\b", 1.6, "Pratyaksha", "statistics"),
    (r"\bmeasured values\b", 2.2, "Pratyaksha", "measured values"),
    (r"\binspection\b", 1.5, "Pratyaksha", "inspection"),
    (r"\bmonitoring\b", 1.5, "Pratyaksha", "monitoring"),
    (r"\bi (?:saw|heard|observed|noticed|smelled|felt)\b", 1.8, "Pratyaksha", "first-person observation"),
    (r"\bthe (?:sensor|display|log|image|recording)\b", 1.7, "Pratyaksha", "instrument reading"),
    (r"\bwe measured\b", 2.0, "Pratyaksha", "we measured"),
    (r"\btelemetry\b", 1.8, "Pratyaksha", "telemetry"),
    (r"\bfield notes\b", 1.6, "Pratyaksha", "field notes"),
    (r"\bvisible\b|\baudible\b", 1.4, "Pratyaksha", "visible/audible"),
    # --- Shabda (testimony / authority) — tiered ``according to`` ---
    (
        r"\baccording to\s+(?:who|world health organization|cdc|unesco|fda|nasa|"
        r"the government|government|official report|court judgement|court judgment)\b",
        3.4,
        "Shabda",
        "according to official body",
    ),
    (
        r"\baccording to\s+(?:experts?|researchers?|scientists?|the study|a study|research|"
        r"medical guidelines?|journal|published)\b",
        2.7,
        "Shabda",
        "according to expert/study",
    ),
    (r"\baccording to\s+(?:my friend|my mom|my dad|a friend|some guy)\b", 0.7, "Shabda", "according to informal source"),
    (r"\baccording to\b", 1.5, "Shabda", "according to (general)"),
    (r"\breported by\b", 2.0, "Shabda", "reported by"),
    (r"\bresearch shows\b", 2.5, "Shabda", "research shows"),
    (r"\bstudy found\b", 2.5, "Shabda", "study found"),
    (r"\bjournal\b", 2.0, "Shabda", "journal"),
    (r"\bpublished\b", 1.9, "Shabda", "published"),
    (r"\bWHO\b", 3.2, "Shabda", "WHO"),
    (r"\bUNESCO\b", 3.0, "Shabda", "UNESCO"),
    (r"\bCDC\b", 3.0, "Shabda", "CDC"),
    (r"\bgovernment\b", 2.4, "Shabda", "government"),
    (r"\bofficial report\b", 2.6, "Shabda", "official report"),
    (r"\bcourt judg(?:e)?ment\b", 2.7, "Shabda", "court judgement"),
    (r"\bexperts?\b", 2.0, "Shabda", "experts"),
    (r"\bscientists?\b", 2.1, "Shabda", "scientists"),
    (r"\bresearchers?\b", 2.0, "Shabda", "researchers"),
    (r"\bmedical guidelines?\b", 2.5, "Shabda", "medical guidelines"),
    (r"\bevidence suggests\b", 2.2, "Shabda", "evidence suggests"),
    (r"\bstud(?:y|ies)\b", 1.6, "Shabda", "study/studies"),
    (r"\breport(?:ed|s)?\b", 1.5, "Shabda", "report/reported"),
    (r"\bthe (?:handbook|manual|court|statute)\b", 2.0, "Shabda", "authoritative document"),
    (r"\bprofessor\b|\bteacher\b", 1.4, "Shabda", "professor/teacher"),
    # --- Upamana (analogy) — phrase-level only; no bare ``like`` ---
    (r"\bsimilar to\b", 2.4, "Upamana", "similar to"),
    (r"\bjust as\b", 2.3, "Upamana", "just as"),
    (r"\bacts like\b", 2.5, "Upamana", "acts like"),
    (r"\bworks like\b", 2.5, "Upamana", "works like"),
    (r"\bresembles\b", 2.4, "Upamana", "resembles"),
    (r"\banalogous to\b", 2.6, "Upamana", "analogous to"),
    (r"\bcompared with\b|\bcompared to\b", 2.2, "Upamana", "compared with/to"),
    (r"\bcomparison\b", 1.8, "Upamana", "comparison"),
    (r"\bmetaphor\b", 2.3, "Upamana", "metaphor"),
    (r"\bin the same way\b", 2.4, "Upamana", "in the same way"),
    (r"\bas if\b", 2.2, "Upamana", "as if"),
    (r"\bis like\b", 2.4, "Upamana", "is like"),
    (r"\blike a\b|\blike an\b", 2.0, "Upamana", "like a/an"),
    # --- Anumana (inference) ---
    (r"\bbecause\b", 2.0, "Anumana", "because"),
    (r"\btherefore\b", 2.3, "Anumana", "therefore"),
    (r"\bthus\b", 2.0, "Anumana", "thus"),
    (r"\bhence\b", 2.1, "Anumana", "hence"),
    (r"\bconsequently\b", 2.2, "Anumana", "consequently"),
    (r"\bimplies\b", 2.1, "Anumana", "implies"),
    (r"\bsuggests\b", 1.6, "Anumana", "suggests"),
    (r"\blikely\b", 1.5, "Anumana", "likely"),
    (r"\bcauses\b", 2.0, "Anumana", "causes"),
    (r"\bif\b.{0,40}\bthen\b", 2.4, "Anumana", "if...then"),
    (r"\bdue to\b", 2.0, "Anumana", "due to"),
    (r"\bresults in\b", 2.1, "Anumana", "results in"),
    (r"\bleads to\b", 2.1, "Anumana", "leads to"),
    (r"\btherefore it follows\b", 2.5, "Anumana", "therefore it follows"),
    (r"\bsince\b", 1.7, "Anumana", "since"),
    (r"\bwhich suggests\b", 1.9, "Anumana", "which suggests"),
    (r"\bmost likely\b", 1.8, "Anumana", "most likely"),
]

# Patterns that suppress weak Upamana readings (e.g. ``I like football``).
_UPAMANA_FALSE_POSITIVE = re.compile(
    r"\b(?:i|we|they|he|she)\s+like\b|\bwould like\b",
    re.IGNORECASE,
)

# Academic ``studies'' (``he studies biology'') is not Shabda testimony.
_STUDY_INFORMAL = re.compile(
    r"\b(?:he|she|they|i|we)\s+stud(?:y|ies)\b|\bstudies at\b|\bstudies for\b",
    re.IGNORECASE,
)
_RESEARCH_CONTEXT = re.compile(
    r"\b(?:research|journal|published|found|shows|according to|reported)\b",
    re.IGNORECASE,
)

_COMPILED_RULES: list[tuple[re.Pattern[str], WeightedRule]] = [
    (re.compile(spec[0], re.IGNORECASE), WeightedRule(*spec)) for spec in _RULE_SPECS
]

# Longer / more specific ``according to`` patterns must win over the general one.
_SORTED_RULES = sorted(_COMPILED_RULES, key=lambda x: len(x[1].pattern), reverse=True)


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.clip(v.astype(np.float64), 1e-12, None)
    return v / v.sum()


def _apply_multi_cue_bonus(raw_scores: dict[str, float], matched: dict[str, list[dict[str, Any]]]) -> None:
    for pramana, cues in matched.items():
        n = len(cues)
        if n <= 1:
            continue
        bonus = min(MULTI_CUE_BONUS_CAP, MULTI_CUE_BONUS * (n - 1))
        raw_scores[pramana] *= 1.0 + bonus


def score_weighted_rules(text: str) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    """
    Return per-pramāṇa weighted scores and matched cue metadata.

    More specific patterns are evaluated first; general ``according to`` is
    skipped when a tiered variant already matched the same span.
    """
    raw: dict[str, float] = {p: 0.0 for p in DEFAULT_CLASS_ORDER}
    matched: dict[str, list[dict[str, Any]]] = {p: [] for p in DEFAULT_CLASS_ORDER}
    according_to_claimed = False

    for regex, rule in _SORTED_RULES:
        if rule.cue.startswith("according to") and according_to_claimed:
            if rule.cue == "according to (general)":
                continue

        m = regex.search(text)
        if not m:
            continue

        if rule.pramana == "Upamana" and _UPAMANA_FALSE_POSITIVE.search(text):
            if rule.cue in {"like a/an"}:
                continue

        if rule.cue == "study/studies" and _STUDY_INFORMAL.search(text):
            if not _RESEARCH_CONTEXT.search(text):
                continue

        raw[rule.pramana] += rule.weight
        matched[rule.pramana].append(
            {"cue": rule.cue, "weight": rule.weight, "span": m.group(0)}
        )

        if rule.cue.startswith("according to") and rule.cue != "according to (general)":
            according_to_claimed = True
        elif rule.cue == "according to (general)":
            according_to_claimed = True

    _apply_multi_cue_bonus(raw, matched)
    return raw, matched


def detect_pattern_signals(text: str) -> dict[str, Any]:
    """
    Interpretability signals: hit counts (legacy) plus weighted scores and cues.
    """
    raw, matched = score_weighted_rules(text)

    return {
        "authority_hits": len(matched["Shabda"]),
        "analogy_hits": len(matched["Upamana"]),
        "inference_hits": len(matched["Anumana"]),
        "observation_hits": len(matched["Pratyaksha"]),
        "authority_score": round(raw["Shabda"], 3),
        "analogy_score": round(raw["Upamana"], 3),
        "inference_score": round(raw["Anumana"], 3),
        "observation_score": round(raw["Pratyaksha"], 3),
        "matched_cues": matched,
    }


def rule_distribution(
    text: str, class_order: tuple[str, ...] = DEFAULT_CLASS_ORDER
) -> np.ndarray:
    """
    Map weighted cue scores to a probability vector over ``class_order``.

    Uses a soft floor (every class gets baseline mass) plus squared raw scores
    so multiple strong cues sharpen the distribution.
    """
    raw, matched = score_weighted_rules(text)
    scores = np.ones(len(class_order), dtype=np.float64)

    for name in class_order:
        scores[list(class_order).index(name)] += raw.get(name, 0.0) ** 1.35

    active_families = sum(1 for name in class_order if matched.get(name))
    if active_families >= 3:
        scores *= 0.88

    return _norm_vec(scores)


def _label_from_probs(probs: np.ndarray, class_order: tuple[str, ...]) -> str:
    return class_order[int(np.argmax(probs))]


def _has_symbolic_evidence(matched: dict[str, list[Any]]) -> bool:
    return any(len(cues) > 0 for cues in matched.values())


def _adjust_confidence_for_agreement(
    fused_max_prob: float,
    ml_label: str,
    rule_label: str | None,
    *,
    symbolic_active: bool,
) -> tuple[float, dict[str, Any]]:
    base_pct = float(fused_max_prob * 100.0)

    if not symbolic_active or rule_label is None:
        return base_pct, {
            "agreement": None,
            "ml_label": ml_label,
            "rule_label": rule_label,
            "base_fused_confidence_pct": round(base_pct, 4),
            "confidence_adjustment": "neutral",
            "symbolic_active": False,
        }

    agree = ml_label == rule_label
    if agree:
        adjusted = min(99.5, base_pct * AGREEMENT_CONF_BOOST + AGREEMENT_CONF_ADD)
        adjustment = "boost"
    else:
        adjusted = max(5.0, base_pct * DISAGREEMENT_CONF_MULT)
        adjustment = "reduce"

    return adjusted, {
        "agreement": agree,
        "ml_label": ml_label,
        "rule_label": rule_label,
        "base_fused_confidence_pct": round(base_pct, 4),
        "confidence_adjustment": adjustment,
        "symbolic_active": True,
    }


def hybrid_fuse(
    ml_probs: np.ndarray,
    text: str,
    *,
    class_order: tuple[str, ...] | list[str] | None = None,
    ml_weight: float | None = None,
    rule_weight: float | None = None,
    alpha: float | None = None,
    routing_mode: str = "fixed",
    routing_reason: str | None = None,
) -> dict[str, Any]:
    """
    Fuse ML softmax with weighted rule-based distribution.

    Parameters
    ----------
    ml_probs
        1D array of class probabilities aligned with ``class_order``.
    text
        Original user text for cue extraction.
    class_order
        Names aligned with ``ml_probs`` indices. If ``None``, use default tuple
        (callers should pass ``label_encoder.classes_`` from disk).
    alpha
        ML fusion weight in [0, 1]. When set, ``ml_weight=alpha`` and
        ``rule_weight=1-alpha``. Overrides ``ml_weight`` / ``rule_weight``.
    ml_weight, rule_weight
        Legacy weights. Defaults follow ablation-optimal alpha=0.2
        (ML 0.2 / rules 0.8). Ignored when ``alpha`` is provided.
    routing_mode
        ``fixed`` or ``adaptive`` — recorded for explainability.
    routing_reason
        Human-readable routing explanation (optional).
    """
    order = tuple(class_order) if class_order is not None else DEFAULT_CLASS_ORDER

    if alpha is not None:
        ml_w = float(alpha)
        rule_w = 1.0 - ml_w
    else:
        # Research default: alpha=0.2 (see reports/alpha_investigation.md).
        ml_w = 0.2 if ml_weight is None else float(ml_weight)
        rule_w = 0.8 if rule_weight is None else float(rule_weight)

    p_ml = _norm_vec(np.asarray(ml_probs, dtype=np.float64).reshape(-1))
    p_rules = rule_distribution(text, class_order=order)
    fused = ml_w * p_ml + rule_w * p_rules
    fused = _norm_vec(fused)

    ml_label = _label_from_probs(p_ml, order)
    raw_scores, matched_cues = score_weighted_rules(text)
    symbolic_active = _has_symbolic_evidence(matched_cues)
    rule_label = _label_from_probs(p_rules, order) if symbolic_active else None
    best_i = int(np.argmax(fused))
    final_label = order[best_i]

    adjusted_confidence, agreement_meta = _adjust_confidence_for_agreement(
        fused[best_i],
        ml_label,
        rule_label,
        symbolic_active=symbolic_active,
    )

    signals = detect_pattern_signals(text)

    return {
        "class_order": list(order),
        "ml_probs": p_ml.tolist(),
        "rule_probs": p_rules.tolist(),
        "fused_probs": fused.tolist(),
        "ml_label": ml_label,
        "rule_label": rule_label,
        "final_label": final_label,
        "adjusted_confidence": adjusted_confidence,
        "pattern_signals": signals,
        "rule_scores": {k: round(v, 4) for k, v in raw_scores.items()},
        "matched_cues": matched_cues,
        "agreement": agreement_meta,
        "weights": {"ml": ml_w, "rules": rule_w},
        "alpha": round(ml_w, 4),
        "routing_mode": routing_mode,
        "routing_reason": routing_reason or f"Fusion alpha={ml_w:.2f} ({routing_mode}).",
    }
