"""
Heuristic fallacy / hetvābhāsa-inspired checks for InferAI.

These are lightweight structural heuristics, not full Nyāya formal reasoning.
Inspired by common debate faults: contradiction, missing premise, weak inference,
unsupported analogy, and unsupported authority.
"""

from __future__ import annotations

import re
from typing import Any

# Contradiction cue pairs (simplified lexical heuristics).
_WET = re.compile(r"\b(?:rain(?:ing)?|wet|damp|storm)\b", re.I)
_DRY = re.compile(r"\b(?:dry|arid|no rain|not raining)\b", re.I)

_INFERENCE = re.compile(
    r"\b(?:therefore|thus|hence|consequently|implies|so)\b", re.I
)
_ANALOGY = re.compile(
    r"\b(?:similar to|like a|like an|analogous to|just as|is like)\b", re.I
)
_AUTHORITY = re.compile(r"\baccording to\b", re.I)
_NAMED_SOURCE = re.compile(
    r"\b(?:who|cdc|fda|nasa|unesco|court|government|study|research|experts?|scientists?|journal)\b",
    re.I,
)

# Very short premise heuristic.
_MIN_PREMISE_CHARS = 12


def _normalize(s: str) -> str:
    return " ".join(str(s or "").split()).strip()


def _weak_similarity(claim: str, premises: str) -> bool:
    """Lexical overlap proxy when embeddings are unavailable in this module."""
    c_tokens = set(_normalize(claim).lower().split())
    p_tokens = set(_normalize(premises).lower().split())
    if not c_tokens or not p_tokens:
        return True
    overlap = len(c_tokens & p_tokens) / max(len(c_tokens | p_tokens), 1)
    return overlap < 0.05


def detect_fallacies(
    text: str,
    claim: str,
    premises: str,
    *,
    reasoning_indicators: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run heuristic fallacy checks on extracted argument structure.

    Returns a dict suitable for API extension with keys:
    ``fallacy_detected``, ``fallacy_type``, ``fallacy_explanation``.
    """
    full = _normalize(text)
    claim_n = _normalize(claim)
    prem_n = _normalize(premises)
    indicators = reasoning_indicators or []

    # 1. Contradiction between premise and claim.
    if prem_n and claim_n:
        if (_WET.search(prem_n) and _DRY.search(claim_n)) or (
            _DRY.search(prem_n) and _WET.search(claim_n)
        ):
            return {
                "fallacy_detected": True,
                "fallacy_label": "contradiction",
                "confidence": 0.9,
                "evidence": "lexical_polarity_wet_dry",
                "reason": "Premise and claim show opposing polarity (wet vs dry).",
                "rules_triggered": ["contradiction_polarity"],
            }

    # 2. Missing / unstated premise.
    has_inference = bool(_INFERENCE.search(full)) or any(
        i.lower() in {"because", "therefore", "thus", "hence", "since"} for i in indicators
    )
    if has_inference and (not prem_n or len(prem_n) < _MIN_PREMISE_CHARS):
        return {
            "fallacy_detected": True,
            "fallacy_label": "unstated_premise",
            "confidence": 0.7,
            "evidence": "short_or_missing_premise",
            "reason": "Inference connector present but premises are missing or too short.",
            "rules_triggered": ["unstated_premise_length"],
        }

    # 3. Weak inference.
    if has_inference and prem_n and claim_n and _weak_similarity(claim_n, prem_n):
        return {
            "fallacy_detected": True,
            "fallacy_label": "weak_inference",
            "confidence": 0.65,
            "evidence": "low_lexical_overlap",
            "reason": "Low lexical overlap between premise and claim despite inference cue.",
            "rules_triggered": ["weak_inference_overlap"],
        }

    # 4. Unsupported analogy.
    if _ANALOGY.search(full) and _weak_similarity(claim_n, prem_n or full):
        return {
            "fallacy_detected": True,
            "fallacy_label": "weak_analogy",
            "confidence": 0.6,
            "evidence": "analogy_keyword_no_overlap",
            "reason": "Analogy language used but semantic overlap is weak.",
            "rules_triggered": ["weak_analogy_keyword_overlap"],
        }

    # 5. Unsupported authority.
    if _AUTHORITY.search(full) and not _NAMED_SOURCE.search(full):
        return {
            "fallacy_detected": True,
            "fallacy_label": "weak_authority",
            "confidence": 0.6,
            "evidence": "authority_phrase_no_named_source",
            "reason": "Authority phrasing present without identifiable/credible source.",
            "rules_triggered": ["weak_authority_phrase"],
        }

    return {
        "fallacy_detected": False,
        "fallacy_label": "none",
        "confidence": 0.0,
        "evidence": None,
        "reason": None,
        "rules_triggered": [],
    }


