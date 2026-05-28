"""Reasoning strength utilities (legacy + composite)."""

from reasoning_strength.composite import (
    composite_reasoning_strength,
    legacy_strength_from_confidence,
)


def reasoning_strength(confidence: float) -> str:
    """Confidence-only strength (kept for backward compatibility)."""
    return legacy_strength_from_confidence(confidence)


__all__ = [
    "reasoning_strength",
    "composite_reasoning_strength",
    "legacy_strength_from_confidence",
]
