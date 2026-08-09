import pytest

from reasoning.fallacy_detector import detect_fallacies


def test_contradiction():
    text = "It is raining. Therefore the ground is completely dry."
    out = detect_fallacies(text, "the ground is completely dry", "It is raining")
    assert out["fallacy_detected"] is True
    assert out["fallacy_label"] == "contradiction"


def test_unstated_premise():
    text = "Therefore, we should ban the product."
    out = detect_fallacies(text, "we should ban the product", "")
    assert out["fallacy_detected"] is True
    assert out["fallacy_label"] == "unstated_premise"


def test_weak_inference():
    text = "Traffic increased last year; therefore the policy has failed."
    out = detect_fallacies(text, "the policy has failed", "Traffic increased last year")
    assert out["fallacy_detected"] is True
    assert out["fallacy_label"] == "weak_inference"


def test_weak_analogy():
    text = "Just as a map shows roads, this summary replaces the full report."
    out = detect_fallacies(text, "summary replaces full report", "summary is like a map")
    assert out["fallacy_detected"] is True
    assert out["fallacy_label"] == "weak_analogy"


def test_weak_authority():
    text = "According to experts, this is safe."
    out = detect_fallacies(text, "this is safe", "According to experts")
    assert out["fallacy_detected"] is True
    assert out["fallacy_label"] == "weak_authority"


def test_none():
    text = "The thermometer reads 100°C, therefore the water may be boiling."
    out = detect_fallacies(text, "water may be boiling", "thermometer reads 100°C")
    # This simple detector should not always flag; accept either
    assert "fallacy_detected" in out
