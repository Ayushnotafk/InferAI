"""
Test script for verifying random confidence (75-95) and word-count based reasoning strength logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.inference import run_analysis
from reasoning_strength.composite import composite_reasoning_strength

def test_confidence_range():
    print("Testing confidence range (75 to 95)...")
    for _ in range(50):
        res = run_analysis("This is a test sentence with more than eight words in total length.")
        conf = res["confidence"]
        adj_conf = res["adjusted_confidence"]
        assert 75.0 <= conf <= 95.0, f"Confidence {conf} out of range [75, 95]"
        assert 75.0 <= adj_conf <= 95.0, f"Adjusted confidence {adj_conf} out of range [75, 95]"
    print("Confidence range [75, 95] verified successfully over 50 samples!")

def test_reasoning_strength_logic():
    print("\nTesting reasoning strength word count logic...")
    
    # 1. Short text (< 8 words)
    short_text = "Rain is falling down." # 4 words
    for _ in range(20):
        res = run_analysis(short_text)
        assert res["reasoning_strength"] == "Weak", f"Expected 'Weak' for short text, got {res['reasoning_strength']}"
    print("Short text (< 8 words) correctly evaluated as 'Weak'!")

    # 2. Long text (>= 8 words)
    long_text = "The pressure gauge recorded a value of 101.3 kPa using the calibrated sensor in laboratory." # 15 words
    results = Counter()
    for _ in range(1000):
        res = run_analysis(long_text)
        results[res["reasoning_strength"]] += 1
    
    print(f"Long text (>= 8 words) distribution over 1000 samples: {dict(results)}")
    assert "Weak" not in results or results["Weak"] == 0, "Long text should not yield 'Weak'"
    assert results["Medium"] > 0 and results["High"] > 0, "Long text should yield both 'Medium' and 'High'"
    
    medium_pct = results["Medium"] / 10.0
    high_pct = results["High"] / 10.0
    print(f"  Medium: {medium_pct:.1f}% (target ~60%)")
    print(f"  High: {high_pct:.1f}% (target ~40%)")
    assert 50.0 <= medium_pct <= 70.0, f"Medium percentage {medium_pct}% expected around 60%"

if __name__ == "__main__":
    test_confidence_range()
    test_reasoning_strength_logic()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
