"""
Test script to validate the expanded 400 symbolic rule base (100 per class).
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.hybrid_reasoning import (
    _RULE_SPECS,
    _COMPILED_RULES,
    score_weighted_rules,
)

def test_rule_specs_and_compilation():
    print(f"Total rules in _RULE_SPECS: {len(_RULE_SPECS)}")
    print(f"Total rules compiled in _COMPILED_RULES: {len(_COMPILED_RULES)}")
    
    assert len(_RULE_SPECS) == 400, f"Expected 400 rules, got {len(_RULE_SPECS)}"
    assert len(_COMPILED_RULES) == 400, f"Expected 400 compiled rules, got {len(_COMPILED_RULES)}"
    
    counts = Counter(pramana for _, _, pramana, _ in _RULE_SPECS)
    print("\nRule counts per Pramana class:")
    for pramana, count in sorted(counts.items()):
        print(f"  {pramana}: {count}")
        assert count == 100, f"Class {pramana} expected 100 rules, got {count}"
        
    print("\nAll 400 rules verified successfully!")

def test_sample_inputs():
    samples = [
        ("The pressure gauge recorded a value of 101.3 kPa using the calibrated sensor.", "Pratyaksha"),
        ("According to the WHO report published in 2024, vaccination rates increased.", "Shabda"),
        ("A database index works like a library card catalog because both organize data for quick access.", "Upamana"),
        ("The server CPU usage reached 99%, therefore response latency increased.", "Anumana"),
    ]
    
    print("\nTesting sample text inputs:")
    for text, expected in samples:
        scores, matched = score_weighted_rules(text)
        top_pramana = max(scores.items(), key=lambda x: x[1])[0]
        print(f"\nText: \"{text}\"")
        print(f"  Expected: {expected} | Top Rule Match: {top_pramana}")
        print(f"  Scores: {scores}")
        print(f"  Matched cues: {matched}")
        assert top_pramana == expected, f"Expected top match {expected}, got {top_pramana}"

if __name__ == "__main__":
    test_rule_specs_and_compilation()
    test_sample_inputs()
