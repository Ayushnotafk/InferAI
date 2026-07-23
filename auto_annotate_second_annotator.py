"""
Simulate a second annotator via rule-based pramāṇa labeling.

Loads ``dataset/annotations/annotator2.csv``, adds ``pramana_label`` using
regex heuristics, and writes ``dataset/annotations/annotator2_completed.csv``.

Usage (from project root):

    python auto_annotate_second_annotator.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "dataset" / "annotations" / "annotator2.csv"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "annotations" / "annotator2_completed.csv"

LABELS = ("Pratyaksha", "Anumana", "Upamana", "Shabda")

# (compiled_pattern, weight) — higher weight = stronger cue for that family
OBSERVATION_CUES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bi (saw|heard|observed|noticed|smelled|felt|watched)\b", re.I), 2.0),
    (re.compile(r"\bwe (saw|heard|observed|measured|counted|recorded)\b", re.I), 2.0),
    (re.compile(r"\bthe (sensor|display|log|image|recording|gauge|thermometer|microscope)\b", re.I), 1.5),
    (re.compile(r"\b(read|shows?|showed|reported|recorded|logged|timestamped)\b", re.I), 1.2),
    (re.compile(r"\b(telemetry|field notes|printout|ultrasound|oximeter|spectroscope)\b", re.I), 1.8),
    (re.compile(r"\b(visible|audible|flickering|flatlined|clipping|measured|reading)\b", re.I), 1.3),
    (re.compile(r"\b\d+(\.\d+)?\s*(%|°c|db|nm|meters?|minutes?|seconds?|hz|gpa)\b", re.I), 1.4),
    (re.compile(r"\bon (the|my) (screen|monitor|dashboard|ward|floor|balcony)\b", re.I), 1.2),
]

INFERENCE_CUES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\btherefore\b", re.I), 2.5),
    (re.compile(r"\bhence\b|\bthus\b|\bconsequently\b", re.I), 2.0),
    (re.compile(r"\bwhich (implies|suggests|means)\b", re.I), 2.2),
    (re.compile(r"\bimplies\b|\bimplying\b", re.I), 2.0),
    (re.compile(r"\b(likely|probably|plausible|suggests? that)\b", re.I), 1.8),
    (re.compile(r"\bbecause\b|\bsince\b|\bdue to\b", re.I), 1.6),
    (re.compile(r"\bso (the|it|we|they|this)\b", re.I), 1.5),
    (re.compile(r"\bwhich (makes|made)\b", re.I), 1.4),
    (re.compile(r"\bconsistent with\b|\bsupports? the (view|hypothesis)\b", re.I), 1.5),
    (re.compile(r"\bif .+ then\b", re.I), 1.3),
    (re.compile(r"\bworking hypothesis\b", re.I), 1.6),
]

ANALOGY_CUES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\bis like\b|\bare like\b|\bwas like\b", re.I), 2.5),
    (re.compile(r"\bworks like\b|\bacts like\b|\bfunctions like\b|\bbehave[sd]? like\b", re.I), 2.3),
    (re.compile(r"\bsimilar to\b|\banalogous to\b", re.I), 2.5),
    (re.compile(r"\bjust as\b|\bcompared to\b|\bresembles\b", re.I), 2.0),
    (re.compile(r"\blike a\b|\blike an\b", re.I), 2.0),
    (re.compile(r"\bmetaphor\b|\blikened\b|\bcompared .+ to\b", re.I), 2.2),
    (re.compile(r"\bis (similar|analogous)\b", re.I), 2.0),
    (re.compile(r"\b(described|portrayed|explained|treated) .{0,40} as a\b", re.I), 2.4),
    (re.compile(r"\b(is|are|was|were) (like|similar to|analogous to)\b", re.I), 2.3),
]

AUTHORITY_CUES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\baccording to\b", re.I), 2.8),
    (re.compile(r"\b(experts?|researchers?|scientists?|historians?|analysts?)\b", re.I), 2.0),
    (re.compile(r"\b(stud(?:y|ies)|report|reports|guideline|guidelines|manual|handbook|bulletin)\b", re.I), 2.0),
    (re.compile(r"\b(who|cdc|fda|nasa|ipcc|cochrane|reuters)\b", re.I), 2.5),
    (re.compile(r"\bthe (court|statute|syllabus|registrar|committee|advisory)\b", re.I), 2.0),
    (re.compile(r"\b(professor|teacher|coach|minister|regulator)\b", re.I), 1.6),
    (re.compile(r"\b(encyclopedia|textbook|journal|referee report|annual report|license)\b", re.I), 1.8),
    (re.compile(r"\b(stated|announced|concludes?|recommends?|warns?|mandates?)\b", re.I), 1.2),
    (re.compile(r"\bprescribing information\b|\binfection-prevention checklist\b", re.I), 2.2),
]

FAMILY_MAP = {
    "Pratyaksha": OBSERVATION_CUES,
    "Anumana": INFERENCE_CUES,
    "Upamana": ANALOGY_CUES,
    "Shabda": AUTHORITY_CUES,
}

# Tie-break priority when scores are equal (deterministic)
_TIE_PRIORITY = ("Shabda", "Anumana", "Upamana", "Pratyaksha")


def _score_family(text: str, cues: list[tuple[re.Pattern[str], float]]) -> float:
    total = 0.0
    for pattern, weight in cues:
        if pattern.search(text):
            total += weight
    return total


def classify_pramana(text: str) -> str:
    """
    Assign one pramāṇa label by summing weighted regex hits per family.

    The category with the highest score wins; ties use ``_TIE_PRIORITY``.
    """
    if not str(text).strip():
        return "Anumana"

    scores = {label: _score_family(text, FAMILY_MAP[label]) for label in LABELS}

    best = max(scores.values())
    if best == 0.0:
        # No strong cue: prefer observation if sensory/measurement verbs appear
        t = text.lower()
        if re.search(
            r"\b(saw|heard|felt|smelled|noticed|read|showed|shows|display|log|gauge|sensor|measured|recorded)\b",
            t,
        ):
            return "Pratyaksha"
        return "Anumana"

    winners = [lab for lab, s in scores.items() if s == best]
    if len(winners) == 1:
        return winners[0]

    for lab in _TIE_PRIORITY:
        if lab in winners:
            return lab
    return winners[0]


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input not found: {INPUT_CSV}\nRun: python prepare_annotation_sheet.py"
        )

    df = pd.read_csv(INPUT_CSV)
    if "text" not in df.columns:
        raise ValueError("annotator2.csv must contain a 'text' column")

    n = len(df)
    df = df.copy()
    df["pramana_label"] = df["text"].astype(str).map(classify_pramana)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Total rows processed: {n}")
    print("\nClass distribution (simulated annotator 2):")
    dist = df["pramana_label"].value_counts().reindex(LABELS, fill_value=0)
    for label in LABELS:
        count = int(dist[label])
        pct = 100.0 * count / n if n else 0.0
        print(f"  {label:12s} {count:4d}  ({pct:5.1f}%)")
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
