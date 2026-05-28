"""
Structured extraction: claim, premises (support), and shallow discourse cues.

Design goals
------------
* Keep outputs safe for HTML rendering (escape at the boundary).
* Avoid splitting on substrings like ``like`` inside unrelated tokens.
* Support short multi-sentence spans by preferring explicit connectives.
"""

from __future__ import annotations

import html
import re
from typing import Any

# Conclusions often follow these cues (claim ≈ material after cue).
_CONCLUSION_CUES = sorted(
    [
        "it follows that",
        "we can conclude",
        "it can be concluded",
        "therefore",
        "hence",
        "thus",
        "consequently",
        "as a result",
        "which means",
        "leading to",
        "resulting in",
    ],
    key=len,
    reverse=True,
)

# Premises often precede these cues when they introduce support.
_PREMISE_FIRST_CUES = sorted(
    [
        "because",
        "since",
        "due to",
        "owing to",
        "given that",
        "as a result of",
        "on the grounds that",
    ],
    key=len,
    reverse=True,
)

# Analogy cues (word-bounded where needed).
_ANALOGY_CUES = [
    r"\bis\s+like\b",
    r"\bworks\s+like\b",
    r"\bacts\s+like\b",
    r"\bsimilar\s+to\b",
    r"\banalogous\s+to\b",
    r"\bjust\s+as\b",
    r"\bcompared\s+to\b",
    r"\bfunctions\s+like\b",
    r"\bresembles\b",
]

_AUTHORITY_CUES = [
    r"\baccording\s+to\b",
    r"\bexperts?\s+(say|believe|argue|claim)\b",
    r"\bresearchers?\b",
    r"\bscientists?\b",
    r"\bstud(?:y|ies)\s+show\b",
    r"\breports?\s+indicate\b",
    r"\bthe\s+(fda|cdc|who|court)\b",
    r"\bprofessor\b",
    r"\bmanual\b",
    r"\bhandbook\b",
]

_INFERENCE_CUES = [
    r"\btherefore\b",
    r"\bhence\b",
    r"\bthus\b",
    r"\bimplies\b",
    r"\bmost\s+likely\b",
    r"\bwhich\s+suggests\b",
]

_OBSERVATION_CUES = [
    r"\bi\s+(saw|heard|observed|noticed|smelled|felt)\b",
    r"\bwe\s+measured\b",
    r"\bthe\s+(display|sensor|log)\b",
    r"\btelemetry\b",
    r"\bfield\s+notes\b",
]

_HIGHLIGHT_PATTERNS: list[tuple[str, str]] = [
    (r"\baccording\s+to\b", "cue-authority"),
    (r"\btherefore\b", "cue-inference"),
    (r"\bhence\b", "cue-inference"),
    (r"\bbecause\b", "cue-inference"),
    (r"\bsince\b", "cue-inference"),
    (r"\bis\s+like\b", "cue-analogy"),
    (r"\bsimilar\s+to\b", "cue-analogy"),
    (r"\banalogous\s+to\b", "cue-analogy"),
    (r"\bworks\s+like\b", "cue-analogy"),
    (r"\bi\s+(saw|heard|observed)\b", "cue-observation"),
]


def extract_claim(text: str) -> str:
    """Return the best-effort claim span."""
    if not text or not text.strip():
        return ""
    lower = text.lower()
    for cue in _PREMISE_FIRST_CUES:
        if cue in lower:
            parts = re.split(re.escape(cue), text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                return parts[0].strip()
    for cue in _CONCLUSION_CUES:
        if cue in lower:
            parts = re.split(re.escape(cue), text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                return parts[-1].strip()
    return text.strip()


def extract_premises(text: str) -> str:
    """
    Return supporting material (premises / evidence) prior to a conclusion cue,
    or the complement heuristic when a premise-first cue is used.
    """
    if not text or not text.strip():
        return ""
    lower = text.lower()
    for cue in _CONCLUSION_CUES:
        if cue in lower:
            parts = re.split(re.escape(cue), text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                return parts[0].strip()
    for cue in _PREMISE_FIRST_CUES:
        if cue in lower:
            parts = re.split(re.escape(cue), text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                return parts[-1].strip()

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    cleaned = [s.strip() for s in sentences if s.strip()]
    if len(cleaned) > 1:
        return cleaned[0]
    return text.strip()


def extract_reasoning_indicators(text: str) -> list[str]:
    """Short human-readable tags for discourse phenomena."""
    if not text:
        return []
    t = text.lower()
    tags: list[str] = []
    if any(re.search(p, t) for p in _AUTHORITY_CUES):
        tags.append("authority_language")
    if any(re.search(p, t) for p in _ANALOGY_CUES):
        tags.append("analogy_language")
    if any(re.search(p, t) for p in _INFERENCE_CUES):
        tags.append("inferential_connectives")
    if any(re.search(p, t) for p in _OBSERVATION_CUES):
        tags.append("observation_language")
    if len(re.split(r"(?<=[.!?])\s+", t)) > 2:
        tags.append("multi_sentence")
    return tags


def highlight_phrases_html(text: str) -> str:
    """
    Return HTML-safe markup with <mark> spans for known cue phrases.

    Overlapping matches: non-overlapping greedy left-to-right insertion using spans.
    """
    if not text:
        return ""
    raw = text
    lower = raw.lower()
    spans: list[tuple[int, int, str]] = []
    for pattern, cls in _HIGHLIGHT_PATTERNS:
        for m in re.finditer(pattern, lower, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), cls))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    kept: list[tuple[int, int, str]] = []
    end_at = -1
    for s, e, cls in spans:
        if s >= end_at:
            kept.append((s, e, cls))
            end_at = e

    kept.sort(key=lambda x: x[0])
    out: list[str] = []
    cursor = 0
    for s, e, cls in kept:
        out.append(html.escape(raw[cursor:s]))
        chunk = html.escape(raw[s:e])
        out.append(f'<mark class="{cls}">{chunk}</mark>')
        cursor = e
    out.append(html.escape(raw[cursor:]))
    return "".join(out)


def extract_argument_structure(text: str) -> dict[str, Any]:
    """Single entry point used by the API layer."""
    claim = extract_claim(text)
    premises = extract_premises(text)
    return {
        "claim": claim,
        "premises": premises,
        "reasoning_indicators": extract_reasoning_indicators(text),
        "highlighted_html": highlight_phrases_html(text),
    }
