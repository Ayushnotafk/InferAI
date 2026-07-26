"""Small HTML/markup helpers for the InferAI dashboard."""

from __future__ import annotations

import html
from typing import Any


def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def strength_badge_class(strength: str) -> str:
    s = (strength or "").strip().lower()
    if s == "strong":
        return "nyx-badge nyx-badge-strong"
    if s == "moderate":
        return "nyx-badge nyx-badge-moderate"
    return "nyx-badge nyx-badge-weak"


def strength_emoji(strength: str) -> str:
    s = (strength or "").strip().lower()
    if s == "strong":
        return "◆"
    if s == "moderate":
        return "◇"
    return "○"


def hero_block() -> str:
    return """
<div class="nyx-hero">
  <div class="nyx-hero-badge">⚖️ Research · Explainability · InferAI</div>
  <h1>InferAI</h1>
  <p>Explainable classification for short arguments — observation, inference, analogy, and testimony — with confidence, reasoning strength, and optional SHAP over embedding space.</p>
</div>
"""


def section_title(text: str) -> str:
    """Minimal section label (no gradient divider) for a cleaner product UI."""
    return f'<p class="nyx-section-title nyx-section-title--compact">{esc(text)}</p>'


def glass_card(title: str, body: str, icon: str = "▸") -> str:
    return f"""
<div class="nyx-glass">
  <div class="nyx-glass-head">{esc(icon)} {esc(title)}</div>
  <div class="nyx-glass-body">{esc(body)}</div>
</div>
"""


def prediction_spotlight(pramana: str, subtitle: str | None = None) -> str:
    sub = (
        f'<div class="nyx-prediction-sub">{esc(subtitle)}</div>'
        if subtitle
        else ""
    )
    return f"""
<div class="nyx-prediction">
  <div class="nyx-prediction-label">Pramāṇa</div>
  <div class="nyx-prediction-value">✦ {esc(pramana)}</div>
  {sub}
</div>
"""


def strength_badge_html(strength: str) -> str:
    cls = strength_badge_class(strength)
    em = strength_emoji(strength)
    return f'<div style="margin-top:0.5rem;"><span class="{cls}">{esc(em)} {esc(strength or "—")}</span></div>'


def shap_note_block(note: str) -> str:
    return f'<div class="nyx-shap-note">{esc(note)}</div>'


def footer_block() -> str:
    return """
<div class="nyx-footer">
  InferAI — Explainable Reasoning Analysis
</div>
"""


def normalize_confidence(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, v))
