"""HTML helpers for the InferAI dashboard UI."""

from __future__ import annotations

import html
from typing import Any, Iterable

# Lucide-style outline icons (24×24 viewBox, stroke currentColor)
_ICONS: dict[str, str] = {
    "sparkles": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/>'
        '<path d="M19 15l.75 2.25L22 18l-2.25.75L19 21l-.75-2.25L16 18l2.25-.75L19 15z"/>'
        "</svg>"
    ),
    "gauge": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M12 15l3.5-3.5"/>'
        '<path d="M4.5 16.5a8.5 8.5 0 1115 0"/>'
        '<path d="M12 20v-1"/>'
        "</svg>"
    ),
    "activity": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
        "</svg>"
    ),
    "message": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>'
        "</svg>"
    ),
    "layers": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        '<polyline points="2 12 12 17 22 12"/>'
        "</svg>"
    ),
    "file-text": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" y1="13" x2="8" y2="13"/>'
        '<line x1="16" y1="17" x2="8" y2="17"/>'
        '<line x1="10" y1="9" x2="8" y2="9"/>'
        "</svg>"
    ),
    "tag": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"/>'
        '<line x1="7" y1="7" x2="7.01" y2="7"/>'
        "</svg>"
    ),
    "highlighter": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M9 11l-6 6v3h9l3-3"/>'
        '<path d="M22 12l-4.6 4.6a2 2 0 01-2.8 0l-5.2-5.2a2 2 0 010-2.8L14 4"/>'
        "</svg>"
    ),
    "bar-chart": (
        '<svg class="ia-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<line x1="12" y1="20" x2="12" y2="10"/>'
        '<line x1="18" y1="20" x2="18" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="16"/>'
        "</svg>"
    ),
}


def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


<<<<<<< HEAD
def icon(name: str) -> str:
    return _ICONS.get(name, "")
=======
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
>>>>>>> 7fa6014e8b1ce70575677496d1136adac2916b14


def normalize_confidence(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, v))


def header_block() -> str:
    return """
<div class="ia-header">
  <h1 class="ia-brand">InferAI</h1>
  <p class="ia-tagline">Explainable Neuro-Symbolic Reasoning</p>
  <p class="ia-sub">Nyāya-inspired hybrid AI for transparent argument classification.</p>
</div>
"""


def section_title(text: str, icon_name: str = "") -> str:
    ico = f'<span class="ia-section-icon">{icon(icon_name)}</span>' if icon_name else ""
    return f'<p class="ia-section">{ico}<span>{esc(text)}</span></p>'


def _label_row(title: str, icon_name: str = "") -> str:
    ico = f'<span class="ia-label-icon">{icon(icon_name)}</span>' if icon_name else ""
    return f'<p class="ia-card-label">{ico}<span>{esc(title)}</span></p>'


def strength_badge(strength: str) -> str:
    raw = (strength or "—").strip()
    key = raw.lower()
    if key == "strong":
        cls = "ia-pill strong"
    elif key == "moderate":
        cls = "ia-pill moderate"
    elif key == "weak":
        cls = "ia-pill weak"
    else:
        cls = "ia-pill"
    display = raw.title() if raw and raw != "—" else "—"
    return f'<span class="{cls}">{esc(display)}</span>'


def prediction_card(prediction: str, reference: str) -> str:
    return f"""
<div class="ia-card ia-card--focus ia-card--prediction">
  {_label_row("Prediction", "sparkles")}
  <p class="ia-card-value accent ia-card-value--xl">{esc(prediction)}</p>
  <p class="ia-card-sub">Reference: {esc(reference)}</p>
</div>
"""


def confidence_card(confidence_pct: float) -> str:
    c = max(0.0, min(100.0, confidence_pct))
    return f"""
<div class="ia-card ia-card--focus">
  {_label_row("Confidence", "gauge")}
  <p class="ia-card-value ia-card-value--xl">{c:.0f}%</p>
  <p class="ia-card-sub">Combined confidence</p>
  <div class="ia-bar ia-bar--in-card"><span style="width:{c:.1f}%"></span></div>
</div>
"""


def strength_card(strength: str) -> str:
    return f"""
<div class="ia-card ia-card--focus">
  {_label_row("Reasoning Strength", "activity")}
  <div class="ia-strength-wrap">{strength_badge(strength)}</div>
  <p class="ia-card-sub">Composite score</p>
</div>
"""


def metrics_row(
    prediction: str,
    reference: str,
    confidence_pct: float,
    strength: str,
    adaptive_alpha: float | None = None,
    routing_reason: str = "",
) -> str:
    alpha_card = ""
    if adaptive_alpha is not None:
        alpha_card = f"""
<div class="ia-card ia-card--focus">
  {_label_row("Adaptive Alpha", "gauge")}
  <p class="ia-card-value ia-card-value--xl">{adaptive_alpha:.2f}</p>
  <p class="ia-card-sub">ML fusion weight</p>
</div>
"""
    routing_block = ""
    if routing_reason:
        routing_block = f"""
<div class="ia-card ia-card--routing">
  {_label_row("Routing", "activity")}
  <p class="ia-card-body">{esc(routing_reason)}</p>
</div>
"""
    return f"""
<div class="ia-metric-grid ia-metric-grid--4">
  {prediction_card(prediction, reference)}
  {confidence_card(confidence_pct)}
  {strength_card(strength)}
  {alpha_card}
</div>
{routing_block}
"""


def fallacy_warning_card(fallacy_type: str, explanation: str) -> str:
    return f"""
<div class="ia-fallacy-warning">
  <p class="ia-fallacy-title">Fallacy detected: {esc(fallacy_type)}</p>
  <p class="ia-fallacy-body">{esc(explanation)}</p>
</div>
"""


def fallacy_ok_card() -> str:
    return """
<div class="ia-fallacy-ok">
  <p class="ia-card-label">Fallacy check</p>
  <p class="ia-card-body">None detected by heuristic screening.</p>
</div>
"""


def confidence_bars(combined: float, model: float) -> str:
    c = max(0.0, min(100.0, combined))
    m = max(0.0, min(100.0, model))
    return f"""
<div class="ia-card ia-card--breakdown">
  {_label_row("Confidence breakdown", "bar-chart")}
  <div class="ia-conf-row">
    <div class="ia-conf-meta">
      <span class="ia-conf-label">Combined</span>
      <span class="ia-conf-pct">{c:.0f}%</span>
    </div>
    <div class="ia-bar ia-bar--lg"><span style="width:{c:.1f}%"></span></div>
  </div>
  <div class="ia-conf-row">
    <div class="ia-conf-meta">
      <span class="ia-conf-label">Model</span>
      <span class="ia-conf-pct">{m:.0f}%</span>
    </div>
    <div class="ia-bar ia-bar--lg secondary"><span style="width:{m:.1f}%"></span></div>
  </div>
</div>
"""


def content_card(title: str, body: str, large: bool = False, icon_name: str = "") -> str:
    body_cls = "ia-card-body lg" if large else "ia-card-body"
    card_cls = "ia-card ia-card--explain" if large else "ia-card"
    return f"""
<div class="{card_cls}">
  {_label_row(title, icon_name)}
  <div class="{body_cls}">{esc(body)}</div>
</div>
"""


def tags_block(items: Iterable[str]) -> str:
    chips = "".join(
        f'<span class="ia-tag">{esc(item)}</span>' for item in items if str(item).strip()
    )
    if not chips:
        return ""
    return f"""
<div class="ia-block">
  {section_title("Reasoning signals", "tag")}
  <div class="ia-tags">{chips}</div>
</div>
"""


def highlight_block(html_fragment: str) -> str:
    return f"""
{section_title("Highlighted cues", "highlighter")}
<div class="ia-highlight">{html_fragment}</div>
"""


def shap_note_block(note: str) -> str:
    if not note:
        return ""
    return f'<div class="ia-note">{esc(note)}</div>'


def examples_label() -> str:
    return '<p class="ia-examples-label">Try an example</p>'


def footer_block() -> str:
    return """
<div class="ia-footer">
  InferAI — Explainable Nyāya-inspired reasoning analysis
</div>
"""


def spacer(size: str = "md") -> str:
    return f'<div class="ia-spacer-{size}"></div>'
