"""
Natural-language explanations with controlled variety.

Templates are intentionally short so they read well in API/Streamlit cards.
"""

from __future__ import annotations

import random
from typing import Optional

_PRAMANA = {
    "Pratyaksha": [
        "This reads as Pratyaksha because the passage foregrounds direct observation or measurement rather than testimony or analogy.",
        "We label this Pratyaksha when the support is basically what could be sensed or recorded, not an expert quote or a likeness argument.",
        "Pratyaksha fits here: the author is asking you to trust what could be checked on the scene, in the lab, or on the instrument panel.",
    ],
    "Anumana": [
        "This looks like Anumāna: the text links a sign or premise to a conclusion you would not get from observation alone.",
        "Anumāna is appropriate because the reasoning move is inferential—evidence is used to justify a claim that goes beyond raw appearance.",
        "The structure here is classic Anumāna: premises are assembled so that a further claim becomes plausible (even if not certain).",
    ],
    "Upamana": [
        "Upamāna shows up when understanding is carried by comparison—similar structure, similar dynamics, similar role in a system.",
        "We classify this as Upamāna because the rhetorical work is analogical: one domain is illuminated by mapping it onto another.",
        "Here, Upamāna is the dominant move: readers are invited to transfer intuition across a similarity, not to cite an authority.",
    ],
    "Shabda": [
        "Śabda-style reasoning leans on testimony: manuals, experts, agencies, or other sources treated as authoritative for the claim.",
        "This is best read as Śabda because the warrant is essentially that a credible person or document says so.",
        "The passage channels Śabda: the justification route is institutional or expert voice rather than personal observation alone.",
    ],
}

_STRENGTH = {
    "Strong": [
        "The reasoning strength looks strong here: the connectors and content give you enough material to audit the move.",
        "We rate strength as strong when the argument is relatively crisp—clear cues, limited hedging, and a substantive support span.",
        "Strong strength: the passage is not just confident in tone; it also gives you premises you can actually inspect.",
    ],
    "Moderate": [
        "Strength is moderate: the move is intelligible, but the support is thinner, more hedged, or a bit mixed stylistically.",
        "A moderate rating fits when you can see what the author is doing, yet you would want more detail before treating it as airtight.",
        "Moderate strength reflects a reasonable but not maximally tight bridge between support and conclusion.",
    ],
    "Weak": [
        "Strength is weak here: the text is vague, highly speculative, or missing premises that would let a reader verify the move.",
        "Weak strength usually means slogans, bare assertions, or heavy hedging without much substantive scaffolding.",
        "We mark weak when the rhetorical posture outruns what the words actually supply as checkable support.",
    ],
}


def generate_explanation(pramana_label: str, strength_label: Optional[str] = None) -> str:
    """
    Return a short, presentation-ready explanation.

    A template is sampled at random (uniform) within the chosen buckets so
    repeated calls do not all read identically.
    """
    p = pramana_label or ""
    pool = _PRAMANA.get(p)
    if not pool:
        return "No tailored explanation template is available for this label yet."

    body = random.choice(pool)
    if not strength_label:
        return body

    s = strength_label.strip().capitalize()
    # Normalize common variants
    if s not in _STRENGTH:
        skey = "Moderate"
    else:
        skey = s

    tail = random.choice(_STRENGTH[skey])
    return f"{body}\n\n{tail}"
