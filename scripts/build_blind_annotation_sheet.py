"""
Build an independent blind annotation sheet for a human expert annotator.

Reviewer fix: C-06 (CRITICAL)
The previous second annotator was rule-assisted (auto_annotate_second_annotator.py),
which creates circularity between the gold labels and the symbolic model.

This script produces a clean annotation sheet with:
- 200 examples sampled from AAEC + IBM reviewed rows (no synthetic)
- No pramana label shown (completely blind)
- No rule output shown (no rule assistance)
- Includes annotation guidelines summary for reference
- Each example has a stable row_id for later ID join

Usage (from project root):
    python scripts/build_blind_annotation_sheet.py

Output
------
dataset/annotations/blind_annotation_sheet_v1.csv
    Columns: row_id, source, text, annotator_label (blank), notes (blank)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "dataset" / "processed"
ANNOTATIONS = ROOT / "dataset" / "annotations"
ANNOTATIONS.mkdir(parents=True, exist_ok=True)

AAEC_WITH_GROUPS = PROCESSED / "aaec_reviewed_with_groups.csv"
IBM_WITH_GROUPS = PROCESSED / "ibm_reviewed_with_groups.csv"
OUTPUT_CSV = ANNOTATIONS / "blind_annotation_sheet_v1.csv"
GUIDELINES_SUMMARY = ANNOTATIONS / "annotation_guidelines_summary.md"

PRAMANA_LABELS = ["Pratyaksha", "Anumana", "Upamana", "Shabda"]
SAMPLE_PER_SOURCE = 100
TARGET_MINORITY = 30  # aim for at least 30 examples of minority classes


def main() -> None:
    aaec = pd.read_csv(AAEC_WITH_GROUPS)
    ibm = pd.read_csv(IBM_WITH_GROUPS)

    # We know the existing labels from manual review — use them to stratify
    # but DO NOT include them in the output sheet (blind annotation).
    aaec["_source"] = "aaec"
    ibm["_source"] = "ibm"
    combined = pd.concat([aaec, ibm], ignore_index=True)
    combined = combined.dropna(subset=["text", "pramana_label"])
    combined = combined[combined["pramana_label"].isin(PRAMANA_LABELS)]

    print(f"Source pool: {len(combined)} rows")
    print(combined["pramana_label"].value_counts().to_string())

    # Stratified sample: oversample minority classes
    parts = []

    # Minority classes — take all available
    for lbl in ["Pratyaksha", "Upamana", "Shabda"]:
        subset = combined[combined["pramana_label"] == lbl]
        n = min(len(subset), TARGET_MINORITY)
        parts.append(subset.sample(n=n, random_state=42))
        print(f"  {lbl}: sampled {n}/{len(subset)}")

    # Anumana — fill rest up to 200
    already = sum(len(p) for p in parts)
    remaining = 200 - already
    anumana = combined[combined["pramana_label"] == "Anumana"]
    parts.append(anumana.sample(n=min(remaining, len(anumana)), random_state=42))
    print(f"  Anumana: sampled {min(remaining, len(anumana))}")

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=99).reset_index(drop=True)
    print(f"\nFinal sample: {len(sample)} rows")

    # Assign stable row IDs (hash of text to be reproducible)
    sample["row_id"] = [
        "blind_" + hashlib.md5(str(t).encode()).hexdigest()[:10]
        for t in sample["text"]
    ]

    # Build the blind sheet — NO pramana label, NO rule output
    blind_sheet = pd.DataFrame({
        "row_id": sample["row_id"],
        "source": sample["_source"],
        "text": sample["text"].str.strip(),
        "annotator_label": "",       # to be filled by human annotator
        "secondary_label": "",       # optional: if span is ambiguous, note secondary
        "confidence_1_5": "",        # 1=very unsure, 5=very sure
        "annotation_notes": "",      # free text
    })

    blind_sheet.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBlind annotation sheet saved: {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Total examples: {len(blind_sheet)}")
    print("\nIMPORTANT: Do NOT share the gold labels or rule outputs with the annotator.")
    print("Annotator should use only the guidelines summary + their own judgment.")

    # Write brief guidelines summary for the annotator
    guidelines = """# InferAI Annotation Guidelines — Quick Reference for Annotators

## Task
Read each argumentative sentence and assign the DOMINANT epistemic mode used.
Choose exactly ONE label from the four options below.

## Labels

### Pratyaksha (Direct perception / observation / measurement)
The claim is grounded in direct sensory experience, instrument readings, 
experimental data, or first-hand observation.

**Key signals:** "I saw", "we measured", "the sensor detected", "the experiment showed",
"temperature was recorded", "observed under microscope", "field notes show"

**Example:** *"The water temperature at the sampling site was 18.3°C, as recorded by our probe."*

---

### Anumana (Inference / logical reasoning)
The claim is inferred from evidence or reasons. A conclusion is drawn from premises.

**Key signals:** "because", "therefore", "thus", "hence", "consequently", 
"implies", "if...then", "leads to", "since", "due to"

**Example:** *"Because crime rates fell after the policy was introduced, the policy must have been effective."*

---

### Upamana (Analogy / comparison)
The claim uses an analogy, comparison, or similarity to make a point.

**Key signals:** "similar to", "just as", "like a", "acts like", "is analogous to",
"resembles", "compared with", "in the same way as"

**Example:** *"Teaching children arithmetic is like building a house — you need a solid foundation before adding floors."*

---

### Shabda (Testimony / authority)
The claim is justified by citing a source, expert, institution, or text.

**Key signals:** "according to", "research shows", "experts say", "WHO reports",
"the study found", "published in", "government data shows", "scientists argue"

**Example:** *"According to the WHO, this vaccine has a 95% efficacy rate."*

---

## Annotation rules

1. **Choose the DOMINANT mode** — the primary epistemic basis of the sentence.
2. If a sentence contains multiple modes, pick the one that does most of the justificatory work.
3. If you are genuinely unsure between two labels, note both in `secondary_label`.
4. Use `confidence_1_5`: 5 = certain, 4 = fairly sure, 3 = uncertain, 2 = two labels seem equal, 1 = confused.
5. Do NOT look up the rules or model outputs — annotate from the text alone.

## Valid label values
`Pratyaksha`, `Anumana`, `Upamana`, `Shabda`
"""

    GUIDELINES_SUMMARY.write_text(guidelines, encoding="utf-8")
    print(f"Guidelines summary: {GUIDELINES_SUMMARY.relative_to(ROOT)}")

    # Save metadata about what was sampled (with gold labels — KEEP PRIVATE)
    meta = sample[["row_id", "text", "pramana_label", "_source", "essay_id", "topic_group"]].copy()
    meta_path = ANNOTATIONS / "blind_annotation_sheet_v1_GOLD_LABELS_PRIVATE.csv"
    meta.to_csv(meta_path, index=False)
    print(f"\n⚠️  Gold labels (private): {meta_path.relative_to(ROOT)}")
    print("   Keep this file private — DO NOT share with annotators!")

    print("\n✅ Blind annotation sheet ready.")
    print(f"   Share with annotators: {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"   Share guidelines: {GUIDELINES_SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
