# InferAI Annotation Guidelines — Quick Reference for Annotators

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
