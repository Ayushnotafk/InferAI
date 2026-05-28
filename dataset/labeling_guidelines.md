# InferAI — Labeling Guidelines

This document defines how human reviewers assign **pramāṇa** labels and **reasoning strength** labels to short English arguments. It is intended for course teams, annotators, and anyone extending `master_dataset.csv` or `test_set.csv`.

---

## 1. Pramāṇa labels (four-way)

We map classical Nyāya *pramāṇas* to **English argument patterns** for supervised learning. Labels must reflect the **dominant** reasoning mode in the span (usually one sentence).

### 1.1 Pratyaksha (observation / perception)

**Definition:** The claim is supported primarily by **direct sensory evidence**, measurement, or an immediately reportable state of affairs (what was seen, heard, felt, measured, or recorded).

**Typical cues:** first-person observation, instrument readouts, “the display shows…”, “we measured…”, “on inspection…”.

**Examples**

1. “The thermometer read 39.2 °C on three consecutive checks.”
2. “Through the microscope, the cell walls looked ruptured in the treated sample.”
3. “On the factory floor, the motor was audibly misfiring every few seconds.”

**Edge cases**

- **Not** Pratyaksha if the sentence mainly **infers** a hidden cause from a sign (“smoke, therefore fire”) → **Anumana**.
- A report that **cites** an authority (“WHO states…”) → **Shabda**, even if factual.

### 1.2 Anumana (inference)

**Definition:** The claim moves from **sign / premise** to a **conclusion** that is not directly observed (hypothesis, diagnosis, troubleshooting, legal inference, scientific inference).

**Typical cues:** “therefore”, “because”, “since”, “implies”, “suggests that”, “likely”, “must be”.

**Examples**

1. “Packet loss spikes at peak hours, so congestion is more plausible than hardware failure.”
2. “The contract is ambiguous; courts often apply contra proferentem.”
3. “She missed two exams and looks exhausted; she is probably unwell.”

**Edge cases**

- **Weak** inference with heavy hedging may still be Anumana, but strength should drop (see §2).
- If the text is **only** analogy (“X is like Y”) with no authority → **Upamana**.

### 1.3 Upamana (analogy / comparison)

**Definition:** Understanding is conveyed by **comparison** or structural similarity (“like”, “similar to”, “analogous to”), not by citing an expert and not by a full inferential syllogism.

**Examples**

1. “Gradient descent is like rolling downhill on a noisy loss surface.”
2. “A Bloom filter is similar to a compact guest list that may admit false positives.”
3. “The immune checkpoint behaves like a brake pedal on T-cell activation.”

**Edge cases**

- The word **“like”** in social media (“I like this”) is **not** analogy. Prefer **multi-token** cues: “is like”, “similar to”, “acts like”.
- If analogy **supports** a further conclusion (“therefore…”), label by the **dominant** move (often Anumana).

### 1.4 Shabda (testimony / authority)

**Definition:** The justification appeals to **authority, documentation, testimony, or institutional sources** (experts, manuals, reports, standards, textbooks).

**Typical cues:** “according to”, “the FDA label states”, “researchers reported”, “the court held”.

**Examples**

1. “According to the IPCC synthesis report, warming is unequivocally human-driven.”
2. “The engineering bulletin certifies this alloy for cryogenic service.”
3. “Reuters reported that the central bank held rates steady.”

**Edge cases**

- **News-like** reporting without explicit attribution can be borderline; prefer Shabda if a **named** institution or role is present.

---

## 2. Reasoning strength (three-way)

Strength is **not** the same as model confidence. It encodes **how well the text supports its own reasoning move** (clarity, connectors, evidence density, hedging).

### 2.1 Strong

**Definition:** Clear structure, explicit connectors or measurement, minimal contradiction, and enough content to evaluate the move.

**Examples**

1. “Bond yields inverted; recessions often follow inversions within two years.” (Anumana, strong)
2. “WHO recommends hand hygiene as a cornerstone of infection control.” (Shabda, strong)

### 2.2 Moderate

**Definition:** Reasonable move, but **thin** evidence, **mixed** signals, or **noticeable hedging**.

**Examples**

1. “It might be GPU-bound, since usage looks high.” (Anumana, moderate)
2. “Experts say exercise helps; details vary by guideline.” (Shabda, moderate)

### 2.3 Weak

**Definition:** Vague, highly speculative, **missing** premises, or **mostly** emotional/slogan-like language.

**Examples**

1. “Obviously they are lying.” (no premises)
2. “This will moon because vibes.” (no substantive support)

---

## 3. Review workflow (`review_status`)

Recommended values:

| `review_status` | Meaning |
|-----------------|--------|
| `pending` | Not yet reviewed |
| `approved` | Two reviewers agree OR single expert reviewer |
| `needs_edit` | Text ok, metadata wrong |
| `rejected` | Exclude from training |

Use `reviewer_notes` for short free-text (e.g., “borderline Shabda vs Anumana; chose Shabda due to named agency”).

---

## 4. Labeling procedure (team)

1. Read the full sentence (or short multi-sentence span) **once** for gist.
2. Decide **pramāṇa** using §1; if ties, choose the **first decisive** move in reading order.
3. Decide **strength** using §2 independently of pramāṇa.
4. Record edge-case reasoning in `reviewer_notes`.
5. Never duplicate exact spans across `master_dataset.csv` and `test_set.csv`.

---

## 5. Quality checklist before merge

- [ ] Balanced counts across the four pramāṇas (±10%).
- [ ] Balanced strength mix (not all “strong”).
- [ ] No empty `text` fields; UTF-8; no leading/trailing whitespace.
- [ ] Avoid toxic or identifiable private content; prefer paraphrases.

---

## 6. Versioning

Update this file when label definitions change, and bump the corpus note in `reviewer_notes` (e.g., “seed corpus v2”).
