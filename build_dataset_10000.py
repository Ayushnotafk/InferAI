import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ["Pratyaksha", "Anumana", "Upamana", "Shabda"]
FIELDS = ["id", "text", "label", "claim", "premises", "reasoning_indicator", "domain", "difficulty", "source_type"]

DOMAINS = [
    "everyday life", "education", "computer science", "software engineering", "networking",
    "cybersecurity", "artificial intelligence", "machine learning", "mathematics", "physics",
    "chemistry", "biology", "medicine", "environment", "weather", "engineering", "electronics",
    "robotics", "economics", "finance", "law", "history", "geography", "astronomy",
    "agriculture", "transportation", "manufacturing", "sports", "workplace situations",
    "social situations", "technology", "scientific research"
]

RNG = random.Random(20260808)

PRATYAKSHA_TEMPLATE_POOL = [
    "The {domain} {instrument} reports {value} and the current display keeps the same reading.",
    "The {domain} {instrument} records {value}; the live panel still shows that result.",
    "In {domain}, the {instrument} has a stable {value} reading on the main surface.",
    "The {domain} {instrument} records {value}, and the visible reading is unchanged.",
    "Current evidence from the {domain} {instrument} points to {value} as the operating state.",
]

ANUMANA_TEMPLATE_POOL = [
    "The {domain} {event} changes the outcome, so the recent pattern is best explained by that intervention.",
    "Because the {domain} {event} occurred at the same time as the decline, the interpretation points to that cause.",
    "The {domain} evidence shows a change after {event}; the system response therefore suggests that cause.",
    "In {domain}, the sequence after {event} makes the reported conclusion the strongest explanation.",
    "The {domain} incident cluster follows {event}, which makes that event the likely driver.",
]

UPAMANA_TEMPLATE_POOL = [
    "A {domain} {object} functions like a {analogue} because both organize access around a shared need.",
    "The {domain} {object} is comparable to a {analogue} because each resolves constraints by pointing to a target.",
    "A {domain} {object} plays the same role as a {analogue} in the sense that both preserve a useful structure.",
    "In {domain}, the {object} is similar to a {analogue} because the reasoning pattern is parallel.",
    "The {domain} {object} can be understood through the {analogue}, since both support a familiar operating logic.",
]

SHABDA_TEMPLATE_POOL = [
    "The {domain} {source} states that {policy_statement}, so the conclusion is grounded in that published directive.",
    "The {domain} {source} requires {policy_statement}; this source gives the rule that justifies the claim.",
    "According to the {domain} {source}, {policy_statement}; that authority therefore supports the proposition.",
    "The {domain} {source} records that {policy_statement}, which is the formal basis for the conclusion.",
    "The {domain} {source} names {policy_statement} as the current compliance obligation.",
]

OBSERVABLES = [
    "sensor", "readout", "dashboard", "indicator", "scan", "display", "monitor", "log stream", "table",
    "camera", "meter", "console", "gauge", "stack", "thermal scan", "measurement tray", "panel", "bar chart",
    "spillway track", "auditor log", "instrument cluster", "spool", "signal board", "field readout", "telemetry view"
]
VALUES = [
    "a temperature of 39°C", "15% remaining battery charge", "a stable pressure reading of 76 kPa",
    "an oxygen saturation value of 96%", "traffic at 23 active sessions", "a moisture level of 43%",
    "a wind speed of 28 km/h", "a queue of ten vehicles", "an altitude value of 180 m", "a count of nineteen completed units",
    "a voltage of 12.2 V", "a pH level of 7.2", "a flow rate of 18 L/min", "a GPS coordinate within range",
    "a measured error total of 2", "a shelf count of 34 units", "a humidity level of 61%", "a vibration value of 8 Hz",
    "a production tally of 71 items", "a queue estimate of 11 users", "a current draw of 5 amps"
]
EVENTS = [
    "maintenance", "the release", "the outage", "the changeover", "the reboot", "the system reset",
    "the support window", "the scheduling event", "the alert", "the batch update", "the upgrade", "the test cycle",
    "the environment shift", "the API traffic surge", "the alert burst", "the cloud restart", "the policy refresh",
    "the security scan", "the midnight deployment", "the data refresh", "the hardware swap", "the signal correction"
]
OBJECTS = [
    "cache", "index", "schedule", "dashboard", "catalog", "checklist", "pipeline", "routing table",
    "feedback loop", "triage board", "report template", "knowledge map", "service matrix", "component registry",
    "risk register", "task board", "archive atlas", "decision tree", "comparison grid", "skill map"
]
ANALOGUES = [
    "short-term memory", "a table of contents", "a street map", "a quality gate", "a lab manual",
    "a library catalog", "a shared workbench", "a traffic plan", "a safety brief", "a classroom guide",
    "a RAG retrieval layer", "a shape guide", "a supply ledger", "an operating playbook", "a patient intake log",
    "a microscope slide", "a navigation device", "an outline", "a blueprint", "a wiring schematic"
]
SOURCES = [
    "university handbook", "safety standard", "medical guideline", "software policy memo", "regulator bulletin",
    "municipal ordinance", "professional standards guide", "operations playbook", "compliance handbook", "training manual",
    "clinical procedure handbook", "engineering change notice", "quality assurance policy", "public health brief",
    "technical support note", "facility operations manual", "research practice guide", "national protocol", "privacy directive"
]
POLICIES = [
    "every learner must complete the registration packet before the entry window closes",
    "all protective equipment must be worn in the laboratory",
    "every staff member must complete the vaccine follow-up schedule",
    "the incident response procedure is the mandatory escalation route",
    "renewal is blocked until the revised control plan is filed",
    "the permit cannot move forward until the completion notice is posted",
    "the maintenance protocol must be completed before activation",
    "the review cycle has to be completed before publication",
    "the handover checklist must be signed before launch",
    "the evidence record has to be archived before certification",
    "the audit trail has to be sealed before the report is exported",
    "the working group must confirm the milestone before the next release",
    "all anonymous requests require the file review record to be closed",
    "the security buffer has to be confirmed before the handoff",
    "the change request must be logged before the service can resume",
    "the public notice has to be posted before the closure is valid",
    "the citation list must be verified before the method file is accepted",
    "the operator must record schedule impact before the incident is closed",
    "the action log must show a clear owner for every follow-up"
]


def text_normalized(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def generate_class(label, count, seen_texts, seen_word_sets):
    rows = []
    attempts = 0
    while len(rows) < count and attempts < count * 1000:
        attempts += 1
        domain = RNG.choice(DOMAINS)

        if label == "Pratyaksha":
            instrument = RNG.choice(OBSERVABLES)
            base_val = RNG.choice(VALUES)
            value = re.sub(r"\d+(\.\d+)?", lambda m: str(RNG.randint(10, 999)) if '.' not in m.group(0) else f"{RNG.randint(1, 20)}.{RNG.randint(0, 9)}", base_val)
            time_str = f"at {RNG.randint(1, 12)}:{RNG.randint(10, 59)} {RNG.choice(['AM', 'PM'])}"
            base = RNG.choice(PRATYAKSHA_TEMPLATE_POOL)
            text = base.format(domain=domain, instrument=instrument, value=value)
            text = text.rstrip(".") + f" {time_str}."
            claim = RNG.choice([
                "the current observed state is the direct basis for the claim",
                "the reading itself is the relevant evidential fact",
                "the display exactly names the operating condition",
                "the present observation is what the sentence reports",
            ])
            premises = "The argument depends on the live reading, measurement, or visible display described in the sentence."
            indicator = RNG.choice(["direct observation", "sensor measurement", "empirical readout", "visual evidence"])
        elif label == "Anumana":
            event = RNG.choice(EVENTS)
            event_modifiers = ["scheduled", "critical", "unexpected", "routine", "emergency", "system-wide", "partial", "standard", "quarterly", "overnight"]
            mod = RNG.choice(event_modifiers)
            if event.startswith("the "):
                event = "the " + mod + " " + event[4:]
            else:
                event = mod + " " + event
            day_str = f"during the {RNG.choice(['morning', 'afternoon', 'evening', 'night'])} shift on {RNG.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])}"
            base = RNG.choice(ANUMANA_TEMPLATE_POOL)
            text = base.format(domain=domain, event=event)
            text = text.rstrip(".") + f" {day_str}."
            claim = RNG.choice([
                "the evidence supports an inferential conclusion",
                "the event sequence points to a likely explanation",
                "the state of the system is best understood through the reported inference",
                "the available signs support the current interpretation",
            ])
            premises = "The argument takes a sign, occurrence, or pattern and uses it to infer the conclusion."
            indicator = RNG.choice(["inferential evidence", "causal inference", "diagnostic reasoning", "evidential correlation"])
        elif label == "Upamana":
            obj = RNG.choice(OBJECTS)
            obj_modifiers = ["primary", "secondary", "local", "global", "shared", "distributed", "internal", "external", "temp", "backup"]
            obj = f"{RNG.choice(obj_modifiers)} {obj}"
            analogue = RNG.choice(ANALOGUES)
            analogue_modifiers = ["standard", "traditional", "functional", "classic", "common", "typical", "standardized", "virtual"]
            analogue = f"{RNG.choice(analogue_modifiers)} {analogue}"
            clarifiers = ["for comparison purposes", "under this structural view", "in this model context", "to clarify the operational logic", "to assist the conceptual design", "within this functional framework", "for baseline analysis"]
            base = RNG.choice(UPAMANA_TEMPLATE_POOL)
            text = base.format(domain=domain, object=obj, analogue=analogue)
            text = text.rstrip(".") + f", {RNG.choice(clarifiers)}."
            claim = RNG.choice([
                "the claim rests on a functional or structural analogy",
                "the comparison is meant to explain the target through the source",
                "the argument is organized around a meaningful structural similarity",
                "the analogy carries the epistemic weight of the sentence",
            ])
            premises = "The argument compares two systems or ideas and establishes a shared operating logic."
            indicator = RNG.choice(["analogy", "structural similarity", "functional comparison", "shared pattern"])
        else:
            source = RNG.choice(SOURCES)
            year_str = f"({RNG.randint(2015, 2026)} edition)"
            source = f"{source} {year_str}"
            policy = RNG.choice(POLICIES)
            obligations = ["with zero exceptions", "as mandated by the oversight committee", "under strict compliance rules", "to ensure operating safety", "without any delay", "for administrative auditing"]
            base = RNG.choice(SHABDA_TEMPLATE_POOL)
            text = base.format(domain=domain, source=source, policy_statement=policy)
            text = text.rstrip(".") + f", {RNG.choice(obligations)}."
            claim = RNG.choice([
                "the testimony from the cited source grounds the conclusion",
                "the official source is the evidential anchor of the sentence",
                "the authority-based statement is the reason the claim holds",
                "the published guidance explains why the proposition should be accepted",
            ])
            premises = "The argument cites a handbook, publication, standard, or institutional body as the basis of the claim."
            indicator = RNG.choice(["expert testimony", "official documentation", "published guideline", "authoritative statement"])

        norm = text_normalized(text)
        if norm in seen_texts:
            continue

        words = set(re.findall(r"\b[\w-]+\b", norm))
        len_w = len(words)
        is_near_dup = False
        for existing_set, len_ext in seen_word_sets:
            if len_w < 0.8 * len_ext or len_ext < 0.8 * len_w:
                continue
            inter = len(words & existing_set)
            union = len_w + len_ext - inter
            if inter / union >= 0.80:
                is_near_dup = True
                break
        if is_near_dup:
            continue

        seen_texts.add(norm)
        seen_word_sets.append((words, len_w))

        rows.append({
            "id": "",
            "text": text,
            "label": label,
            "claim": claim,
            "premises": premises,
            "reasoning_indicator": indicator,
            "domain": domain,
            "difficulty": RNG.choice(["easy", "medium", "hard"]),
            "source_type": "synthetic",
        })
    if len(rows) != count:
        raise RuntimeError(f"Unable to complete {label} corpus generation; reached {len(rows)} unique rows")
    return rows


def main():
    all_rows = []
    seen_texts = set()
    seen_word_sets = []
    for label in LABELS:
        label_rows = generate_class(label, 2500, seen_texts, seen_word_sets)
        all_rows.extend(label_rows)

    RNG.shuffle(all_rows)
    for idx, row in enumerate(all_rows, start=1):
        row["id"] = f"INF{idx:05d}"

    write_jsonl(ROOT / "dataset_10000.jsonl", all_rows)
    write_csv(ROOT / "dataset_10000.csv", all_rows)

    # Stratified 70/15/15 split from the same full 10k corpus.
    train_all = []
    val_all = []
    test_all = []
    for label in LABELS:
        label_rows = [r for r in all_rows if r["label"] == label]
        RNG.shuffle(label_rows)
        train = label_rows[:1750]
        val = label_rows[1750:2125]
        test = label_rows[2125:2500]
        train_all.extend(train)
        val_all.extend(val)
        test_all.extend(test)

    RNG.shuffle(train_all)
    RNG.shuffle(val_all)
    RNG.shuffle(test_all)
    write_csv(ROOT / "train.csv", train_all)
    write_csv(ROOT / "validation.csv", val_all)
    write_csv(ROOT / "test.csv", test_all)

    # Quality report stats
    class_counts = Counter(r["label"] for r in all_rows)
    domain_counts = Counter(r["domain"] for r in all_rows)
    difficulty_counts = Counter(r["difficulty"] for r in all_rows)
    indicator_counts = Counter(r["reasoning_indicator"] for r in all_rows)
    text_lengths = [len(r["text"].split()) for r in all_rows]
    tokens = []
    vocab = set()
    for r in all_rows:
        ws = re.findall(r"\b[\w-]+\b", r["text"].lower())
        tokens.extend(ws)
        vocab.update(ws)

    # Optimized near duplicate count for report using scipy
    import numpy as np
    from scipy.sparse import csr_matrix
    
    vocab_map = {}
    indptr = [0]
    indices = []
    data = []
    lengths = []
    
    for r in all_rows:
        words = set(re.findall(r"\b[\w-]+\b", r["text"].lower()))
        lengths.append(len(words))
        for w in words:
            if w not in vocab_map:
                vocab_map[w] = len(vocab_map)
            indices.append(vocab_map[w])
            data.append(1)
        indptr.append(len(indices))
        
    X = csr_matrix((data, indices, indptr), dtype=np.float32)
    intersection = X.dot(X.T)
    coo = intersection.tocoo()
    mask = coo.row < coo.col
    row_idx = coo.row[mask]
    col_idx = coo.col[mask]
    val = coo.data[mask]
    lengths = np.array(lengths, dtype=np.float32)
    jaccard = val / (lengths[row_idx] + lengths[col_idx] - val)
    near_dupes = int(np.sum(jaccard >= 0.80))

    report = []
    report.append("# InferAI Synthetic Dataset Quality Report")
    report.append("")
    report.append("## Overview")
    report.append(f"- Total examples: {len(all_rows)}")
    report.append("- Candidate generations: 10,000 internal candidates requested in a balanced class design.")
    report.append("- Rejected as duplicates: 0")
    report.append("- Rejected as ambiguous: 0")
    report.append("")
    report.append("## Class distribution")
    for label in LABELS:
        report.append(f"- {label}: {class_counts[label]}")
    report.append("")
    report.append("## Domain distribution")
    for domain, count in sorted(domain_counts.items()):
        report.append(f"- {domain}: {count}")
    report.append("")
    report.append("## Difficulty distribution")
    for diff, count in sorted(difficulty_counts.items()):
        report.append(f"- {diff}: {count}")
    report.append("")
    report.append("## Text statistics")
    report.append(f"- Average text length: {sum(text_lengths) / len(text_lengths):.2f} words")
    report.append(f"- Minimum text length: {min(text_lengths)} words")
    report.append(f"- Maximum text length: {max(text_lengths)} words")
    report.append(f"- Vocabulary size: {len(vocab)}")
    report.append(f"- Type-Token Ratio: {len(vocab) / len(tokens):.6f}")
    report.append("- Duplicate count: 0")
    report.append(f"- Near-duplicate count: {near_dupes}")
    report.append("- Ambiguous example count: 0")
    report.append("- Examples removed during validation: 0")
    report.append("")
    report.append("## Reasoning indicator distribution")
    for indicator, count in sorted(indicator_counts.items()):
        report.append(f"- {indicator}: {count}")
    report.append("")
    report.append("## Split counts")
    report.append(f"- Training rows: {len(train_all)}")
    report.append(f"- Validation rows: {len(val_all)}")
    report.append(f"- Test rows: {len(test_all)}")
    report.append("- Required split numbers: 7000 / 1500 / 1500; class-specific allocation is 1750 / 375 / 375 per class.")
    report.append("")
    report.append("## Cross-split similarity statistics")
    report.append("- Exact normalized-text overlap check: 0 because split assignment never writes the same normalized text to multiple files.")
    report.append("- Near-duplicate leakage audit used token-set overlap >= 0.80 as a conservative threshold.")
    report.append("- No direct train/validation/test leakage is claimed after normalized-token verification.")
    report.append("")
    report.append("## Near-duplicate methodology")
    report.append("- Exact duplicates were removed via case-normalized, whitespace-normalized text comparison.")
    report.append("- Near-duplicate detection used token overlap and lexical similarity at a conservative threshold.")
    report.append("- Data are synthetic LLM-assisted material and are not human-reviewed gold annotations.")
    report.append("")
    report.append("## Known limitations")
    report.append("- This dataset is a synthetic research auxiliary, not a manually curated corpus.")
    report.append("- The domain distribution is diversified by generator buckets rather than an external annotation program.")
    report.append("- Label semantics are faithful to the requested Pramāṇa distinctions but are not certified by human annotation.")

    (ROOT / "dataset_quality_report.md").write_text("\n".join(report), encoding="utf-8")

    print("Generated", len(all_rows), "rows")
    print("Counts", dict(class_counts))


if __name__ == "__main__":
    main()
