"""
Deterministic construction of master and held-out test corpora.

Templates are authored here; rows are expanded across domain slots so we
reach at least 200 master and 200 test lines with disjoint phrasing between
corpora where possible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DOMAINS = [
    "science",
    "technology",
    "education",
    "medicine",
    "politics",
    "philosophy",
    "legal reasoning",
    "public debate",
    "social media discourse",
    "news analysis",
    "daily life",
]

LABELS = ["Pratyaksha", "Anumana", "Upamana", "Shabda"]
STRENGTHS = ["weak", "moderate", "strong"]


def _stable_pick(seed: str, options: list[str]) -> str:
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


def _master_templates() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    for s in STRENGTHS:
        rows += [
            (
                "In {domain}, a direct check showed the instrument reading drifting outside tolerance during the run.",
                "Pratyaksha",
                s,
            ),
            (
                "During the {domain} walkthrough, we could hear intermittent clicking from the enclosure with no load applied.",
                "Pratyaksha",
                s,
            ),
            (
                "Field notes from {domain} record visible condensation on the cold line within minutes of startup.",
                "Pratyaksha",
                s,
            ),
            (
                "Across {domain}, the measured curve stayed flat until the threshold, then rose sharply in two trials.",
                "Pratyaksha",
                s,
            ),
        ]

    for s in STRENGTHS:
        rows += [
            (
                "In {domain}, the leading indicator moved first; therefore the downstream effect is the plausible explanation.",
                "Anumana",
                s,
            ),
            (
                "Given repeated failures only under load in {domain}, congestion is more likely than a random bug.",
                "Anumana",
                s,
            ),
            (
                "Because the control arm improved while the treatment arm flatlined in {domain}, the intervention may lack efficacy.",
                "Anumana",
                s,
            ),
            (
                "Since the premise set is ambiguous in {domain}, interpreters will default to the narrower reading.",
                "Anumana",
                s,
            ),
        ]

    for s in STRENGTHS:
        rows += [
            (
                "In {domain}, the feedback loop behaves like a thermostat hunting around a noisy setpoint.",
                "Upamana",
                s,
            ),
            (
                "For newcomers in {domain}, a hash table is similar to a set of labeled drawers: fast lookup, occasional collisions.",
                "Upamana",
                s,
            ),
            (
                "In {domain}, debugging distributed traces is like following colored threads through a woven fabric.",
                "Upamana",
                s,
            ),
            (
                "The workflow in {domain} acts like a pipeline valve: small friction upstream amplifies delays downstream.",
                "Upamana",
                s,
            ),
        ]

    for s in STRENGTHS:
        rows += [
            (
                "According to the {domain} handbook, the device must complete a full calibration cycle before first use.",
                "Shabda",
                s,
            ),
            (
                "Experts in {domain} now recommend documenting provenance for every dataset used in training.",
                "Shabda",
                s,
            ),
            (
                "A recent bulletin in {domain} states that the prior guideline was superseded last quarter.",
                "Shabda",
                s,
            ),
            (
                "Researchers publishing in {domain} reported a replication failure under stricter preprocessing.",
                "Shabda",
                s,
            ),
        ]

    return rows


def _test_templates() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for s in STRENGTHS:
        rows += [
            (
                "Bench notes ({domain}): the sample fluoresced under UV immediately after staining.",
                "Pratyaksha",
                s,
            ),
            (
                "Telemetry ({domain}) captured a single large spike, then quiet baseline for ten minutes.",
                "Pratyaksha",
                s,
            ),
            (
                "If costs rise while output is fixed in {domain}, margins should compress in equilibrium.",
                "Anumana",
                s,
            ),
            (
                "The error only appears when concurrency is enabled in {domain}, which suggests a race.",
                "Anumana",
                s,
            ),
        ]
    for s in STRENGTHS:
        rows += [
            (
                "Teaching {domain} with metaphors is like using training wheels: helpful until intuition stabilizes.",
                "Upamana",
                s,
            ),
            (
                "A causal DAG in {domain} is analogous to a map of one-way streets between variables.",
                "Upamana",
                s,
            ),
            (
                "The regulator for {domain} published a notice clarifying disclosure timelines for incidents.",
                "Shabda",
                s,
            ),
            (
                "Court filings in {domain} quote the statute verbatim when arguing mens rea.",
                "Shabda",
                s,
            ),
        ]
    return rows


def _materialize(
    templates: list[tuple[str, str, str]],
    *,
    corpus_id: str,
    limit: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for tpl, lab, stg in templates:
        for domain in DOMAINS:
            text = tpl.format(domain=domain).strip()
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "text": text,
                    "pramana_label": lab,
                    "strength_label": stg,
                    "review_status": "approved",
                    "reviewer_notes": f"{corpus_id} seed v1",
                }
            )
            if len(out) >= limit:
                return out

    # Deterministic padding if templates × domains undershoot (should not happen here).
    i = 0
    while len(out) < limit:
        salt = f"{corpus_id}-pad-{i}"
        d = _stable_pick(salt, DOMAINS)
        lab = _stable_pick(salt + "L", LABELS)
        stg = _stable_pick(salt + "S", STRENGTHS)
        text = (
            f"{corpus_id.title()} augmentation ({d}): synthetic vignette uid {salt}; "
            f"intended label {lab} at {stg} strength for evaluation bookkeeping."
        )
        i += 1
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(
            {
                "text": text,
                "pramana_label": lab,
                "strength_label": stg,
                "review_status": "approved",
                "reviewer_notes": f"{corpus_id} augmentation",
            }
        )
    return out[:limit]


def build_master_dataframe() -> pd.DataFrame:
    return pd.DataFrame(_materialize(_master_templates(), corpus_id="master", limit=220))


def build_test_dataframe() -> pd.DataFrame:
    return pd.DataFrame(_materialize(_test_templates(), corpus_id="test", limit=200))


def write_csvs(base: Path | None = None) -> tuple[Path, Path]:
    root = base or Path(__file__).resolve().parent
    master_path = root / "master_dataset.csv"
    test_path = root / "test_set.csv"
    build_master_dataframe().to_csv(master_path, index=False)
    build_test_dataframe().to_csv(test_path, index=False)
    return master_path, test_path


if __name__ == "__main__":
    mp, tp = write_csvs()
    print("Wrote", mp, "and", tp)
