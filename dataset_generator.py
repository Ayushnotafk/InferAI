"""
Build ``dataset/raw/infer_dataset.csv`` for training.

Pipeline
--------
1. (Re)materialize ``dataset/master_dataset.csv`` and ``dataset/test_set.csv``.
2. Synthesize diverse additional rows targeting ~4k total size.
3. Exclude any text that appears in the held-out test file from training.

The training table keeps legacy columns ``text`` and ``label`` where
``label`` is the pramāṇa class name.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

from dataset.labeled_corpus import write_csvs

random.seed(42)

ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Synthetic expansion (higher diversity than the original prototype templates)
# -----------------------------------------------------------------------------

_DOMAINS = [
    "climate modeling",
    "clinical triage",
    "constitutional law",
    "cybersecurity operations",
    "econometrics",
    "history of science",
    "urban planning",
    "school policy",
    "software reliability",
    "astrobiology",
    "journalism ethics",
    "supply chains",
]

_OPENERS = [
    "Taken together, the observations suggest",
    "On balance, the evidence indicates",
    "From what we can verify on the record,",
    "If we treat the premises charitably,",
    "Under ordinary conditions in this domain,",
]

_CONNECTORS = [
    "therefore",
    "hence",
    "so",
    "which implies",
    "which makes it plausible that",
    "which supports the view that",
]


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _synthetic_pratyaksha() -> str:
    d = random.choice(_DOMAINS)
    return _clean(
        f"In {d}, the instrument trace shows a repeatable pattern across three runs. "
        f"The operator log notes the same anomaly at the same timestamp window."
    )


def _synthetic_anumana() -> str:
    d = random.choice(_DOMAINS)
    c = random.choice(_CONNECTORS)
    return _clean(
        f"{random.choice(_OPENERS)} a bottleneck in {d}. "
        f"The downstream metric lags whenever the upstream queue saturates; {c} the system is contention-limited."
    )


def _synthetic_upamana() -> str:
    d = random.choice(_DOMAINS)
    return _clean(
        f"Explaining {d} to newcomers is similar to teaching navigation: "
        f"you start from landmarks people already trust, then map unfamiliar terrain onto those anchors."
    )


def _synthetic_shabda() -> str:
    d = random.choice(_DOMAINS)
    return _clean(
        f"According to the published standard for {d}, reviewers must document assumptions and cite the controlling source. "
        f"Experts in the field treat that requirement as a minimum bar for credibility."
    )


def _synthetic_mixed() -> tuple[str, str]:
    """Occasionally blend two rhetorical moves; label follows the final cue."""
    d = random.choice(_DOMAINS)
    if random.random() < 0.5:
        text = _clean(
            f"We measured the artifact directly in {d}. "
            f"Still, the manual states a different nominal range, so practitioners defer to the handbook when reporting."
        )
        return text, "Shabda"
    text = _clean(
        f"The dashboard in {d} looked stable at first glance. "
        f"Yet latency spikes only during failover, which suggests a race in the controller path."
    )
    return text, "Anumana"


def build_synthetic_rows(target_count: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    generators = [
        ("Pratyaksha", _synthetic_pratyaksha),
        ("Anumana", _synthetic_anumana),
        ("Upamana", _synthetic_upamana),
        ("Shabda", _synthetic_shabda),
    ]
    while len(rows) < target_count:
        if random.random() < 0.12:
            rows.append(_synthetic_mixed())
            continue
        label, fn = random.choice(generators)
        rows.append((fn(), label))
    return rows[:target_count]


def main() -> None:
    write_csvs(ROOT / "dataset")

    master = pd.read_csv(ROOT / "dataset" / "master_dataset.csv")
    test_df = pd.read_csv(ROOT / "dataset" / "test_set.csv")

    exclude = set(test_df["text"].astype(str).str.strip().str.lower())
    master_rows: list[tuple[str, str]] = []
    for _, r in master.iterrows():
        t = str(r["text"]).strip()
        if t.lower() in exclude:
            continue
        master_rows.append((t, str(r["pramana_label"]).strip()))

    # Target total size in [3000, 5000] — fixed for reproducibility.
    target_total = 4200
    synth_n = max(0, target_total - len(master_rows))
    synth_rows = build_synthetic_rows(synth_n)

    combined: list[tuple[str, str]] = master_rows + synth_rows
    random.shuffle(combined)

    out_rows = [(t, lab) for t, lab in combined if t.strip().lower() not in exclude]

    out = pd.DataFrame(out_rows, columns=["text", "label"])
    out_path = ROOT / "dataset" / "raw" / "infer_dataset.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out)} rows (master={len(master_rows)}, synthetic={len(synth_rows)})")


if __name__ == "__main__":
    main()
