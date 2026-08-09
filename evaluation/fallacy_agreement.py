"""Compute Fleiss' Kappa and raw agreement for fallacy annotations.

This script expects per-annotator JSONL files under `results/fallacy_annotations/`
with objects containing `id` and `assigned_label`.

Outputs:
 - results/fallacy/fleiss_kappa.json
 - results/fallacy/fleiss_kappa.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

RESULTS_DIR = Path("results/fallacy")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR = Path("results/fallacy_annotations")


def load_annotations() -> Dict[str, List[str]]:
    """Return mapping id -> list of labels (one per annotator)."""
    data = defaultdict(list)
    files = sorted(ANN_DIR.glob("*.jsonl"))
    if not files:
        print("No annotation files found in results/fallacy_annotations/")
        return {}
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if not obj:
                    continue
                data[obj["id"]].append(obj.get("assigned_label"))
    return data


def fleiss_kappa(counts: List[List[int]]) -> float:
    # counts: N x k matrix where counts[i][j] is number annotators who chose category j for subject i
    N = len(counts)
    if N == 0:
        return 0.0
    k = len(counts[0])
    n = sum(counts[0])
    P = []
    for i in range(N):
        row = counts[i]
        Pi = (sum(c * (c - 1) for c in row)) / (n * (n - 1))
        P.append(Pi)
    P_bar = mean(P)
    p_j = [sum(counts[i][j] for i in range(N)) / (N * n) for j in range(k)]
    P_e = sum(p * p for p in p_j)
    if (1 - P_e) == 0:
        return 0.0
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def main():
    data = load_annotations()
    if not data:
        print("No annotations to process.")
        return
    ids = sorted(data.keys())
    # collect unique labels
    label_set = sorted({lab for labs in data.values() for lab in labs})
    label_to_idx = {l: i for i, l in enumerate(label_set)}

    counts = []
    complete = True
    for id_ in ids:
        labs = data[id_]
        # require equal number of annotators per item
        if len({len(labs)}) != 1:
            complete = False
        row = [0] * len(label_set)
        for lab in labs:
            row[label_to_idx[lab]] += 1
        counts.append(row)

    if not complete:
        print("Annotations appear incomplete or uneven; agreement calculation pending.")

    kappa = fleiss_kappa(counts)

    raw_agreements = []
    for row in counts:
        n = sum(row)
        if n <= 1:
            raw_agreements.append(0.0)
            continue
        raw_agreements.append((sum(c * (c - 1) for c in row)) / (n * (n - 1)))

    result = {
        "fleiss_kappa": kappa,
        "raw_agreement": mean(raw_agreements),
        "n_items": len(ids),
        "n_annotators_per_item": sum(counts[0]) if counts else 0,
        "labels": label_set,
    }

    (RESULTS_DIR / "fleiss_kappa.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Markdown summary
    md = ["# Fallacy Annotation Agreement Report\n"]
    md.append(f"- Items: {result['n_items']}")
    md.append(f"- Annotators per item (first item shown): {result['n_annotators_per_item']}")
    md.append(f"- Labels: {', '.join(label_set)}")
    md.append(f"- Fleiss Kappa: {result['fleiss_kappa']:.4f}")
    md.append(f"- Raw agreement (mean): {result['raw_agreement']:.4f}")
    md.append("\nInterpretation guide:\n- < 0.40 = weak\n- 0.40–0.60 = moderate\n- 0.60–0.80 = substantial\n- > 0.80 = strong\n")
    (RESULTS_DIR / "fleiss_kappa.md").write_text("\n".join(md), encoding="utf-8")
    print("Agreement report written to results/fallacy/")


if __name__ == "__main__":
    main()
