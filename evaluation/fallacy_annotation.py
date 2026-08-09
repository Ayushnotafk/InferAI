"""Simple annotation CLI for fallacy dataset.

Usage:
  python evaluation/fallacy_annotation.py --annotator annotator_1

This script randomizes examples, hides existing labels, prompts the annotator
for a label and optional comment, and saves results under
`results/fallacy_annotations/{annotator}.jsonl` without overwriting previous
annotations.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any


DATA_PATH = Path("dataset/fallacy/fallacy_gold.jsonl")
OUT_DIR = Path("results/fallacy_annotations")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_examples():
    examples = []
    with DATA_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            examples.append(json.loads(line))
    return examples


def annotate(annotator: str, limit: int | None = None):
    ex = load_examples()
    random.shuffle(ex)
    if limit:
        ex = ex[:limit]

    out_path = OUT_DIR / f"{annotator}.jsonl"
    # append mode so we don't overwrite previous annotations
    existing_ids = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj.get("id"))
                except Exception:
                    continue

    with out_path.open("a", encoding="utf-8") as outfh:
        for item in ex:
            if item.get("id") in existing_ids:
                continue
            print("\n---\n")
            print("ID:", item.get("id"))
            print("Text:\n", item.get("text"))
            print("Claim:", item.get("claim"))
            print("Premises:\n", item.get("premises"))
            label = input("Assign fallacy label (none/contradiction/unstated_premise/weak_inference/weak_analogy/weak_authority): ").strip()
            comment = input("Optional comment (enter to skip): ").strip()
            ann = {
                "id": item.get("id"),
                "annotator": annotator,
                "assigned_label": label or "none",
                "comment": comment,
            }
            outfh.write(json.dumps(ann, ensure_ascii=False) + "\n")
            outfh.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    annotate(args.annotator, args.limit)


if __name__ == "__main__":
    main()
