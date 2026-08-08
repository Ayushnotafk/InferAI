#!/usr/bin/env python3
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FIELDS = ["id", "text", "label", "claim", "premises", "reasoning_indicator", "domain", "difficulty", "source_type"]
VALID_LABELS = {"Pratyaksha", "Anumana", "Upamana", "Shabda"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_DOMAINS = {
    "everyday life", "education", "computer science", "software engineering", "networking", "cybersecurity", "artificial intelligence", "machine learning",
    "mathematics", "physics", "chemistry", "biology", "medicine", "environment", "weather", "engineering", "electronics", "robotics", "economics",
    "finance", "law", "history", "geography", "astronomy", "agriculture", "transportation", "manufacturing", "sports", "workplace situations",
    "social situations", "technology", "scientific research",
}


def fail(msg):
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


def normalize_text(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def near_duplicate_count(rows):
    import numpy as np
    from scipy.sparse import csr_matrix
    
    vocab = {}
    indptr = [0]
    indices = []
    data = []
    lengths = []
    
    for r in rows:
        words = set(re.findall(r"\b[\w-]+\b", r["text"].lower()))
        lengths.append(len(words))
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
            indices.append(vocab[w])
            data.append(1)
        indptr.append(len(indices))
        
    X = csr_matrix((data, indices, indptr), dtype=np.float32)
    intersection = X.dot(X.T)
    
    coo = intersection.tocoo()
    mask = coo.row < coo.col
    row = coo.row[mask]
    col = coo.col[mask]
    val = coo.data[mask]
    
    lengths = np.array(lengths, dtype=np.float32)
    jaccard = val / (lengths[row] + lengths[col] - val)
    
    count = np.sum(jaccard >= 0.80)
    return int(count)


def main():
    errors = 0

    dataset_jsonl = ROOT / "dataset_10000.jsonl"
    dataset_csv = ROOT / "dataset_10000.csv"
    train_csv = ROOT / "train.csv"
    validation_csv = ROOT / "validation.csv"
    test_csv = ROOT / "test.csv"

    needed_files = [dataset_jsonl, dataset_csv, train_csv, validation_csv, test_csv]
    for p in needed_files:
        if not p.exists():
            errors += fail(f"Missing required artifact: {p.name}")

    if errors:
        return 1

    rows = []
    with open(dataset_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                errors += fail(f"Blank line in JSONL at line {line_no}")
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                errors += fail(f"Malformed JSONL at line {line_no}: {e}")
                continue
            rows.append(obj)

    if len(rows) != 10000:
        errors += fail(f"Expected exactly 10000 records, found {len(rows)}")

    # Required fields
    for row_no, row in enumerate(rows, 1):
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            errors += fail(f"Record {row_no} is missing fields: {', '.join(missing)}")
            continue
        for field in REQUIRED_FIELDS:
            if row.get(field) in (None, ""):
                errors += fail(f"Record {row_no} has empty value in required field '{field}'")
                break
        if row.get("label") not in VALID_LABELS:
            errors += fail(f"Record {row_no} has invalid label {row.get('label')}")
        if row.get("difficulty") not in VALID_DIFFICULTIES:
            errors += fail(f"Record {row_no} has invalid difficulty {row.get('difficulty')}")
        if row.get("domain") not in VALID_DOMAINS:
            errors += fail(f"Record {row_no} has invalid domain {row.get('domain')}")
        if row.get("source_type") != "synthetic":
            errors += fail(f"Record {row_no} has invalid source_type {row.get('source_type')}")

    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors += fail("Duplicate IDs found")
    if not all(re.match(r"^INF\d{5}$", x) for x in ids):
        errors += fail("IDs are not in expected INF00001 format")

    texts = [normalize_text(r["text"]) for r in rows]
    if len(texts) != len(set(texts)):
        errors += fail("Normalized text duplicates found")

    c = Counter(r["label"] for r in rows)
    for label in VALID_LABELS:
        if c.get(label, 0) != 2500:
            errors += fail(f"Class balance failure: {label}={c.get(label, 0)}")
    if set(c.keys()) != VALID_LABELS:
        errors += fail("Unexpected labels present")

    for csv_path in [dataset_csv, train_csv, validation_csv, test_csv]:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != REQUIRED_FIELDS:
                errors += fail(f"Schema mismatch in {csv_path.name}: expected {REQUIRED_FIELDS}")

    # Split validation: deterministic stratified files
    expected_by_split = {
        "train": (7000, {label: 1750 for label in VALID_LABELS}),
        "validation": (1500, {label: 375 for label in VALID_LABELS}),
        "test": (1500, {label: 375 for label in VALID_LABELS}),
    }
    for split_name, split_path in {"train": train_csv, "validation": validation_csv, "test": test_csv}.items():
        with open(split_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            split_rows = list(reader)
        expected_count, expected_label_counts = expected_by_split[split_name]
        if len(split_rows) != expected_count:
            errors += fail(f"Split file {split_name} has {len(split_rows)} rows, expected {expected_count}")
        observed_label_counts = Counter(r["label"] for r in split_rows)
        for label in VALID_LABELS:
            if observed_label_counts.get(label, 0) != expected_label_counts[label]:
                errors += fail(f"Split {split_name} has label count {label}={observed_label_counts.get(label, 0)}, expected {expected_label_counts[label]}")

    # Leakage check: trivial train/test overlap (normalized text)
    train_rows = []
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        train_rows = [{k: v for k, v in r.items()} for r in csv.DictReader(f)]
    test_rows = []
    with open(test_csv, "r", encoding="utf-8", newline="") as f:
        test_rows = [{k: v for k, v in r.items()} for r in csv.DictReader(f)]
    train_texts = {normalize_text(r["text"]) for r in train_rows}
    test_texts = {normalize_text(r["text"]) for r in test_rows}
    if train_texts & test_texts:
        errors += fail("Train/test leakage: normalized text overlap detected")

    # Near duplicates in final dataset
    if near_duplicate_count(rows) > 50:
        errors += fail(f"Final selected dataset has too many near duplicates: {near_duplicate_count(rows)}")

    # JSONL schema vs CSV schema
    with open(dataset_csv, "r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(rows):
        errors += fail("CSV/JSONL row count mismatch")

    if errors:
        return 1

    print("PASS: validate_dataset.py verification succeeded")
    print(f"Rows: {len(rows)}; labels: {dict(c)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
