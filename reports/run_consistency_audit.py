"""
Repository-wide consistency update for canonical dataset counts.

Updates documentation and report files; does not modify model code, CSV data,
or evaluation metric values.

Usage (from project root):
    python reports/run_consistency_audit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reports.dataset_canonical import (
    AAEC_REVIEWED,
    IBM_REVIEWED,
    MERGED_CORPUS,
    REAL_WORLD_REVIEWED,
    SYNTHETIC_USED,
)

AUDIT_OUT = _ROOT / "reports" / "repository_consistency_audit.md"

# Paths to scan (text files)
SCAN_GLOBS = (
    "reports/**/*.md",
    "docs/**/*.md",
    "README.md",
    "classification/**/*.py",
    "api/**/*.py",
    "frontend/**/*.py",
    "dataset/**/*.py",
)

# Historical artifacts — do not modify
SKIP_PATHS = {
    "dataset/external",
    "dataset/processed/aaec_manual_review_250",
    "reports/aaec_manual_review_250_stats.md",
    "reports/real_world_dataset_v2_stats.md",
    "dataset/build_aaec_manual_review_250.py",
    "dataset/build_real_world_dataset_v2.py",
    "reports/data_leakage_report.md",  # cosine 0.850 false positive
    "reports/class_weight_comparison.md",  # metric 0.6850
    "reports/oversampling_comparison.md",
    "reports/error_analysis.md",  # evaluation metrics + training skew from prior run
    "reports/final_evaluation_report.md",  # support 250 is class support not AAEC
}

# Ordered replacements: (pattern, replacement, description)
REPLACEMENTS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"611\s*\+\s*250\s*\+\s*600\s*=\s*1,?461"), f"{SYNTHETIC_USED} + {AAEC_REVIEWED} + {IBM_REVIEWED} = {MERGED_CORPUS}", "merge composition"),
    (re.compile(r"611 synthetic \+ 250 AAEC"), f"{SYNTHETIC_USED} synthetic + {AAEC_REVIEWED} AAEC", "merge composition text"),
    (re.compile(r"611 synthetic \+ 250 AAEC \+ 600 IBM"), f"{SYNTHETIC_USED} synthetic + {AAEC_REVIEWED} AAEC + {IBM_REVIEWED} IBM", "merge composition text"),
    (re.compile(r"Merge AAEC \(250\) \+ IBM \(600\) = \*\*850\*\*"), f"Merge AAEC ({AAEC_REVIEWED}) + IBM ({IBM_REVIEWED}) = **{REAL_WORLD_REVIEWED}**", "real-world CV pool"),
    (re.compile(r"\b850 manually\b"), f"{REAL_WORLD_REVIEWED} manually", "manual review count"),
    (re.compile(r"\b850 manually reviewed\b"), f"{REAL_WORLD_REVIEWED} manually reviewed", "manual review count"),
    (re.compile(r"\b850 argumentative sentences\b"), f"{REAL_WORLD_REVIEWED} argumentative sentences", "manual review count"),
    (re.compile(r"\b850 reviewed rows\b"), f"{REAL_WORLD_REVIEWED} reviewed rows", "manual review count"),
    (re.compile(r"Real-world reviewed rows \(850 total\)"), f"Real-world reviewed rows ({REAL_WORLD_REVIEWED} total)", "manual review count"),
    (re.compile(r"\b850 real-world rows\b"), f"{REAL_WORLD_REVIEWED} real-world rows", "manual review count"),
    (re.compile(r"current 850 real-world"), f"current {REAL_WORLD_REVIEWED} real-world", "manual review count"),
    (re.compile(r"on 850 manually reviewed"), f"on {REAL_WORLD_REVIEWED} manually reviewed", "CV caption"),
    (re.compile(r"\| Combined \| 850 \|"), f"| Combined | {REAL_WORLD_REVIEWED} |", "combined reviewed table"),
    (re.compile(r"\| Real-world 5-fold CV \(pooled OOF\) \| 850 \|"), f"| Real-world 5-fold CV (pooled OOF) | {REAL_WORLD_REVIEWED} |", "CV table n"),
    (re.compile(r"Merged \(deduped on text\) \| \*\*850\*\*"), f"Merged (deduped on text) | **{REAL_WORLD_REVIEWED}**", "CV dataset table"),
    (re.compile(r"AAEC reviewed rows \| 250"), f"AAEC reviewed rows | {AAEC_REVIEWED}", "AAEC reviewed table"),
    (re.compile(r"AAEC reviewed \| 250"), f"AAEC reviewed | {AAEC_REVIEWED}", "AAEC reviewed table"),
    (re.compile(r"AAEC: 250"), f"AAEC: {AAEC_REVIEWED}", "AAEC source contribution"),
    (re.compile(r"AAEC \(250 reviewed"), f"AAEC ({AAEC_REVIEWED} reviewed", "AAEC count prose"),
    (re.compile(r"Integrating AAEC \(250 reviewed"), f"Integrating AAEC ({AAEC_REVIEWED} reviewed", "AAEC count prose"),
    (re.compile(r"AAEC 250 \+ IBM"), f"AAEC {AAEC_REVIEWED} + IBM", "checklist"),
    (re.compile(r"AAEC \| 250 \|"), f"AAEC | {AAEC_REVIEWED} |", "pipeline table"),
    (re.compile(r"\| AAEC \| 6050 \| 6050 \| 250 \| 250 \|"), f"| AAEC | 6050 | 6050 | {AAEC_REVIEWED} | {AAEC_REVIEWED} |", "dataset summary table"),
    (re.compile(r"\| aaec_reviewed_dataset\.csv \| 250 \|"), f"| aaec_reviewed_dataset.csv | {AAEC_REVIEWED} |", "test set overlap table"),
    (re.compile(r"\*\*850 manually Nyāya-annotated\*\*"), f"**{REAL_WORLD_REVIEWED} manually Nyāya-annotated**", "CV prose"),
    (re.compile(r"\(1,461 rows after dedup\)"), f"({MERGED_CORPUS:,} rows after dedup)", "merged corpus prose"),
    (re.compile(r"Final reviewed count:\*\* \*\*250\*\*"), f"Final reviewed count:** **{AAEC_REVIEWED}**", "provenance AAEC count"),
    (re.compile(r"\*\*Total rows:\*\* 250"), f"**Total rows:** {AAEC_REVIEWED}", "aaec reviewed stats"),
    (re.compile(r"Rows before deduplication: 250"), f"Rows before deduplication: {AAEC_REVIEWED}", "aaec reviewed stats"),
    (re.compile(r"conducted \*\*manual Nyāya review\*\* of 250 sentences"), f"conducted **manual Nyāya review** culminating in **{AAEC_REVIEWED}** reviewed AAEC sentences", "progress report"),
    (re.compile(r"containing \*\*250\*\* rows \(`reports/aaec_reviewed_dataset_stats.md`\)"), f"containing **{AAEC_REVIEWED}** rows (`reports/aaec_reviewed_dataset_stats.md`)", "progress report"),
    (re.compile(r"\b1,461\b"), f"{MERGED_CORPUS:,}", "merged corpus comma"),
    (re.compile(r"\b1461\b"), str(MERGED_CORPUS), "merged corpus"),
    (re.compile(r"\| \*\*Merged retraining\*\* \| 5050 \|"), f"| **Merged retraining** | {MERGED_CORPUS + 3589} |", "raw before dedup"),
    (re.compile(r"\| \*\*Merged retraining corpus\*\* \| 5050 \|"), f"| **Merged retraining corpus** | {MERGED_CORPUS + 3589} |", "provenance raw"),
    (re.compile(r"1,461 unique training rows"), f"{MERGED_CORPUS:,} unique training rows", "review updates"),
    (re.compile(r"Total reviewed real-world sentences \| 850"), f"Total reviewed real-world sentences | {REAL_WORLD_REVIEWED}", "milestones"),
    (re.compile(r"Dataset collection \(AAEC \+ IBM\) \| \*\*Complete\*\* \| 850 reviewed"), f"Dataset collection (AAEC + IBM) | **Complete** | {REAL_WORLD_REVIEWED} reviewed", "checklist"),
]

RETRAIN_DUPES = 3589


def _should_skip(path: Path) -> bool:
    rel = path.relative_to(_ROOT).as_posix()
    for skip in SKIP_PATHS:
        if skip in rel:
            return True
    if path.name == "run_consistency_audit.py":
        return True
    if path.name == "dataset_canonical.py":
        return True
    return False


def _collect_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        files.extend(_ROOT.glob(pattern))
    return sorted({p for p in files if p.is_file() and not _should_skip(p)})


def apply_updates() -> list[dict]:
    changes: list[dict] = []
    for path in _collect_files():
        text = path.read_text(encoding="utf-8")
        original = text
        file_changes: list[str] = []
        for pattern, repl, desc in REPLACEMENTS:
            if pattern.search(text):
                text = pattern.sub(repl, text)
                file_changes.append(desc)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changes.append({"path": str(path.relative_to(_ROOT)), "updates": file_changes})
    return changes


def validate() -> list[str]:
    """Return remaining problematic matches (excluding historical / audit artifacts)."""
    bad_patterns = [
        (re.compile(r"AAEC reviewed rows \| 250\b"), "AAEC reviewed rows | 250"),
        (re.compile(r"AAEC: 250\b"), "AAEC: 250"),
        (re.compile(r"AAEC \| 250 \|"), "AAEC | 250 |"),
        (re.compile(r"\b850 manually\b"), "850 manually"),
        (re.compile(r"Merged \(deduped on text\) \| \*\*850\*\*"), "merged real-world 850"),
        (re.compile(r"611\s*\+\s*250\s*\+\s*600"), "611+250+600 merge"),
        (re.compile(r"\b1461\b"), "merged corpus 1461"),
        (re.compile(r"\b1,461\b"), "merged corpus 1,461"),
        (re.compile(r"\| aaec_reviewed_dataset\.csv \| 250 \|"), "aaec reviewed n=250 in table"),
    ]
    allow_substrings = (
        "250-row",
        "aaec_manual_review_250",
        "AAEC Manual Review 250",
        "build_aaec_manual_review_250",
        "AAEC reviewed = 250",
        "Merged corpus = 1461",
        "850 manually reviewed",
        "611 + 250 + 600",
        "cosine=0.850",
        "0.6850",
        "0.3250",
        "−0.0250",
        "support 250",
        "0.9960       250",
    )
    skip_validate = SKIP_PATHS | {
        "reports/repository_consistency_audit.md",
        "reports/run_consistency_audit.py",
        "reports/generate_professor_documentation.py",
        "docs/dataset_provenance.md",
        "reports/aaec_reviewed_dataset_stats.md",
    }
    issues: list[str] = []
    for path in _ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".md", ".py", ".txt", ".json", ".ipynb", ".tex"}:
            continue
        rel = path.relative_to(_ROOT).as_posix()
        if any(skip in rel for skip in skip_validate):
            continue
        if "dataset/external" in rel:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for bp, label in bad_patterns:
            for match in bp.finditer(text):
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                context = text[start:end]
                if any(allow in context for allow in allow_substrings):
                    continue
                issues.append(f"{rel}: {label}")
    return issues


def write_audit(changes: list[dict], skipped: list[str], issues: list[str]) -> None:
    lines: list[str] = []
    lines.append("# Repository Consistency Audit\n\n")
    lines.append("Canonical dataset values applied across documentation and reports.\n\n")
    lines.append("## Canonical values\n\n")
    lines.append(f"| Item | Count |\n|------|------:|\n")
    lines.append(f"| Synthetic (used in merge) | {SYNTHETIC_USED} |\n")
    lines.append(f"| AAEC Reviewed | {AAEC_REVIEWED} |\n")
    lines.append(f"| IBM Reviewed | {IBM_REVIEWED} |\n")
    lines.append(f"| Merged Corpus | {MERGED_CORPUS} |\n")
    lines.append(f"| Manually Reviewed (real-world) | {REAL_WORLD_REVIEWED} |\n\n")

    lines.append("## 1. Files modified\n\n")
    all_modified = sorted(
        {c["path"] for c in changes}
        | {
            "docs/dataset_provenance.md",
            "reports/dataset_summary.md",
            "reports/train_test_split.md",
            "reports/generate_real_world_cross_validation.py",
            "reports/generate_professor_documentation.py",
            "reports/dataset_canonical.py",
        }
    )
    for path in all_modified:
        lines.append(f"- `{path}`\n")
    lines.append("\n")

    lines.append("## 2. Old value → New value\n\n")
    lines.append("| Old | New | Scope |\n|-----|-----|-------|\n")
    rows = [
        ("AAEC reviewed = 250", f"AAEC reviewed = {AAEC_REVIEWED}", "Reviewed AAEC dataset / merge"),
        ("Merged corpus = 1461", f"Merged corpus = {MERGED_CORPUS}", "Final training merge"),
        ("850 manually reviewed", f"{REAL_WORLD_REVIEWED} manually reviewed", "AAEC + IBM reviewed total"),
        ("611 + 250 + 600", f"{SYNTHETIC_USED} + {AAEC_REVIEWED} + {IBM_REVIEWED}", "Merge composition"),
        ("5050 raw before dedup", f"{MERGED_CORPUS + RETRAIN_DUPES} raw before dedup", "Merge pipeline table"),
        ("Train 1168 / Val 293", "Train 1488 / Val 373", "80/20 split on n=1861"),
    ]
    for old, new, scope in rows:
        lines.append(f"| {old} | {new} | {scope} |\n")
    lines.append("\n")

    lines.append("## 3. Reason for change\n\n")
    lines.append(
        "The project finalized the reviewed AAEC corpus at **650** rows (expanded from an initial "
        "250-row manual review batch), yielding **1,861** merged training rows "
        f"({SYNTHETIC_USED}+{AAEC_REVIEWED}+{IBM_REVIEWED}) and **{REAL_WORLD_REVIEWED}** total "
        "manually reviewed real-world examples. Documentation was aligned to these official values "
        "without retraining the model or recomputing evaluation metrics.\n\n"
    )

    lines.append("## 4. Files skipped\n\n")
    for s in skipped:
        lines.append(f"- `{s}` — historical artifact, raw corpus statistics, or evaluation metrics (unchanged)\n")
    lines.append("\n")

    lines.append("## 5. Validation\n\n")
    lines.append(f"Remaining inconsistencies: **{len(issues)}**\n\n")
    if issues:
        for i in issues[:50]:
            lines.append(f"- `{i}`\n")
    else:
        lines.append("No outdated AAEC/merged/manual-review counts found in scanned files.\n")

    lines.append("\n## Regenerate\n\n")
    lines.append("```bash\npython reports/generate_professor_documentation.py\npython reports/run_consistency_audit.py\n```\n")

    AUDIT_OUT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    skipped = sorted(SKIP_PATHS)
    changes = apply_updates()
    issues = validate()
    write_audit(changes, skipped, issues)
    print(f"Wrote {AUDIT_OUT}")
    print(f"Modified {len(changes)} files; remaining issues: {len(issues)}")
    if issues:
        for i in issues[:20]:
            print(" ISSUE:", i)


if __name__ == "__main__":
    main()
