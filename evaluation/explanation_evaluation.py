"""
Human explainability evaluation sheet export and summarization.

Likert dimensions (1–5): clarity, trust, plausibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.publication_figures import plot_likert_bars

LIKERT_COLUMNS = ("clarity", "trust", "plausibility")
LIKERT_MIN, LIKERT_MAX = 1, 5


def export_evaluation_sheet(
    samples: list[dict[str, str]],
    out_path: str | Path,
) -> Path:
    """
    Export a human evaluation template (CSV).

    Each sample should include at least ``text`` and ``explanation`` keys.
    """
    rows = []
    for i, s in enumerate(samples):
        rows.append(
            {
                "sample_id": i + 1,
                "text": s.get("text", ""),
                "predicted_label": s.get("predicted_label", ""),
                "explanation": s.get("explanation", ""),
                "clarity": "",
                "trust": "",
                "plausibility": "",
                "reviewer_notes": "",
            }
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def export_printable_form(
    samples: list[dict[str, str]],
    out_path: str | Path,
) -> Path:
    """Export a Markdown form suitable for printing / HITL annotation sessions."""
    lines = [
        "# InferAI Explainability Evaluation Form",
        "",
        "Rate each explanation on a **1–5** Likert scale:",
        "",
        "| Score | Meaning |",
        "|------:|---------|",
        "| 1 | Very poor |",
        "| 2 | Poor |",
        "| 3 | Acceptable |",
        "| 4 | Good |",
        "| 5 | Excellent |",
        "",
        "Dimensions: **Clarity**, **Trust**, **Plausibility**.",
        "",
        "---",
        "",
    ]
    for i, s in enumerate(samples, start=1):
        lines.extend(
            [
                f"## Sample {i}",
                "",
                f"**Text:** {s.get('text', '')}",
                "",
                f"**Predicted label:** {s.get('predicted_label', '')}",
                "",
                f"**Explanation:** {s.get('explanation', '')}",
                "",
                "- Clarity (1–5): ____",
                "- Trust (1–5): ____",
                "- Plausibility (1–5): ____",
                "- Notes: ____",
                "",
                "---",
                "",
            ]
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def summarize_completed_sheet(
    completed_csv: str | Path,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compute mean/std for each Likert dimension; optionally export tables/plots."""
    df = pd.read_csv(completed_csv)
    summary: dict[str, Any] = {"n": len(df), "dimensions": {}}
    table_rows = []
    for col in LIKERT_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        dim = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": int(vals.min()),
            "max": int(vals.max()),
            "n_rated": int(len(vals)),
        }
        summary["dimensions"][col] = dim
        table_rows.append({"dimension": col, **dim})

    if out_dir and table_rows:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table_rows).to_csv(out / "explainability_likert_summary.csv", index=False)
        plot_likert_bars(summary, out / "explainability_likert_bars.png")
        lines = [
            "# Explainability Evaluation Summary",
            "",
            f"Completed ratings: {summary['n']}",
            "",
            "| Dimension | Mean | Std | Min | Max | N |",
            "|-----------|------|-----|-----|-----|---|",
        ]
        for row in table_rows:
            lines.append(
                f"| {row['dimension']} | {row['mean']:.2f} | {row['std']:.2f} | "
                f"{row['min']} | {row['max']} | {row['n_rated']} |"
            )
        (out / "explainability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


# Backward-compatible alias.
summarize_likert_responses = summarize_completed_sheet
