"""
Publication-ready tables (Markdown + LaTeX) for InferAI research reports.

Tables:
  1 Dataset statistics
  2 Benchmark comparison
  3 Ablation study
  4 Adaptive routing statistics
  5 Fallacy detection
  6 Explainability evaluation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _esc_latex(s: Any) -> str:
    text = str(s)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(a, b)
    return text


def df_to_markdown(df: pd.DataFrame, caption: str) -> str:
    lines = [f"### {caption}", ""]
    if df.empty:
        lines.append("_No data._")
        lines.append("")
        return "\n".join(lines)
    headers = list(df.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[h]) for h in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    col_spec = "l" + "c" * (len(cols) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_esc_latex(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(_esc_latex(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_esc_latex(_fmt(row[c])) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def export_publication_tables(
    *,
    out_dir: str | Path,
    dataset_stats: dict[str, Any] | None = None,
    benchmark_modes: dict[str, Any] | None = None,
    ablation: list[dict[str, Any]] | None = None,
    adaptive: dict[str, Any] | None = None,
    fallacy: dict[str, Any] | None = None,
    explainability: dict[str, Any] | None = None,
    confidence_intervals: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write Table 1–6 as Markdown and LaTeX under ``out_dir/tables``."""
    out = Path(out_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # Table 1 — dataset statistics
    t1_rows = []
    if dataset_stats:
        for split, st in dataset_stats.items():
            if not isinstance(st, dict):
                continue
            t1_rows.append(
                {
                    "Split": split,
                    "N": st.get("n_rows"),
                    "Vocab": st.get("vocabulary_size"),
                    "Avg tokens": st.get("avg_sentence_tokens"),
                    "TTR": st.get("ttr"),
                    "Guiraud R": st.get("guiraud_r"),
                }
            )
    t1 = pd.DataFrame(t1_rows)
    written["table1"] = _write_pair(out, "table1_dataset_statistics", t1, "Dataset statistics")

    # Table 2 — benchmark (+ optional CIs merged as strings)
    t2_rows = []
    ci_map: dict[tuple[str, str], dict] = {}
    for row in confidence_intervals or []:
        ci_map[(row["mode"], row["metric"])] = row
    for mode, m in (benchmark_modes or {}).items():
        acc = m.get("accuracy")
        mf1 = m.get("macro_f1")
        acc_ci = ci_map.get((mode, "accuracy"))
        f1_ci = ci_map.get((mode, "macro_f1"))
        t2_rows.append(
            {
                "Mode": mode,
                "Accuracy": acc,
                "Acc 95% CI": (
                    f"[{acc_ci['ci_low']:.3f}, {acc_ci['ci_high']:.3f}]" if acc_ci else "—"
                ),
                "Macro F1": mf1,
                "F1 95% CI": (
                    f"[{f1_ci['ci_low']:.3f}, {f1_ci['ci_high']:.3f}]" if f1_ci else "—"
                ),
                "ECE": m.get("ece"),
            }
        )
    t2 = pd.DataFrame(t2_rows)
    written["table2"] = _write_pair(out, "table2_benchmark_comparison", t2, "Benchmark comparison")

    # Table 3 — ablation
    t3 = pd.DataFrame(ablation or [])
    if not t3.empty:
        keep = [c for c in ("alpha", "accuracy", "macro_f1", "ece") if c in t3.columns]
        t3 = t3[keep]
    written["table3"] = _write_pair(out, "table3_ablation_study", t3, "Alpha ablation study")

    # Table 4 — adaptive routing
    t4 = pd.DataFrame(
        [
            {
                "Mean alpha": (adaptive or {}).get("mean_alpha"),
                "Min alpha": (adaptive or {}).get("min_alpha"),
                "Max alpha": (adaptive or {}).get("max_alpha"),
                "Std alpha": (adaptive or {}).get("std_alpha"),
                "% ML dominated": (adaptive or {}).get("pct_ml_dominated"),
                "% Rules dominated": (adaptive or {}).get("pct_rules_dominated"),
                "% Balanced": (adaptive or {}).get("pct_balanced"),
                "Accuracy": (adaptive or {}).get("accuracy"),
            }
        ]
    ) if adaptive else pd.DataFrame()
    written["table4"] = _write_pair(
        out, "table4_adaptive_routing", t4, "Adaptive routing statistics"
    )

    # Table 5 — fallacy
    t5 = pd.DataFrame(
        [
            {
                "N": (fallacy or {}).get("n_samples"),
                "Accuracy": (fallacy or {}).get("accuracy"),
                "Binary P": (fallacy or {}).get("binary_precision"),
                "Binary R": (fallacy or {}).get("binary_recall"),
                "FP": (fallacy or {}).get("false_positives"),
                "FN": (fallacy or {}).get("false_negatives"),
            }
        ]
    ) if fallacy and "error" not in (fallacy or {}) else pd.DataFrame()
    written["table5"] = _write_pair(out, "table5_fallacy_detection", t5, "Fallacy detection")

    # Table 6 — explainability
    t6_rows = []
    for dim, stats in ((explainability or {}).get("dimensions") or {}).items():
        t6_rows.append(
            {
                "Dimension": dim,
                "Mean": stats.get("mean"),
                "Std": stats.get("std"),
                "Min": stats.get("min"),
                "Max": stats.get("max"),
                "N": stats.get("n_rated"),
            }
        )
    t6 = pd.DataFrame(t6_rows)
    written["table6"] = _write_pair(
        out, "table6_explainability", t6, "Explainability evaluation (Likert 1–5)"
    )

    # Combined files
    stems = {
        "table1": "table1_dataset_statistics",
        "table2": "table2_benchmark_comparison",
        "table3": "table3_ablation_study",
        "table4": "table4_adaptive_routing",
        "table5": "table5_fallacy_detection",
        "table6": "table6_explainability",
    }
    md_all = ["# Publication Tables", ""]
    tex_all = [
        "% Auto-generated InferAI tables (requires booktabs)",
        r"\usepackage{booktabs}",
        "",
    ]
    for key, stem in stems.items():
        md_path = out / f"{stem}.md"
        tex_path = out / f"{stem}.tex"
        if md_path.is_file():
            md_all.append(md_path.read_text(encoding="utf-8"))
        if tex_path.is_file():
            tex_all.append(tex_path.read_text(encoding="utf-8"))
    (out / "all_tables.md").write_text("\n".join(md_all), encoding="utf-8")
    (out / "all_tables.tex").write_text("\n".join(tex_all), encoding="utf-8")
    return written


def _write_pair(
    out: Path, stem: str, df: pd.DataFrame, caption: str
) -> Path:
    label = stem.replace("_", ":")
    md = df_to_markdown(df, caption)
    tex = df_to_latex(df, caption, label=f"tab:{stem}")
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    (out / f"{stem}.tex").write_text(tex, encoding="utf-8")
    if not df.empty:
        df.to_csv(out / f"{stem}.csv", index=False)
    return out / f"{stem}.md"
