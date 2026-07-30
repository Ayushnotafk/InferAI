#!/usr/bin/env python3
"""
Generate a publication-ready InferAI research report bundle.

Writes CSV / PNG / PDF-ready figures, significance tests, and Markdown/LaTeX
tables under ``results/report/``.

Usage:
    python scripts/generate_research_report.py
    python scripts/generate_research_report.py --skip-slow
    python scripts/generate_research_report.py --include-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.adaptive_router import DEFAULT_ALPHA
from evaluation.adaptive_analysis import analyze_adaptive_routing
from evaluation.benchmark import run_ablation_study, run_benchmark
from evaluation.calibration import run_calibration_study
from evaluation.cross_validation import run_cross_validation
from evaluation.dataset_statistics import summarize_splits
from evaluation.error_analysis import run_error_analysis
from evaluation.explanation_evaluation import (
    export_evaluation_sheet,
    export_printable_form,
)
from evaluation.fallacy_evaluation import evaluate_fallacy_detector
from evaluation.freeze import freeze_current_artifacts
from evaluation.llm_baseline import available_providers, run_llm_benchmark
from evaluation.ood_evaluation import run_ood_evaluation
from evaluation.publication_figures import (
    plot_metric_vs_alpha_pub,
    stitch_figures_pdf,
)
from evaluation.publication_tables import export_publication_tables
from evaluation.robustness import run_robustness_evaluation
from evaluation.shap_analysis import run_shap_publication_analysis
from evaluation.statistical_tests import run_significance_suite


def _write_mode_table(modes: dict, out: Path) -> None:
    rows = []
    for mode, m in modes.items():
        rows.append(
            {
                "mode": mode,
                "accuracy": m.get("accuracy"),
                "macro_precision": m.get("macro_precision"),
                "macro_recall": m.get("macro_recall"),
                "macro_f1": m.get("macro_f1"),
                "weighted_f1": m.get("weighted_f1"),
                "ece": m.get("ece"),
                "mean_alpha": m.get("mean_alpha"),
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "benchmark_comparison_table.csv", index=False)


def _markdown_report(payload: dict, out: Path) -> None:
    modes = payload.get("modes", {})
    abl = payload.get("ablation", [])
    lines = [
        "# InferAI Research Report",
        "",
        "Auto-generated publication bundle. See `reports/alpha_investigation.md` "
        "and `statistical_significance.md` for statistical analysis.",
        "",
        f"**Default fixed hybrid α:** {DEFAULT_ALPHA}",
        "",
        "## Benchmark comparison",
        "",
        "| Mode | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | ECE | Mean α |",
        "|------|----------|---------|---------|----------|-------------|-----|--------|",
    ]
    for mode, m in modes.items():
        lines.append(
            "| {mode} | {acc:.3f} | {mp} | {mr} | {mf:.3f} | {wf} | {ece} | {a} |".format(
                mode=mode,
                acc=float(m.get("accuracy") or 0),
                mp=f"{m['macro_precision']:.3f}" if m.get("macro_precision") is not None else "—",
                mr=f"{m['macro_recall']:.3f}" if m.get("macro_recall") is not None else "—",
                mf=float(m.get("macro_f1") or 0),
                wf=f"{m['weighted_f1']:.3f}" if m.get("weighted_f1") is not None else "—",
                ece=f"{m['ece']:.3f}" if m.get("ece") is not None else "—",
                a=f"{m['mean_alpha']:.3f}" if m.get("mean_alpha") is not None else "—",
            )
        )

    if abl:
        lines.extend(
            [
                "",
                "## Alpha ablation",
                "",
                "| α | Accuracy | Macro F1 | ECE |",
                "|---|----------|----------|-----|",
            ]
        )
        for row in abl:
            lines.append(
                f"| {row['alpha']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | {row['ece']:.3f} |"
            )

    sig = payload.get("statistical_significance") or {}
    if sig.get("comparisons"):
        lines.extend(
            [
                "",
                "## Statistical significance (McNemar)",
                "",
                "| Comparison | p-value | Significant |",
                "|------------|---------|-------------|",
            ]
        )
        for row in sig["comparisons"]:
            lines.append(
                f"| {row['comparison']} | {row['p_value']:.4f} | {row['significant']} |"
            )

    cv = payload.get("cross_validation") or {}
    if cv:
        lines.extend(
            [
                "",
                "## Cross-validation",
                "",
                f"- Mean accuracy: {cv.get('mean_accuracy'):.4f} ± {cv.get('std_accuracy'):.4f}",
                f"- Mean macro F1: {cv.get('mean_macro_f1'):.4f} ± {cv.get('std_macro_f1'):.4f}",
            ]
        )

    cal = payload.get("calibration") or {}
    if cal:
        lines.extend(["", "## Calibration", "", cal.get("recommendation", "")])

    adaptive = payload.get("adaptive_analysis") or {}
    if adaptive:
        lines.extend(
            [
                "",
                "## Adaptive routing",
                "",
                f"- Mean α: {adaptive.get('mean_alpha')}",
                f"- Min / Max: {adaptive.get('min_alpha')} / {adaptive.get('max_alpha')}",
                adaptive.get("explanation", ""),
            ]
        )

    for key, title in (
        ("error_analysis", "Error analysis"),
        ("robustness", "Robustness"),
        ("ood", "OOD evaluation"),
        ("fallacy_evaluation", "Fallacy detector"),
    ):
        block = payload.get(key) or {}
        if not block:
            continue
        lines.extend(["", f"## {title}", ""])
        if key == "robustness":
            lines.append(
                f"Clean acc={block.get('clean_accuracy'):.3f}; "
                f"worst={block.get('worst_perturbation')}; "
                f"mean drop={block.get('mean_drop'):.3f}"
            )
        elif key == "ood":
            lines.append(block.get("narrative", ""))
        elif key == "fallacy_evaluation" and "error" not in block:
            lines.append(
                f"Binary P/R={block.get('binary_precision'):.3f}/"
                f"{block.get('binary_recall'):.3f}; "
                f"FP/FN={block.get('false_positives')}/{block.get('false_negatives')}"
            )
        elif key == "error_analysis":
            lines.append(block.get("narrative", ""))

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Tables: `tables/all_tables.md`, `tables/all_tables.tex`",
            "- Significance: `statistical_significance.csv`",
            "- Figures: PNG+PDF at 300 DPI; bundle `figures_bundle.pdf`",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/freeze_model.py --tag research",
            "python scripts/generate_research_report.py",
            "```",
            "",
        ]
    )
    (out / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="InferAI publication report generator")
    parser.add_argument("--test-csv", default=str(_ROOT / "dataset" / "test_set.csv"))
    parser.add_argument("--out", default=str(_ROOT / "results" / "report"))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--include-llm", action="store_true")
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip full mode sweep/ablation/CV/calibration/robustness/OOD/SHAP",
    )
    parser.add_argument("--max-llm-samples", type=int, default=None)
    parser.add_argument(
        "--quick-stress",
        action="store_true",
        help="Limit robustness/OOD/SHAP sample counts for a faster pass",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.freeze:
        print(f"Frozen: {freeze_current_artifacts(tag='research_report')}")

    payload: dict = {"default_alpha": DEFAULT_ALPHA, "modes": {}, "ablation": []}
    stress_n = 40 if args.quick_stress else None

    if not args.skip_slow:
        for mode in ("ml", "symbolic", "hybrid", "adaptive"):
            print(f"[benchmark] {mode}")
            payload["modes"][mode] = run_benchmark(
                args.test_csv, mode=mode, out_dir=out  # type: ignore[arg-type]
            )

        print("[ablation] alpha sweep")
        abl_df = run_ablation_study(args.test_csv, out_dir=out)
        payload["ablation"] = abl_df.to_dict(orient="records")
        for metric, ylabel in (
            ("accuracy", "Accuracy"),
            ("macro_f1", "Macro F1"),
            ("ece", "Expected Calibration Error"),
        ):
            plot_metric_vs_alpha_pub(
                abl_df, metric, fig_dir / f"{metric}_vs_alpha.png", ylabel=ylabel
            )

        print("[statistical significance]")
        payload["statistical_significance"] = run_significance_suite(
            args.test_csv, out_dir=out
        )

        print("[cross-validation]")
        payload["cross_validation"] = run_cross_validation(out_dir=out)

        print("[calibration]")
        payload["calibration"] = run_calibration_study(
            test_csv=args.test_csv, out_dir=out
        )

        print("[robustness]")
        payload["robustness"] = run_robustness_evaluation(
            args.test_csv, out_dir=out, max_samples=stress_n
        )

        print("[ood]")
        payload["ood"] = run_ood_evaluation(
            id_csv=args.test_csv, out_dir=out, max_samples=stress_n
        )

        print("[shap]")
        payload["shap"] = run_shap_publication_analysis(
            args.test_csv, out_dir=out, max_samples=stress_n or 80
        )
    else:
        abl_path = _ROOT / "results" / "benchmarks" / "ablation_alpha_sweep.csv"
        if abl_path.is_file():
            import pandas as pd

            abl_df = pd.read_csv(abl_path)
            payload["ablation"] = abl_df.to_dict(orient="records")
            for metric, ylabel in (
                ("accuracy", "Accuracy"),
                ("macro_f1", "Macro F1"),
                ("ece", "Expected Calibration Error"),
            ):
                if metric in abl_df.columns:
                    plot_metric_vs_alpha_pub(
                        abl_df, metric, fig_dir / f"{metric}_vs_alpha.png", ylabel=ylabel
                    )

    if args.include_llm:
        for provider in available_providers():
            print(f"[llm] {provider}")
            result = run_llm_benchmark(
                args.test_csv,
                provider=provider,
                out_dir=out,
                max_samples=args.max_llm_samples,
            )
            payload["modes"][result["mode"]] = result

    print("[adaptive analysis]")
    payload["adaptive_analysis"] = analyze_adaptive_routing(args.test_csv, out_dir=out)

    print("[error analysis]")
    payload["error_analysis"] = run_error_analysis(
        args.test_csv, alpha=DEFAULT_ALPHA, adaptive=False, out_dir=out
    )

    print("[fallacy evaluation]")
    payload["fallacy_evaluation"] = evaluate_fallacy_detector(out_dir=out)

    print("[dataset statistics]")
    payload["dataset_statistics"] = summarize_splits(out_dir=out)

    demo_samples = [
        {
            "text": "Smoke rises from the hill; therefore there is fire.",
            "predicted_label": "Anumana",
            "explanation": "Inferential cues (therefore) support Anumana.",
        },
        {
            "text": "According to WHO, vaccination reduces disease spread.",
            "predicted_label": "Shabda",
            "explanation": "Authoritative citation supports Shabda.",
        },
    ]
    export_evaluation_sheet(demo_samples, out / "explainability_eval_sheet.csv")
    export_printable_form(demo_samples, out / "explainability_printable_form.md")

    print("[publication tables]")
    export_publication_tables(
        out_dir=out,
        dataset_stats=payload.get("dataset_statistics"),
        benchmark_modes=payload.get("modes"),
        ablation=payload.get("ablation"),
        adaptive=payload.get("adaptive_analysis"),
        fallacy=payload.get("fallacy_evaluation"),
        explainability=None,
        confidence_intervals=(
            (payload.get("statistical_significance") or {}).get("confidence_intervals")
        ),
    )

    _write_mode_table(payload["modes"], out)
    _markdown_report(payload, out)

    figure_candidates = list(fig_dir.glob("*.png")) + list(out.glob("*.png"))
    if figure_candidates:
        stitch_figures_pdf(figure_candidates, out / "figures_bundle.pdf")

    (out / "research_report_payload.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote research report to {out}")


if __name__ == "__main__":
    main()
