#!/usr/bin/env python3
"""
Run InferAI research benchmarks end-to-end.

Usage (from project root):
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --freeze
    python scripts/run_benchmarks.py --test-csv dataset/test_set.csv --out results/benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.benchmark import run_ablation_study, run_benchmark
from evaluation.freeze import freeze_current_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="InferAI benchmark runner")
    parser.add_argument(
        "--test-csv",
        default=str(_ROOT / "dataset" / "test_set.csv"),
        help="Labeled evaluation CSV (text + pramana_label or label)",
    )
    parser.add_argument(
        "--out",
        default=str(_ROOT / "results" / "benchmarks"),
        help="Output directory for CSV, plots, and reports",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze current model + rules before benchmarking",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip alpha ablation sweep",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Optionally evaluate GPT-4 / Gemini when API keys are set",
    )
    parser.add_argument("--max-llm-samples", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.freeze:
        freeze_dir = freeze_current_artifacts(tag="benchmark")
        print(f"Frozen artifacts: {freeze_dir}")

    summary: dict = {"modes": {}, "ablation_csv": None}

    for mode in ("ml", "symbolic", "hybrid", "adaptive"):
        print(f"Running benchmark: {mode}")
        result = run_benchmark(
            args.test_csv,
            mode=mode,  # type: ignore[arg-type]
            out_dir=out,
        )
        summary["modes"][mode] = {
            k: result[k]
            for k in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "ece",
                "mean_alpha",
            )
            if k in result
        }

    if args.include_llm:
        from evaluation.llm_baseline import available_providers, run_llm_benchmark

        for provider in available_providers():
            print(f"Running LLM benchmark: {provider}")
            result = run_llm_benchmark(
                args.test_csv,
                provider=provider,
                out_dir=out,
                max_samples=args.max_llm_samples,
            )
            summary["modes"][result["mode"]] = {
                k: result[k]
                for k in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")
                if k in result
            }

    if not args.skip_ablation:
        print("Running alpha ablation study...")
        df = run_ablation_study(args.test_csv, out_dir=out)
        summary["ablation_csv"] = str(out / "ablation_alpha_sweep.csv")
        summary["ablation"] = df.to_dict(orient="records")

    report_path = out / "benchmark_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
