"""Adaptive routing analysis for InferAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from classification.adaptive_router import DEFAULT_ALPHA
from evaluation.benchmark import predict_with_alpha
from evaluation.publication_figures import plot_alpha_distribution, plot_categorical_counts

# Bucket thresholds around the ablation-optimal base.
ML_DOMINATED = 0.55
RULES_DOMINATED = 0.35


def analyze_adaptive_routing(
    test_csv: str | Path,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Collect per-sample adaptive alphas and routing statistics.

    Buckets:
      - ML dominated: alpha >= 0.55
      - Rules dominated: alpha <= 0.35
      - Balanced: otherwise
    """
    df = pd.read_csv(test_csv)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    texts = df["text"].astype(str).tolist()
    y_true = df[label_col].astype(str).tolist()

    rows: list[dict[str, Any]] = []
    for text, yt in zip(texts, y_true):
        pred, ml_conf, meta = predict_with_alpha(text, adaptive=True)
        alpha = float(meta.get("alpha", DEFAULT_ALPHA))
        if alpha >= ML_DOMINATED:
            bucket = "ml_dominated"
        elif alpha <= RULES_DOMINATED:
            bucket = "rules_dominated"
        else:
            bucket = "balanced"
        rows.append(
            {
                "text": text,
                "y_true": yt,
                "y_pred": pred,
                "correct": pred == yt,
                "ml_confidence": ml_conf,
                "adaptive_alpha": alpha,
                "routing_mode": meta.get("routing_mode"),
                "routing_reason": meta.get("routing_reason"),
                "ml_label": meta.get("ml_label"),
                "rule_label": meta.get("rule_label"),
                "bucket": bucket,
            }
        )

    rdf = pd.DataFrame(rows)
    alphas = rdf["adaptive_alpha"].tolist()
    bucket_counts = rdf["bucket"].value_counts().to_dict()
    n = len(rdf)

    summary: dict[str, Any] = {
        "n_samples": n,
        "mean_alpha": float(np.mean(alphas)),
        "min_alpha": float(np.min(alphas)),
        "max_alpha": float(np.max(alphas)),
        "std_alpha": float(np.std(alphas)),
        "median_alpha": float(np.median(alphas)),
        "pct_ml_dominated": 100.0 * bucket_counts.get("ml_dominated", 0) / n,
        "pct_rules_dominated": 100.0 * bucket_counts.get("rules_dominated", 0) / n,
        "pct_balanced": 100.0 * bucket_counts.get("balanced", 0) / n,
        "accuracy": float(rdf["correct"].mean()),
        "bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
        "accuracy_by_bucket": {
            b: float(rdf.loc[rdf["bucket"] == b, "correct"].mean())
            for b in sorted(bucket_counts)
        },
        "explanation": (
            "Adaptive routing improves over fixed high-ML hybrids by lowering alpha "
            "when symbolic cues are present (recovering rule-correct / ML-wrong cases) "
            "and raising alpha when cues are absent so ML remains the decision maker "
            f"under the soft rule floor. Empirical mean alpha={float(np.mean(alphas)):.3f} "
            f"(base={DEFAULT_ALPHA})."
        ),
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rdf.to_csv(out / "adaptive_routing_samples.csv", index=False)
        pd.DataFrame([summary]).to_csv(out / "adaptive_routing_summary.csv", index=False)
        plot_alpha_distribution(alphas, out / "adaptive_alpha_distribution.png")
        plot_categorical_counts(
            summary["bucket_counts"],
            out / "routing_decision_distribution.png",
            title="Routing decision distribution",
            xlabel="Routing bucket",
        )
        (out / "adaptive_routing_analysis.md").write_text(
            _to_markdown(summary), encoding="utf-8"
        )
    return summary


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Adaptive Routing Analysis",
        "",
        f"- Samples: {summary['n_samples']}",
        f"- Mean α: {summary['mean_alpha']:.4f}",
        f"- Min / Max α: {summary['min_alpha']:.4f} / {summary['max_alpha']:.4f}",
        f"- Std α: {summary['std_alpha']:.4f}",
        f"- Accuracy: {summary['accuracy']:.4f}",
        "",
        "## Routing buckets",
        "",
        f"| Bucket | % | Accuracy |",
        f"|--------|---|----------|",
    ]
    for b, pct_key in (
        ("ml_dominated", "pct_ml_dominated"),
        ("rules_dominated", "pct_rules_dominated"),
        ("balanced", "pct_balanced"),
    ):
        acc = summary["accuracy_by_bucket"].get(b, float("nan"))
        lines.append(f"| {b} | {summary[pct_key]:.1f}% | {acc:.3f} |")
    lines.extend(["", "## Interpretation", "", summary["explanation"], ""])
    return "\n".join(lines)
