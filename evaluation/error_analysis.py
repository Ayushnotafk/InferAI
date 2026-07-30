"""Error analysis for InferAI neuro-symbolic predictions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from classification.adaptive_router import DEFAULT_ALPHA
from evaluation.benchmark import predict_with_alpha

SHORT_TOKENS = 12
LONG_TOKENS = 40

CLUSTER_RECOMMENDATIONS = {
    "rule_failure": (
        "Expand or reweight symbolic cues for the confused class; audit false-positive "
        "patterns (e.g. informal 'according to', bare 'like')."
    ),
    "embedding_failure": (
        "Collect more minority-class training examples; consider class-balanced fine-tuning "
        "of the logistic head on hard embedding neighborhoods."
    ),
    "ambiguous_argument": (
        "Add dual-annotator review for boundary cases; optionally abstain or surface "
        "multi-label uncertainty when ML and rules disagree with low confidence."
    ),
    "short_argument": (
        "Short spans lack cues — prefer ML or request more context; add short-form "
        "templates to the training corpus."
    ),
    "long_argument": (
        "Long multi-claim texts mix pramāṇas — improve claim/premise segmentation before "
        "classification, or classify claim-local windows."
    ),
}


def _token_len(text: str) -> int:
    return len(str(text).split())


def _assign_cluster(row: pd.Series) -> str:
    """Heuristic failure cluster (mutually preferential order)."""
    n = _token_len(row["text"])
    if n <= SHORT_TOKENS:
        return "short_argument"
    if n >= LONG_TOKENS:
        return "long_argument"
    low_conf = float(row["ml_confidence"]) < 55.0
    disagree = (
        row["rule_label"] is not None
        and row["ml_label"] != row["rule_label"]
        and low_conf
    )
    if disagree:
        return "ambiguous_argument"
    if row["rule_correct"] is False and row["rule_label"] is not None:
        return "rule_failure"
    if not bool(row["ml_correct"]):
        return "embedding_failure"
    return "ambiguous_argument"


def run_error_analysis(
    test_csv: str | Path,
    *,
    alpha: float | None = None,
    adaptive: bool = False,
    out_dir: str | Path | None = None,
    low_confidence_threshold: float = 55.0,
) -> dict[str, Any]:
    """
    Identify confused pairs, low-confidence errors, ML vs symbolic failures,
    and clustered failure modes with recommendations.
    """
    df = pd.read_csv(test_csv)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    texts = df["text"].astype(str).tolist()
    y_true = df[label_col].astype(str).tolist()

    rows: list[dict[str, Any]] = []
    for text, yt in zip(texts, y_true):
        pred, ml_conf, meta = predict_with_alpha(
            text,
            alpha=None if adaptive else (alpha if alpha is not None else DEFAULT_ALPHA),
            adaptive=adaptive,
        )
        ml_label = meta.get("ml_label")
        rule_label = meta.get("rule_label")
        rows.append(
            {
                "text": text,
                "y_true": yt,
                "y_pred": pred,
                "correct": pred == yt,
                "ml_label": ml_label,
                "rule_label": rule_label,
                "ml_correct": ml_label == yt,
                "rule_correct": (rule_label == yt) if rule_label else None,
                "ml_confidence": ml_conf,
                "adjusted_confidence": meta.get("adjusted_confidence"),
                "alpha": meta.get("alpha"),
                "routing_reason": meta.get("routing_reason"),
                "n_tokens": _token_len(text),
            }
        )

    rdf = pd.DataFrame(rows)
    errors = rdf[~rdf["correct"]].copy()
    if len(errors):
        errors["cluster"] = errors.apply(_assign_cluster, axis=1)
    else:
        errors["cluster"] = pd.Series(dtype=str)

    confusion_pairs = Counter(
        (r.y_true, r.y_pred) for r in errors.itertuples(index=False)
    )
    top_confused = [
        {"true": a, "pred": b, "count": c}
        for (a, b), c in confusion_pairs.most_common(10)
    ]

    low_conf_errors = errors[errors["ml_confidence"] < low_confidence_threshold]
    fusion_hurt = rdf[(rdf["ml_correct"]) & (~rdf["correct"])]
    rule_rescues = rdf[(~rdf["ml_correct"]) & (rdf["correct"])]
    ml_failures = rdf[(~rdf["ml_correct"]) & (~rdf["correct"])]
    symbolic_failures = rdf[
        rdf["rule_label"].notna()
        & (rdf["rule_correct"] == False)  # noqa: E712
        & (~rdf["correct"])
    ]

    cluster_counts = errors["cluster"].value_counts().to_dict() if len(errors) else {}
    cluster_recs = [
        {
            "cluster": c,
            "count": int(n),
            "recommendation": CLUSTER_RECOMMENDATIONS.get(c, ""),
        }
        for c, n in sorted(cluster_counts.items(), key=lambda x: -x[1])
    ]

    summary: dict[str, Any] = {
        "n_samples": len(rdf),
        "n_errors": int((~rdf["correct"]).sum()),
        "accuracy": float(rdf["correct"].mean()),
        "top_confused_classes": top_confused,
        "n_low_confidence_errors": int(len(low_conf_errors)),
        "n_fusion_hurt_ml_correct": int(len(fusion_hurt)),
        "n_rule_rescues": int(len(rule_rescues)),
        "n_ml_failures": int(len(ml_failures)),
        "n_symbolic_failures": int(len(symbolic_failures)),
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "cluster_recommendations": cluster_recs,
        "narrative": _narrative(
            top_confused, fusion_hurt, rule_rescues, ml_failures, symbolic_failures
        ),
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rdf.to_csv(out / "error_analysis_all_predictions.csv", index=False)
        errors.to_csv(out / "error_analysis_errors.csv", index=False)
        low_conf_errors.to_csv(out / "error_analysis_low_confidence.csv", index=False)
        fusion_hurt.to_csv(out / "error_analysis_fusion_hurt.csv", index=False)
        rule_rescues.to_csv(out / "error_analysis_rule_rescues.csv", index=False)
        ml_failures.to_csv(out / "error_analysis_ml_failures.csv", index=False)
        symbolic_failures.to_csv(out / "error_analysis_symbolic_failures.csv", index=False)
        pd.DataFrame(top_confused).to_csv(out / "error_analysis_confused_pairs.csv", index=False)
        pd.DataFrame(cluster_recs).to_csv(out / "error_analysis_clusters.csv", index=False)
        for cluster in cluster_counts:
            errors[errors["cluster"] == cluster].to_csv(
                out / f"error_cluster_{cluster}.csv", index=False
            )
        (out / "error_analysis_report.md").write_text(
            _to_markdown(summary), encoding="utf-8"
        )
    return summary


def _narrative(
    top_confused: list[dict[str, Any]],
    fusion_hurt: pd.DataFrame,
    rule_rescues: pd.DataFrame,
    ml_failures: pd.DataFrame,
    symbolic_failures: pd.DataFrame,
) -> str:
    parts = []
    if top_confused:
        top = top_confused[0]
        parts.append(
            f"Most frequent confusion: {top['true']} → {top['pred']} ({top['count']} cases)."
        )
    parts.append(
        f"Rules rescued {len(rule_rescues)} ML errors; fusion overrode {len(fusion_hurt)} "
        f"correct ML decisions; {len(ml_failures)} residual ML failures; "
        f"{len(symbolic_failures)} symbolic-aligned residual errors."
    )
    return " ".join(parts)


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Error Analysis Report",
        "",
        f"- Samples: {summary['n_samples']}",
        f"- Errors: {summary['n_errors']}",
        f"- Accuracy: {summary['accuracy']:.4f}",
        f"- Low-confidence errors: {summary['n_low_confidence_errors']}",
        f"- Rule rescues: {summary['n_rule_rescues']}",
        f"- Fusion hurt (ML was correct): {summary['n_fusion_hurt_ml_correct']}",
        f"- Residual ML failures: {summary['n_ml_failures']}",
        f"- Symbolic failures: {summary['n_symbolic_failures']}",
        "",
        "## Top confused class pairs",
        "",
        "| True | Predicted | Count |",
        "|------|-----------|-------|",
    ]
    for row in summary["top_confused_classes"]:
        lines.append(f"| {row['true']} | {row['pred']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Failure clusters",
            "",
            "| Cluster | Count | Recommendation |",
            "|---------|-------|----------------|",
        ]
    )
    for row in summary.get("cluster_recommendations", []):
        lines.append(
            f"| {row['cluster']} | {row['count']} | {row['recommendation']} |"
        )
    lines.extend(["", "## Narrative", "", summary["narrative"], ""])
    return "\n".join(lines)
