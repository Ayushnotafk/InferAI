"""
InferAI benchmarking: pure ML, pure symbolic, hybrid, and adaptive routing.

Alpha convention
----------------
``alpha`` is the ML weight in hybrid fusion:
  - alpha = 1.0  → pure ML (Sentence-BERT + logistic regression)
  - alpha = 0.0  → pure symbolic rules
  - 0 < alpha < 1 → hybrid neuro-symbolic fusion
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from classification.adaptive_router import DEFAULT_ALPHA, resolve_alpha
from classification.hybrid_reasoning import hybrid_fuse, score_weighted_rules
from classification.predictor import predict_pramana_detailed
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import compute_classification_metrics, expected_calibration_error
from evaluation.plots import plot_ablation_summary

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST = _ROOT / "dataset" / "test_set.csv"

ABLATION_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _load_test_csv(path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    texts = df["text"].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    return texts, labels


def predict_with_alpha(
    text: str,
    *,
    alpha: float | None = None,
    adaptive: bool = False,
) -> tuple[str, float, dict[str, Any]]:
    """
    Run inference with explicit or adaptive alpha.

    Returns (predicted_label, ml_confidence_pct, hybrid_metadata).
    """
    detail = predict_pramana_detailed(text)
    proba = detail["probabilities"]
    classes = detail["classes"]
    ml_conf = float(detail["ml_confidence"])

    raw_scores, matched = score_weighted_rules(text)
    resolved_alpha, mode, reason = resolve_alpha(
        ml_confidence_pct=ml_conf,
        matched_cues=matched,
        rule_scores=raw_scores,
        fixed_alpha=alpha,
        adaptive=adaptive,
    )

    hybrid = hybrid_fuse(
        proba,
        text,
        class_order=classes,
        alpha=resolved_alpha,
        routing_mode=mode,
        routing_reason=reason,
    )
    return hybrid["final_label"], ml_conf, hybrid


def run_benchmark(
    test_csv: str | Path | None = None,
    *,
    mode: Literal["ml", "symbolic", "hybrid", "adaptive"] = "hybrid",
    alpha: float = DEFAULT_ALPHA,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate one routing configuration on a labeled CSV.

    Modes:
      - ml: alpha=1.0
      - symbolic: alpha=0.0
      - hybrid: fixed alpha (default 0.8)
      - adaptive: dynamic alpha per sample
    """
    test_path = Path(test_csv) if test_csv else DEFAULT_TEST
    texts, y_true = _load_test_csv(test_path)

    if mode == "ml":
        use_alpha, use_adaptive = 1.0, False
    elif mode == "symbolic":
        use_alpha, use_adaptive = 0.0, False
    elif mode == "adaptive":
        use_alpha, use_adaptive = None, True
    else:
        use_alpha, use_adaptive = alpha, False

    y_pred: list[str] = []
    prob_rows: list[np.ndarray] = []
    class_names: list[str] | None = None
    alphas_used: list[float] = []

    for text in texts:
        label, _, meta = predict_with_alpha(
            text, alpha=use_alpha, adaptive=use_adaptive
        )
        y_pred.append(label)
        prob_rows.append(np.asarray(meta["fused_probs"], dtype=np.float64))
        class_names = meta["class_order"]
        alphas_used.append(float(meta.get("alpha", use_alpha or DEFAULT_ALPHA)))

    assert class_names is not None
    metrics = compute_classification_metrics(y_true, y_pred, labels=class_names)
    y_prob = np.vstack(prob_rows)
    ece = expected_calibration_error(y_true, y_prob, class_names)

    result: dict[str, Any] = {
        "mode": mode,
        "alpha": use_alpha,
        "adaptive": use_adaptive,
        "n_samples": len(texts),
        "test_csv": str(test_path),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "ece": ece,
        "mean_alpha": float(np.mean(alphas_used)) if alphas_used else None,
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
        "labels": class_names,
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = mode if mode != "hybrid" else f"hybrid_a{use_alpha}"
        pd.DataFrame([{k: v for k, v in result.items() if k != "classification_report"}]).to_csv(
            out / f"benchmark_{tag}.csv", index=False
        )
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            str(out / f"confusion_{tag}.png"),
            title=f"InferAI — {tag}",
        )
        (out / f"report_{tag}.txt").write_text(metrics["classification_report"], encoding="utf-8")

    return result


def run_ablation_study(
    test_csv: str | Path | None = None,
    *,
    alphas: list[float] | None = None,
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Sweep fixed alpha values and export CSV + plots (accuracy, macro F1, ECE).
    """
    test_path = Path(test_csv) if test_csv else DEFAULT_TEST
    texts, y_true = _load_test_csv(test_path)
    sweep = alphas if alphas is not None else ABLATION_ALPHAS

    rows: list[dict[str, Any]] = []
    for a in sweep:
        y_pred: list[str] = []
        prob_rows: list[np.ndarray] = []
        class_names: list[str] | None = None
        for text in texts:
            label, _, meta = predict_with_alpha(text, alpha=a, adaptive=False)
            y_pred.append(label)
            prob_rows.append(np.asarray(meta["fused_probs"], dtype=np.float64))
            class_names = meta["class_order"]
        assert class_names is not None
        m = compute_classification_metrics(y_true, y_pred, labels=class_names)
        ece = expected_calibration_error(y_true, np.vstack(prob_rows), class_names)
        rows.append(
            {
                "alpha": a,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "ece": ece,
            }
        )

    df = pd.DataFrame(rows)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "ablation_alpha_sweep.csv", index=False)
        plot_ablation_summary(df, out)
        (out / "ablation_summary.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
    return df
