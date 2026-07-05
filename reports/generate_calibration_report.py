"""
Model calibration analysis on the held-out test set.

Outputs:
  - reports/calibration_report.md
  - reports/figures/calibration_reliability_diagram.png
  - reports/figures/calibration_curve.png
  - reports/csv/calibration_metrics.csv

Usage:
    python reports/generate_calibration_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings
from reports._common import apply_plot_style, dataframe_to_markdown, ensure_reports_dir, save_dataframe_csv

N_BINS = 10


def expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = N_BINS
) -> float:
    """Multiclass ECE using max-probability confidence bins."""
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = predictions == y_true

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        if not np.any(in_bin):
            continue
        acc = float(np.mean(correct[in_bin]))
        conf = float(np.mean(confidences[in_bin]))
        ece += np.mean(in_bin) * abs(acc - conf)
    return float(ece)


def multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """Mean Brier score across one-vs-rest binary tasks."""
    scores: list[float] = []
    for c in range(n_classes):
        binary = (y_true == c).astype(int)
        scores.append(brier_score_loss(binary, y_proba[:, c]))
    return float(np.mean(scores))


def main() -> None:
    ensure_reports_dir()
    fig_dir = _ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(_ROOT / "models" / "nyaya_model.pkl")
    le = joblib.load(_ROOT / "models" / "label_encoder.pkl")
    tdf = pd.read_csv(_ROOT / "dataset" / "test_set.csv")

    texts = tdf["text"].astype(str).tolist()
    y_true = le.transform(tdf["pramana_label"].astype(str))
    X = generate_embeddings(texts)
    y_proba = model.predict_proba(X)
    n_classes = len(le.classes_)

    ece = expected_calibration_error(y_true, y_proba, N_BINS)
    brier = multiclass_brier(y_true, y_proba, n_classes)

    # Reliability diagram (max-confidence binary calibration)
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    correct = (predictions == y_true).astype(int)
    prob_true, prob_pred = calibration_curve(correct, confidences, n_bins=N_BINS, strategy="uniform")

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    ax.plot(prob_pred, prob_true, marker="o", color="#059669", label="Model")
    ax.set_xlabel("Mean predicted confidence (max prob)")
    ax.set_ylabel("Fraction of positives (correct)")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    rel_path = fig_dir / "calibration_reliability_diagram.png"
    plt.savefig(rel_path, bbox_inches="tight")
    plt.close()

    # Per-class calibration curves (one-vs-rest)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes_flat = axes.ravel()
    for c, cls in enumerate(le.classes_):
        binary = (y_true == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        pt, pp = calibration_curve(binary, y_proba[:, c], n_bins=min(8, N_BINS), strategy="quantile")
        ax = axes_flat[c]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.plot(pp, pt, marker="o", color="#10b981")
        ax.set_title(cls)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.grid(alpha=0.3)
    plt.suptitle("Per-class calibration curves (one-vs-rest)", y=1.02)
    plt.tight_layout()
    cal_path = fig_dir / "calibration_curve.png"
    plt.savefig(cal_path, bbox_inches="tight")
    plt.close()

    metrics_df = pd.DataFrame(
        [
            {"Metric": "Expected Calibration Error (ECE)", "Value": round(ece, 4)},
            {"Metric": "Multiclass Brier score (mean OvR)", "Value": round(brier, 4)},
            {"Metric": "Mean max confidence", "Value": round(float(np.mean(confidences)), 4)},
            {"Metric": "Accuracy at mean confidence", "Value": round(float(np.mean(correct)), 4)},
        ]
    )
    save_dataframe_csv(metrics_df, _ROOT / "reports" / "csv" / "calibration_metrics.csv")

    trustworthy = ece < 0.10
    lines: list[str] = []
    lines.append("# InferAI — Model Calibration Report\n\n")
    lines.append(
        "Probability calibration on `dataset/test_set.csv` using max-confidence ECE and "
        "one-vs-rest Brier scores. Generated by `reports/generate_calibration_report.py`.\n\n"
    )

    lines.append("## Metrics\n\n")
    lines.append(dataframe_to_markdown(metrics_df) + "\n\n")

    lines.append("## Reliability diagram\n\n")
    lines.append(f"![Reliability diagram](figures/calibration_reliability_diagram.png)\n\n")

    lines.append("## Per-class calibration curves\n\n")
    lines.append(f"![Calibration curves](figures/calibration_curve.png)\n\n")

    lines.append("## Interpretation\n\n")
    if trustworthy:
        lines.append(
            f"ECE ({ece:.3f}) is below 0.10, suggesting **moderately trustworthy** confidence "
            "estimates on the held-out set: predicted max-probabilities align reasonably with "
            "empirical accuracy within bins.\n\n"
        )
    else:
        lines.append(
            f"ECE ({ece:.3f}) exceeds 0.10, indicating **miscalibrated** confidence: the model "
            "is often over- or under-confident relative to observed accuracy. This is common when "
            "logistic regression is trained on synthetic-heavy data but evaluated on curated "
            "real-world examples.\n\n"
        )

    lines.append(
        f"The multiclass Brier score ({brier:.4f}) aggregates probabilistic error across classes; "
        "lower is better. Users should treat `confidence` and `adjusted_confidence` in the API as "
        "**ranking signals** rather than calibrated probabilities until post-hoc calibration "
        "(e.g. temperature scaling or isotonic regression on a validation fold) is applied.\n"
    )

    out = _ROOT / "reports" / "calibration_report.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
