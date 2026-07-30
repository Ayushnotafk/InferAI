"""
Statistical significance testing for InferAI mode comparisons.

Implements McNemar's test, bootstrap confidence intervals, and paired
permutation tests on per-sample correctness indicators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.benchmark import predict_with_alpha
from evaluation.metrics import compute_classification_metrics

ALPHA_SIG = 0.05
N_BOOTSTRAP = 1000
N_PERMUTATION = 2000
RNG = np.random.default_rng(42)


def collect_mode_predictions(
    texts: list[str],
    y_true: list[str],
    mode: str,
) -> dict[str, Any]:
    """Run one routing mode and return predictions + correctness mask."""
    y_pred: list[str] = []
    confidences: list[float] = []
    from classification.adaptive_router import DEFAULT_ALPHA

    if mode == "ml":
        alpha, adaptive = 1.0, False
    elif mode == "symbolic":
        alpha, adaptive = 0.0, False
    elif mode == "adaptive":
        alpha, adaptive = None, True
    else:  # hybrid — ablation-optimal fixed alpha
        alpha, adaptive = DEFAULT_ALPHA, False

    for text in texts:
        pred, ml_conf, meta = predict_with_alpha(text, alpha=alpha, adaptive=adaptive)
        y_pred.append(pred)
        confidences.append(float(meta.get("adjusted_confidence", ml_conf)))

    correct = np.array([p == t for p, t in zip(y_pred, y_true)], dtype=bool)
    metrics = compute_classification_metrics(y_true, y_pred)
    return {
        "mode": mode,
        "y_pred": y_pred,
        "correct": correct,
        "confidences": confidences,
        "metrics": metrics,
    }


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """
    McNemar's test on paired binary correctness.

    Contigency: b = A wrong & B right, c = A right & B wrong.
    Uses exact binomial test when b + c < 25, else chi-square with continuity.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b_count = int((~a & b).sum())  # A wrong, B right
    c_count = int((a & ~b).sum())  # A right, B wrong
    n_discordant = b_count + c_count

    if n_discordant == 0:
        p_value = 1.0
        statistic = 0.0
        method = "no_discordant_pairs"
    elif n_discordant < 25:
        # Exact McNemar (binomial)
        p_value = float(stats.binomtest(b_count, n_discordant, 0.5).pvalue)
        statistic = float(b_count)
        method = "exact_binomial"
    else:
        statistic = (abs(b_count - c_count) - 1) ** 2 / n_discordant
        p_value = float(stats.chi2.sf(statistic, df=1))
        method = "chi_square_continuity"

    return {
        "statistic": statistic,
        "p_value": p_value,
        "b_a_wrong_b_right": b_count,
        "c_a_right_b_wrong": c_count,
        "method": method,
        "significant": bool(p_value < ALPHA_SIG),
    }


def paired_permutation_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    *,
    n_perm: int = N_PERMUTATION,
) -> dict[str, Any]:
    """
    Two-sided paired permutation test on accuracy difference.

    Under H0, randomly swap A/B labels within each paired sample.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    observed = float(a.mean() - b.mean())
    n = len(a)
    count = 0
    for _ in range(n_perm):
        swap = RNG.random(n) < 0.5
        a_perm = np.where(swap, b, a)
        b_perm = np.where(swap, a, b)
        diff = float(a_perm.mean() - b_perm.mean())
        if abs(diff) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {
        "observed_diff": observed,
        "p_value": float(p_value),
        "n_perm": n_perm,
        "significant": bool(p_value < ALPHA_SIG),
    }


def bootstrap_metric_ci(
    y_true: list[str],
    y_pred: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    *,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for an arbitrary classification metric."""
    n = len(y_true)
    idx = np.arange(n)
    scores = []
    for _ in range(n_boot):
        sample = RNG.choice(idx, size=n, replace=True)
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        scores.append(metric_fn(yt, yp))
    arr = np.asarray(scores, dtype=float)
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return {
        "point": float(metric_fn(y_true, y_pred)),
        "ci_low": lo,
        "ci_high": hi,
        "mean_boot": float(arr.mean()),
        "std_boot": float(arr.std(ddof=1)),
    }


def bootstrap_classification_cis(
    y_true: list[str],
    y_pred: list[str],
    *,
    n_boot: int = N_BOOTSTRAP,
) -> dict[str, dict[str, float]]:
    """95% bootstrap CIs for accuracy, macro precision/recall/F1."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    def _acc(yt, yp):
        return float(accuracy_score(yt, yp))

    def _mp(yt, yp):
        return float(precision_score(yt, yp, average="macro", zero_division=0))

    def _mr(yt, yp):
        return float(recall_score(yt, yp, average="macro", zero_division=0))

    def _mf(yt, yp):
        return float(f1_score(yt, yp, average="macro", zero_division=0))

    return {
        "accuracy": bootstrap_metric_ci(y_true, y_pred, _acc, n_boot=n_boot),
        "macro_precision": bootstrap_metric_ci(y_true, y_pred, _mp, n_boot=n_boot),
        "macro_recall": bootstrap_metric_ci(y_true, y_pred, _mr, n_boot=n_boot),
        "macro_f1": bootstrap_metric_ci(y_true, y_pred, _mf, n_boot=n_boot),
    }


def run_significance_suite(
    test_csv: str | Path,
    *,
    out_dir: str | Path | None = None,
    comparisons: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Compare routing modes with McNemar + permutation tests and bootstrap CIs.
    """
    df = pd.read_csv(test_csv)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    texts = df["text"].astype(str).tolist()
    y_true = df[label_col].astype(str).tolist()

    modes = ("ml", "symbolic", "hybrid", "adaptive")
    preds: dict[str, dict[str, Any]] = {}
    for mode in modes:
        preds[mode] = collect_mode_predictions(texts, y_true, mode)

    comparisons = comparisons or [
        ("ml", "hybrid"),
        ("ml", "adaptive"),
        ("symbolic", "hybrid"),
        ("hybrid", "adaptive"),
    ]

    rows: list[dict[str, Any]] = []
    for a, b in comparisons:
        mcn = mcnemar_test(preds[a]["correct"], preds[b]["correct"])
        perm = paired_permutation_test(preds[a]["correct"], preds[b]["correct"])
        # Prefer McNemar for the primary "significant" flag; report both.
        rows.append(
            {
                "comparison": f"{a} vs {b}",
                "mode_a": a,
                "mode_b": b,
                "acc_a": float(preds[a]["correct"].mean()),
                "acc_b": float(preds[b]["correct"].mean()),
                "mcnemar_p": mcn["p_value"],
                "mcnemar_significant": mcn["significant"],
                "mcnemar_method": mcn["method"],
                "permutation_p": perm["p_value"],
                "permutation_significant": perm["significant"],
                "p_value": mcn["p_value"],
                "significant": "Yes" if mcn["significant"] else "No",
            }
        )

    ci_rows: list[dict[str, Any]] = []
    for mode, payload in preds.items():
        cis = bootstrap_classification_cis(y_true, payload["y_pred"])
        for metric, ci in cis.items():
            ci_rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "point": ci["point"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                }
            )

    result = {
        "comparisons": rows,
        "confidence_intervals": ci_rows,
        "n_samples": len(texts),
        "alpha": ALPHA_SIG,
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "statistical_significance.csv", index=False)
        pd.DataFrame(ci_rows).to_csv(out / "bootstrap_confidence_intervals.csv", index=False)
        (out / "statistical_significance.md").write_text(
            _significance_markdown(result), encoding="utf-8"
        )
    return result


def _significance_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Statistical Significance Analysis",
        "",
        f"Paired comparisons on n={result['n_samples']} held-out samples. "
        f"Primary test: McNemar (alpha={result['alpha']}). "
        "Permutation p-values are reported as a sensitivity check.",
        "",
        "| Comparison | Acc A | Acc B | McNemar p | Significant | Permutation p |",
        "|------------|-------|-------|-----------|-------------|---------------|",
    ]
    for row in result["comparisons"]:
        p = row["mcnemar_p"]
        p_str = "<0.0001" if p < 0.0001 else f"{p:.4f}"
        lines.append(
            f"| {row['comparison']} | {row['acc_a']:.3f} | {row['acc_b']:.3f} | "
            f"{p_str} | {row['significant']} | {row['permutation_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- **McNemar** tests whether discordant prediction errors differ between two systems.",
            "- A significant result (p < 0.05) indicates one system corrects errors the other misses "
            "more often than chance.",
            "- Bootstrap CIs for Accuracy / Precision / Recall / Macro F1 are in "
            "`bootstrap_confidence_intervals.csv`.",
            "",
        ]
    )
    return "\n".join(lines)
