"""
Post-hoc calibration experiments for InferAI ML probabilities.

Compares Temperature Scaling, Platt (sigmoid), and Isotonic Regression.
Selects the method that minimizes ECE without decreasing accuracy by >1 pp.
Does not modify the production API; results are evaluation-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from classification.adaptive_router import DEFAULT_ALPHA
from classification.embedder import generate_embeddings
from classification.hybrid_reasoning import hybrid_fuse
from evaluation.metrics import expected_calibration_error
from evaluation.publication_figures import _PALETTE, _save

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent


def _ece_from_probs(
    y_true: list[str], probs: np.ndarray, class_names: list[str]
) -> float:
    return expected_calibration_error(y_true, probs, class_names)


def _reliability_bins(
    y_true: list[str], probs: np.ndarray, class_names: list[str], n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    conf = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    pred = [class_names[i] for i in pred_idx]
    acc = np.array([1.0 if p == t else 0.0 for p, t in zip(pred, y_true)])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        bin_conf.append(conf[mask].mean())
        bin_acc.append(acc[mask].mean())
    return np.asarray(bin_conf), np.asarray(bin_acc)


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Fit a single temperature T by minimizing NLL on calibration logits."""
    # Grid search is stable and dependency-light.
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 5.0, 46):
        scaled = logits / t
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=1, keepdims=True)
        nll = -np.mean(np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def run_calibration_study(
    train_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fit calibrators on a held-out slice of train, evaluate on test_set.

    Methods: none (baseline), temperature, platt, isotonic.
    """
    train_path = Path(train_csv or _ROOT / "dataset" / "raw" / "nyaya_dataset_merged.csv")
    test_path = Path(test_csv or _ROOT / "dataset" / "test_set.csv")
    if not train_path.is_file():
        train_path = _ROOT / "dataset" / "master_dataset.csv"

    model = joblib.load(_ROOT / "models" / "nyaya_model.pkl")
    le = joblib.load(_ROOT / "models" / "label_encoder.pkl")
    class_names = le.classes_.tolist()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    tlab = "pramana_label" if "pramana_label" in train_df.columns else "label"
    elab = "pramana_label" if "pramana_label" in test_df.columns else "label"

    # Calibration split from training distribution (not the final test set).
    cal_df, _ = train_test_split(
        train_df,
        test_size=0.7,
        random_state=random_state,
        stratify=train_df[tlab],
    )
    # Cap calibration size for speed.
    if len(cal_df) > 400:
        cal_df = cal_df.sample(n=400, random_state=random_state)

    cal_texts = cal_df["text"].astype(str).tolist()
    cal_y = cal_df[tlab].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()
    test_y = test_df[elab].astype(str).tolist()

    X_cal = generate_embeddings(cal_texts)
    X_te = generate_embeddings(test_texts)

    # Logits / probs from frozen production model.
    if hasattr(model, "decision_function"):
        logits_cal = model.decision_function(X_cal)
        logits_te = model.decision_function(X_te)
        if logits_cal.ndim == 1:
            logits_cal = np.column_stack([-logits_cal, logits_cal])
            logits_te = np.column_stack([-logits_te, logits_te])
    else:
        probs_tmp = model.predict_proba(X_cal)
        logits_cal = np.log(np.clip(probs_tmp, 1e-12, 1.0))
        probs_tmp = model.predict_proba(X_te)
        logits_te = np.log(np.clip(probs_tmp, 1e-12, 1.0))

    probs_cal_raw = model.predict_proba(X_cal)
    probs_te_raw = model.predict_proba(X_te)
    y_cal_idx = le.transform(cal_y)

    # --- Temperature ---
    T = fit_temperature(logits_cal, y_cal_idx)
    probs_te_temp = apply_temperature(logits_te, T)

    # --- Platt (sigmoid) via CalibratedClassifierCV on a clone fit ---
    # Fit a fresh LR on cal set then calibrate with sigmoid on the same fold structure.
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    platt = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    platt.fit(X_cal, y_cal_idx)
    probs_te_platt = platt.predict_proba(X_te)

    # --- Isotonic ---
    iso = CalibratedClassifierCV(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        method="isotonic",
        cv=3,
    )
    iso.fit(X_cal, y_cal_idx)
    probs_te_iso = iso.predict_proba(X_te)

    methods = {
        "none": probs_te_raw,
        "temperature": probs_te_temp,
        "platt": probs_te_platt,
        "isotonic": probs_te_iso,
    }

    rows: list[dict[str, Any]] = []
    baseline_acc = float(
        accuracy_score(test_y, [class_names[i] for i in probs_te_raw.argmax(1)])
    )

    for name, probs in methods.items():
        # Align class order to label encoder classes.
        if probs.shape[1] != len(class_names):
            continue
        pred_labels = [class_names[i] for i in probs.argmax(axis=1)]
        acc = float(accuracy_score(test_y, pred_labels))
        ece = _ece_from_probs(test_y, probs, class_names)

        # Also evaluate hybrid fusion with calibrated ML probs.
        hybrid_preds = []
        hybrid_probs = []
        for text, p in zip(test_texts, probs):
            h = hybrid_fuse(p, text, class_order=class_names, alpha=DEFAULT_ALPHA)
            hybrid_preds.append(h["final_label"])
            hybrid_probs.append(h["fused_probs"])
        hybrid_acc = float(accuracy_score(test_y, hybrid_preds))
        hybrid_ece = _ece_from_probs(test_y, np.asarray(hybrid_probs), class_names)

        rows.append(
            {
                "method": name,
                "temperature": T if name == "temperature" else None,
                "ml_accuracy": acc,
                "ml_ece": ece,
                "hybrid_accuracy": hybrid_acc,
                "hybrid_ece": hybrid_ece,
                "acc_drop_vs_none": baseline_acc - acc,
            }
        )

    # Select best: minimize hybrid ECE subject to hybrid accuracy drop ≤ 0.01.
    eligible = [r for r in rows if (baseline_acc - r["hybrid_accuracy"]) <= 0.01]
    if not eligible:
        eligible = rows
    best = min(eligible, key=lambda r: r["hybrid_ece"])

    result = {
        "methods": rows,
        "best_method": best["method"],
        "best_hybrid_ece": best["hybrid_ece"],
        "best_hybrid_accuracy": best["hybrid_accuracy"],
        "temperature": T,
        "recommendation": (
            f"Prefer **{best['method']}** calibration: hybrid ECE="
            f"{best['hybrid_ece']:.4f}, hybrid accuracy={best['hybrid_accuracy']:.4f}. "
            "Production API remains unchanged; apply calibrator offline if adopted."
        ),
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "calibration_comparison.csv", index=False)
        (out / "calibration_report.md").write_text(
            _cal_markdown(result), encoding="utf-8"
        )
        # Reliability diagrams for baseline vs best.
        for name in ("none", best["method"]):
            probs = methods[name]
            bc, ba = _reliability_bins(test_y, probs, class_names)
            fig, ax = plt.subplots(figsize=(5.5, 5.0))
            ax.plot([0, 1], [0, 1], "--", color=_PALETTE["grid"], label="perfect")
            if len(bc):
                ax.plot(bc, ba, "o-", color=_PALETTE["line"], label=name)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Reliability diagram — {name}")
            ax.legend(frameon=False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            _save(fig, out / f"calibration_reliability_{name}.png")

    return result


def _cal_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Calibration Study",
        "",
        result["recommendation"],
        "",
        "| Method | ML Acc | ML ECE | Hybrid Acc | Hybrid ECE |",
        "|--------|--------|--------|------------|------------|",
    ]
    for r in result["methods"]:
        lines.append(
            f"| {r['method']} | {r['ml_accuracy']:.3f} | {r['ml_ece']:.3f} | "
            f"{r['hybrid_accuracy']:.3f} | {r['hybrid_ece']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)
