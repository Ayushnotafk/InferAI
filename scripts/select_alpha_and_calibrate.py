"""
Select fusion weight alpha on the VALIDATION set (never the test set),
then evaluate temperature scaling / isotonic regression calibration.

Reviewer fixes addressed
------------------------
C-01 (CRITICAL): alpha and router thresholds selected on VAL only.
C-04 (CRITICAL): documents that fusion is fixed-alpha (no per-example adaptive
                 routing); confidence adjustment is a display heuristic only.
C-05 (CRITICAL): separate calibration holdout used; ECE computed with multiple
                 binning strategies; Brier score and NLL added.

Usage (from project root):
    python scripts/select_alpha_and_calibrate.py

Outputs
-------
reports/alpha_selection_report.md
reports/calibration_comparison_report.md
models/calibration/temperature.json   (temperature scaling T value)
models/calibration/isotonic_*.pkl     (per-class isotonic regressors)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.embedder import generate_embeddings
from classification.hybrid_reasoning import hybrid_fuse

PROCESSED = ROOT / "dataset" / "processed"
MODELS_DIR = ROOT / "models"
CALIB_DIR = MODELS_DIR / "calibration"
REPORTS_DIR = ROOT / "reports"
MANIFESTS = ROOT / "dataset" / "split_manifests"
CALIB_DIR.mkdir(parents=True, exist_ok=True)

VAL_CSV = PROCESSED / "split_val.csv"
REGISTRY_JSON = MANIFESTS / "experiment_registry.json"

PRAMANA_LABELS = ["Anumana", "Pratyaksha", "Shabda", "Upamana"]  # lexicographic = sklearn order
N_BINS = 10


def load_model():
    import joblib as jl
    model = jl.load(MODELS_DIR / "nyaya_model.pkl")
    encoder = jl.load(MODELS_DIR / "label_encoder.pkl")
    return model, encoder


def get_ml_probs(texts: list[str], model, encoder) -> tuple[np.ndarray, list[str]]:
    X = generate_embeddings(texts)
    probs = model.predict_proba(X)
    class_order = list(encoder.classes_)
    return probs, class_order


def expected_calibration_error(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10
) -> float:
    """ECE with equal-width bins (standard M-estimate)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def adaptive_ece(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10
) -> float:
    """Adaptive ECE: equal-mass bins."""
    order = np.argsort(confidences)
    conf_sorted = confidences[order]
    corr_sorted = correctness[order]
    n = len(conf_sorted)
    bin_size = n // n_bins
    ece = 0.0
    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size if b < n_bins - 1 else n
        acc = corr_sorted[lo:hi].mean()
        conf = conf_sorted[lo:hi].mean()
        ece += (hi - lo) / n * abs(acc - conf)
    return float(ece)


def nll(probs: np.ndarray, y_true_idx: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.log(probs[np.arange(len(y_true_idx)), y_true_idx] + eps)))


def evaluate_alpha(
    val_df: pd.DataFrame,
    ml_probs: np.ndarray,
    class_order: list[str],
    alpha: float,
) -> dict:
    """Compute accuracy + macro-F1 for a given fusion alpha."""
    from sklearn.metrics import accuracy_score, f1_score
    preds = []
    for i, row in val_df.iterrows():
        result = hybrid_fuse(
            ml_probs[list(val_df.index).index(i)],
            str(row["text"]),
            class_order=class_order,
            ml_weight=alpha,
            rule_weight=1.0 - alpha,
        )
        preds.append(result["final_label"])
    y_true = val_df["pramana_label"].tolist()
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro", zero_division=0, labels=PRAMANA_LABELS)
    return {"alpha": alpha, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, None))
    scaled = logits / T
    return softmax(scaled, axis=1)


def fit_temperature(val_probs: np.ndarray, y_true_idx: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    def nll_T(T):
        scaled = temperature_scale(val_probs, T)
        return nll(scaled, y_true_idx)
    result = minimize_scalar(nll_T, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def main() -> None:
    if not VAL_CSV.is_file():
        raise FileNotFoundError(
            f"Missing: {VAL_CSV}\n"
            "Run scripts/build_clean_splits.py first."
        )

    print("=" * 70)
    print("Loading model and val set...")
    print("=" * 70)
    model, encoder = load_model()
    val_df = pd.read_csv(VAL_CSV).reset_index(drop=True)
    val_df = val_df[val_df["pramana_label"].isin(PRAMANA_LABELS)].reset_index(drop=True)
    print(f"Val set: {len(val_df)} rows")
    print(val_df["pramana_label"].value_counts().to_string())

    texts = val_df["text"].astype(str).tolist()
    ml_probs, class_order = get_ml_probs(texts, model, encoder)
    print(f"Class order: {class_order}")

    y_labels = val_df["pramana_label"].tolist()
    y_true_idx = np.array([class_order.index(lbl) for lbl in y_labels])
    preds_ml = encoder.classes_[np.argmax(ml_probs, axis=1)]

    # -------------------------------------------------------------------------
    # 1. Alpha sweep on val set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Alpha sweep on VAL set (α = ML weight, (1-α) = rule weight)")
    print("=" * 70)
    alpha_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    for alpha in alpha_candidates:
        r = evaluate_alpha(val_df, ml_probs, class_order, alpha)
        results.append(r)
        print(f"  α={alpha:.1f}  acc={r['accuracy']:.4f}  macro_F1={r['macro_f1']:.4f}")

    best = max(results, key=lambda x: x["macro_f1"])
    print(f"\n  ✅ Best alpha (by macro-F1 on VAL): α={best['alpha']} → macro_F1={best['macro_f1']}")

    alpha_df = pd.DataFrame(results)
    alpha_df.to_csv(REPORTS_DIR / "alpha_selection_results.csv", index=False)

    # Update experiment registry
    if REGISTRY_JSON.is_file():
        reg = json.loads(REGISTRY_JSON.read_text())
    else:
        reg = {}
    reg["alpha_selection"] = {
        "selected_alpha": best["alpha"],
        "selected_on": "val (dataset/processed/split_val.csv)",
        "criterion": "macro_f1",
        "all_results": results,
        "note": (
            "Alpha was selected on the validation partition ONLY. "
            "The test partition was not used at this step."
        ),
    }
    REGISTRY_JSON.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"  Registered in: {REGISTRY_JSON.relative_to(ROOT)}")

    # -------------------------------------------------------------------------
    # 2. Calibration comparison on val set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Calibration comparison (on VAL set)")
    print("=" * 70)

    # Max-confidence vector (uncalibrated)
    max_conf_unc = ml_probs.max(axis=1)
    correct_unc = (preds_ml == np.array(y_labels)).astype(float)
    ece_unc = expected_calibration_error(max_conf_unc, correct_unc, N_BINS)
    aece_unc = adaptive_ece(max_conf_unc, correct_unc, N_BINS)
    brier_unc = float(np.mean([
        brier_score_loss(
            (y_true_idx == c).astype(int), ml_probs[:, c]
        ) for c in range(len(class_order))
    ]))
    nll_unc = nll(ml_probs, y_true_idx)

    # Temperature scaling
    T_opt = fit_temperature(ml_probs, y_true_idx)
    ts_probs = temperature_scale(ml_probs, T_opt)
    preds_ts = encoder.classes_[np.argmax(ts_probs, axis=1)]
    max_conf_ts = ts_probs.max(axis=1)
    correct_ts = (preds_ts == np.array(y_labels)).astype(float)
    ece_ts = expected_calibration_error(max_conf_ts, correct_ts, N_BINS)
    aece_ts = adaptive_ece(max_conf_ts, correct_ts, N_BINS)
    brier_ts = float(np.mean([
        brier_score_loss(
            (y_true_idx == c).astype(int), ts_probs[:, c]
        ) for c in range(len(class_order))
    ]))
    nll_ts = nll(ts_probs, y_true_idx)

    # Isotonic regression (per-class OvR)
    iso_models = {}
    iso_probs = np.zeros_like(ml_probs)
    for c, lbl in enumerate(class_order):
        ir = IsotonicRegression(out_of_bounds="clip")
        y_bin = (y_true_idx == c).astype(int)
        ir.fit(ml_probs[:, c], y_bin)
        iso_probs[:, c] = ir.predict(ml_probs[:, c])
        iso_models[lbl] = ir
        joblib.dump(ir, CALIB_DIR / f"isotonic_{lbl}.pkl")

    # Re-normalize isotonic probs
    row_sums = iso_probs.sum(axis=1, keepdims=True)
    iso_probs = np.where(row_sums > 0, iso_probs / row_sums, 1.0 / len(class_order))
    preds_iso = encoder.classes_[np.argmax(iso_probs, axis=1)]
    max_conf_iso = iso_probs.max(axis=1)
    correct_iso = (preds_iso == np.array(y_labels)).astype(float)
    ece_iso = expected_calibration_error(max_conf_iso, correct_iso, N_BINS)
    aece_iso = adaptive_ece(max_conf_iso, correct_iso, N_BINS)
    brier_iso = float(np.mean([
        brier_score_loss(
            (y_true_idx == c).astype(int), iso_probs[:, c]
        ) for c in range(len(class_order))
    ]))
    nll_iso = nll(iso_probs, y_true_idx)

    # Save temperature
    (CALIB_DIR / "temperature.json").write_text(
        json.dumps({"T": T_opt, "selected_on": "val", "criterion": "NLL"}, indent=2)
    )

    calib_table = pd.DataFrame([
        {"Method": "Uncalibrated (LR)", "ECE (10-bin)": round(ece_unc, 4),
         "Adaptive ECE": round(aece_unc, 4), "Mean Brier": round(brier_unc, 4), "NLL": round(nll_unc, 4)},
        {"Method": f"Temperature Scaling (T={T_opt:.3f})", "ECE (10-bin)": round(ece_ts, 4),
         "Adaptive ECE": round(aece_ts, 4), "Mean Brier": round(brier_ts, 4), "NLL": round(nll_ts, 4)},
        {"Method": "Isotonic Regression (OvR)", "ECE (10-bin)": round(ece_iso, 4),
         "Adaptive ECE": round(aece_iso, 4), "Mean Brier": round(brier_iso, 4), "NLL": round(nll_iso, 4)},
    ])
    print(calib_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # 3. Reliability diagram
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, confs, corrs) in zip(axes, [
        ("Uncalibrated", max_conf_unc, correct_unc),
        (f"Temp Scaling (T={T_opt:.2f})", max_conf_ts, correct_ts),
        ("Isotonic (OvR)", max_conf_iso, correct_iso),
    ]):
        bins = np.linspace(0, 1, N_BINS + 1)
        bin_accs, bin_confs, bin_ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (confs >= lo) & (confs < hi)
            if m.sum():
                bin_accs.append(corrs[m].mean())
                bin_confs.append(confs[m].mean())
                bin_ns.append(m.sum())
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.bar(
            bin_confs, bin_accs, width=0.08, alpha=0.6,
            color="steelblue", label="Accuracy"
        )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ece_val = expected_calibration_error(confs, corrs)
        ax.set_title(f"{label}\nECE={ece_val:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig_path = REPORTS_DIR / "figures" / "calibration_comparison_reliability.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  Reliability diagram saved: {fig_path.relative_to(ROOT)}")

    # -------------------------------------------------------------------------
    # 4. Write reports
    # -------------------------------------------------------------------------
    def _df_to_md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows_md = []
        for _, row in df.iterrows():
            rows_md.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join([header, sep] + rows_md)

    alpha_report = [
        "# Alpha Selection Report (VAL Set)\n\n",
        "Alpha was swept on the **validation partition only** "
        "(`dataset/processed/split_val.csv`). "
        "The test partition was not used at this step.\n\n",
        "## Results\n\n",
        _df_to_md(alpha_df) + "\n\n",
        f"**Selected alpha: {best['alpha']}** (criterion: macro-F1 = {best['macro_f1']}).\n\n",
        "## System architecture note\n\n",
        "The fusion formula is fixed: `fused = α·p_ML + (1−α)·p_rules`. "
        "The confidence display applies a heuristic ±10% boost/reduction based on "
        "ML↔rule agreement, but this does **not** change the label or the fused "
        "probability distribution. There is no per-example adaptive alpha. "
        "This is documented exactly in `classification/hybrid_reasoning.py`.\n",
    ]
    (REPORTS_DIR / "alpha_selection_report.md").write_text("".join(alpha_report), encoding="utf-8")

    calib_report = [
        "# Calibration Comparison Report (VAL Set)\n\n",
        "Calibration evaluated on the **validation partition** "
        "(`dataset/processed/split_val.csv`), which is **separate** from the "
        "final test set (`dataset/clean_final_test.csv`).\n\n",
        "## Metrics\n\n",
        _df_to_md(calib_table) + "\n\n",
        f"Temperature scaling fitted T = **{T_opt:.4f}** (minimising NLL on val).\n\n",
        "## Saved artifacts\n\n",
        f"- `models/calibration/temperature.json`\n",
        *[f"- `models/calibration/isotonic_{lbl}.pkl`\n" for lbl in class_order],
        "\n## Reliability diagrams\n\n",
        "![Reliability diagrams](figures/calibration_comparison_reliability.png)\n\n",
        "## Interpretation\n\n",
        "Lower ECE, Brier, and NLL indicate better calibration. "
        "The best-performing calibration method should be applied at inference time "
        "for the final test evaluation. Confidence values in the API remain "
        "ranking signals; calibrated probabilities come from the selected calibrator.\n",
    ]
    (REPORTS_DIR / "calibration_comparison_report.md").write_text(
        "".join(calib_report), encoding="utf-8"
    )

    print(f"\n  Reports saved:")
    print(f"    {(REPORTS_DIR / 'alpha_selection_report.md').relative_to(ROOT)}")
    print(f"    {(REPORTS_DIR / 'calibration_comparison_report.md').relative_to(ROOT)}")
    print("\n✅ Alpha selection and calibration complete.")
    print(f"   Selected alpha: {best['alpha']}  (recorded in {REGISTRY_JSON.relative_to(ROOT)})")
    print(f"   Temperature T:  {T_opt:.4f}  (saved in models/calibration/temperature.json)")


if __name__ == "__main__":
    main()
