"""
Out-of-distribution (OOD) evaluation for InferAI.

Compares in-distribution ``test_set.csv`` behavior against a real-world
labeled corpus and reports accuracy, confidence, prediction distribution,
and Jensen–Shannon divergence between predicted label distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from classification.adaptive_router import DEFAULT_ALPHA
from evaluation.benchmark import predict_with_alpha
from evaluation.metrics import compute_classification_metrics
from evaluation.publication_figures import plot_categorical_counts, plot_confidence_distribution

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OOD = _ROOT / "dataset" / "real_world_dataset_v2.csv"
DEFAULT_ID = _ROOT / "dataset" / "test_set.csv"
LABELS = ("Anumana", "Pratyaksha", "Shabda", "Upamana")


def _pred_distribution(preds: list[str]) -> np.ndarray:
    counts = np.array([preds.count(lab) for lab in LABELS], dtype=float)
    if counts.sum() == 0:
        return np.ones(len(LABELS)) / len(LABELS)
    return counts / counts.sum()


def _evaluate_corpus(
    csv_path: Path,
    *,
    alpha: float = DEFAULT_ALPHA,
    max_samples: int | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    texts = df["text"].astype(str).tolist()
    y_true = df[label_col].astype(str).tolist()
    if max_samples is not None:
        texts, y_true = texts[:max_samples], y_true[:max_samples]

    y_pred: list[str] = []
    confs: list[float] = []
    for t in texts:
        pred, _, meta = predict_with_alpha(t, alpha=alpha, adaptive=False)
        y_pred.append(pred)
        confs.append(float(meta.get("adjusted_confidence", 0.0)))

    metrics = compute_classification_metrics(y_true, y_pred, labels=list(LABELS))
    return {
        "path": str(csv_path),
        "n_samples": len(texts),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mean_confidence": float(np.mean(confs)),
        "std_confidence": float(np.std(confs)),
        "pred_distribution": {lab: float(y_pred.count(lab)) for lab in LABELS},
        "true_distribution": {lab: float(y_true.count(lab)) for lab in LABELS},
        "y_pred": y_pred,
        "y_true": y_true,
        "confidences": confs,
    }


def run_ood_evaluation(
    id_csv: str | Path | None = None,
    ood_csv: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Compare ID vs OOD hybrid performance and prediction-shift (JSD)."""
    id_path = Path(id_csv or DEFAULT_ID)
    ood_path = Path(ood_csv or DEFAULT_OOD)
    if not ood_path.is_file():
        ood_path = _ROOT / "dataset" / "processed" / "ibm_reviewed_dataset.csv"

    id_res = _evaluate_corpus(id_path, max_samples=max_samples)
    ood_res = _evaluate_corpus(ood_path, max_samples=max_samples)

    p_id = _pred_distribution(id_res["y_pred"])
    p_ood = _pred_distribution(ood_res["y_pred"])
    jsd = float(jensenshannon(p_id, p_ood, base=2) ** 2)  # JSD in [0,1] with base 2

    result = {
        "id": {k: v for k, v in id_res.items() if k not in {"y_pred", "y_true", "confidences"}},
        "ood": {k: v for k, v in ood_res.items() if k not in {"y_pred", "y_true", "confidences"}},
        "jsd_prediction_distribution": jsd,
        "accuracy_drop": id_res["accuracy"] - ood_res["accuracy"],
        "confidence_drop": id_res["mean_confidence"] - ood_res["mean_confidence"],
        "narrative": (
            f"OOD accuracy={ood_res['accuracy']:.3f} vs ID={id_res['accuracy']:.3f} "
            f"(drop={id_res['accuracy'] - ood_res['accuracy']:.3f}). "
            f"Mean confidence ID={id_res['mean_confidence']:.1f}% / "
            f"OOD={ood_res['mean_confidence']:.1f}%. "
            f"Jensen-Shannon divergence between predicted label distributions="
            f"{jsd:.4f} (0=identical, 1=maximal)."
        ),
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"split": "id", **result["id"]},
                {"split": "ood", **result["ood"]},
            ]
        ).to_csv(out / "ood_evaluation.csv", index=False)
        pd.DataFrame(
            [
                {
                    "jsd_prediction_distribution": jsd,
                    "accuracy_drop": result["accuracy_drop"],
                    "confidence_drop": result["confidence_drop"],
                }
            ]
        ).to_csv(out / "ood_shift_summary.csv", index=False)
        plot_categorical_counts(
            {k: int(v) for k, v in ood_res["pred_distribution"].items()},
            out / "ood_prediction_distribution.png",
            title="OOD prediction distribution",
            xlabel="Predicted pramāṇa",
        )
        plot_confidence_distribution(
            ood_res["confidences"],
            out / "ood_confidence_distribution.png",
            title="OOD confidence distribution",
        )
        (out / "ood_report.md").write_text(_ood_md(result), encoding="utf-8")
    return result


def _ood_md(result: dict[str, Any]) -> str:
    idr, oodr = result["id"], result["ood"]
    return "\n".join(
        [
            "# Out-of-Distribution Evaluation",
            "",
            result["narrative"],
            "",
            "| Split | n | Accuracy | Macro F1 | Mean conf |",
            "|-------|---|----------|----------|-----------|",
            f"| ID | {idr['n_samples']} | {idr['accuracy']:.3f} | {idr['macro_f1']:.3f} | {idr['mean_confidence']:.1f} |",
            f"| OOD | {oodr['n_samples']} | {oodr['accuracy']:.3f} | {oodr['macro_f1']:.3f} | {oodr['mean_confidence']:.1f} |",
            "",
            f"JSD (predicted labels): **{result['jsd_prediction_distribution']:.4f}**",
            "",
        ]
    )
