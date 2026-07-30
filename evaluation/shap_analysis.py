"""
Publication-oriented SHAP analysis for InferAI's linear embedding classifier.

Produces top positive/negative dimensions, global mean-|SHAP|, and per-class
importance figures at 300 DPI. Does not change the API explanation payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classification.embedder import generate_embeddings
from evaluation.publication_figures import _PALETTE, _save
from explanation_engine.shap_explainer import (
    _shap_vector_for_class,
    get_explainer,
    load_classifier,
)

_ROOT = Path(__file__).resolve().parent.parent


def run_shap_publication_analysis(
    test_csv: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    max_samples: int = 80,
    top_k: int = 15,
) -> dict[str, Any]:
    """Aggregate SHAP over a sample of texts for global / per-class figures."""
    path = Path(test_csv or _ROOT / "dataset" / "test_set.csv")
    df = pd.read_csv(path).head(max_samples)
    texts = df["text"].astype(str).tolist()

    model, le = load_classifier()
    explainer = get_explainer()
    class_names = le.classes_.tolist()
    n_features = int(getattr(model, "n_features_in_", 384))

    global_abs = np.zeros(n_features, dtype=float)
    pos_mass = np.zeros(n_features, dtype=float)
    neg_mass = np.zeros(n_features, dtype=float)
    per_class_abs = {c: np.zeros(n_features, dtype=float) for c in class_names}
    per_class_n = {c: 0 for c in class_names}

    X = generate_embeddings(texts)
    preds = model.predict(X)
    shap_out = explainer.shap_values(X)

    for i, pred in enumerate(preds):
        label = class_names[int(pred)]
        # Extract vector for sample i, class pred.
        arr = np.asarray(shap_out)
        if isinstance(shap_out, list):
            vec = np.asarray(shap_out[int(pred)])[i].reshape(-1)
        elif arr.ndim == 3:
            vec = arr[i, :, int(pred)]
        elif arr.ndim == 2:
            vec = arr[i]
        else:
            vec = _shap_vector_for_class(shap_out, int(pred))

        global_abs += np.abs(vec)
        pos_mass += np.clip(vec, 0, None)
        neg_mass += np.clip(-vec, 0, None)
        per_class_abs[label] += np.abs(vec)
        per_class_n[label] += 1

    global_abs /= max(len(texts), 1)
    pos_mass /= max(len(texts), 1)
    neg_mass /= max(len(texts), 1)

    top_pos = np.argsort(-pos_mass)[:top_k]
    top_neg = np.argsort(-neg_mass)[:top_k]
    top_global = np.argsort(-global_abs)[:top_k]

    result = {
        "n_samples": len(texts),
        "top_positive": [
            {"embedding_dim": int(i), "mean_positive_shap": float(pos_mass[i])}
            for i in top_pos
        ],
        "top_negative": [
            {"embedding_dim": int(i), "mean_negative_shap": float(neg_mass[i])}
            for i in top_neg
        ],
        "top_global": [
            {"embedding_dim": int(i), "mean_abs_shap": float(global_abs[i])}
            for i in top_global
        ],
        "per_class_top": {},
    }

    for c in class_names:
        if per_class_n[c] == 0:
            continue
        avg = per_class_abs[c] / per_class_n[c]
        idx = np.argsort(-avg)[:top_k]
        result["per_class_top"][c] = [
            {"embedding_dim": int(i), "mean_abs_shap": float(avg[i])} for i in idx
        ]

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result["top_positive"]).to_csv(out / "shap_top_positive.csv", index=False)
        pd.DataFrame(result["top_negative"]).to_csv(out / "shap_top_negative.csv", index=False)
        pd.DataFrame(result["top_global"]).to_csv(out / "shap_top_global.csv", index=False)

        _bar_dims(
            [r["embedding_dim"] for r in result["top_positive"]],
            [r["mean_positive_shap"] for r in result["top_positive"]],
            out / "shap_top_positive.png",
            "Top positive SHAP dimensions",
            _PALETTE["line"],
        )
        _bar_dims(
            [r["embedding_dim"] for r in result["top_negative"]],
            [r["mean_negative_shap"] for r in result["top_negative"]],
            out / "shap_top_negative.png",
            "Top negative SHAP dimensions",
            _PALETTE["accent"],
        )
        _bar_dims(
            [r["embedding_dim"] for r in result["top_global"]],
            [r["mean_abs_shap"] for r in result["top_global"]],
            out / "shap_global_importance.png",
            "Global mean |SHAP| importance",
            _PALETTE["bar"],
        )

        # Per-class small multiples.
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False)
        for ax, c in zip(axes.ravel(), class_names):
            rows = result["per_class_top"].get(c, [])[:10]
            if not rows:
                ax.set_visible(False)
                continue
            dims = [str(r["embedding_dim"]) for r in rows][::-1]
            vals = [r["mean_abs_shap"] for r in rows][::-1]
            ax.barh(dims, vals, color=_PALETTE["bar"])
            ax.set_title(c)
            ax.set_xlabel("mean |SHAP|")
        fig.suptitle("Per-class SHAP importance")
        fig.tight_layout()
        _save(fig, out / "shap_per_class_importance.png")

        (out / "shap_analysis.md").write_text(
            "# SHAP Publication Analysis\n\n"
            "Values refer to MiniLM embedding dimensions (latent axes), not tokens.\n"
            f"Aggregated over n={len(texts)} samples.\n",
            encoding="utf-8",
        )
    return result


def _bar_dims(dims, vals, out_path, title, color) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [str(d) for d in dims][::-1]
    ax.barh(labels, list(vals)[::-1], color=color, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("SHAP mass")
    ax.set_ylabel("Embedding dimension")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, Path(out_path))
