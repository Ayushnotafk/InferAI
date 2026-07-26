"""
SHAP explanations for the trained Logistic Regression classifier on Sentence-BERT embeddings.

Uses ``shap.LinearExplainer``, which is exact for linear models and appropriate here because
sklearn's ``LogisticRegression`` is linear in the embedding space. Individual dimensions are
latent semantic axes of the MiniLM vector rather than human-readable tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import shap

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings

_model = None
_label_encoder = None
_explainer = None
_background: np.ndarray | None = None


def _project_root() -> Path:
    return _ROOT


def load_classifier():
    global _model, _label_encoder
    if _model is None:
        _model = joblib.load(_project_root() / "models" / "infer_model.pkl")
    if _label_encoder is None:
        _label_encoder = joblib.load(_project_root() / "models" / "label_encoder.pkl")
    return _model, _label_encoder


def load_background_matrix() -> np.ndarray:
    global _background
    if _background is None:
        path = _project_root() / "models" / "shap_background.npy"
        if not path.is_file():
            raise FileNotFoundError(
                "Missing models/shap_background.npy. Run classification/train_model.py after updating the trainer."
            )
        _background = np.load(path)
    return _background


def get_explainer():
    global _explainer
    if _explainer is None:
        model, _ = load_classifier()
        background = load_background_matrix()
        _explainer = shap.LinearExplainer(model, background)
    return _explainer


def _shap_vector_for_class(shap_out, class_index: int) -> np.ndarray:
    """Normalize SHAP outputs to a 1D vector of length n_features for ``class_index``."""
    arr = np.asarray(shap_out)
    if arr.ndim == 3:
        # Common shape: (n_samples, n_features, n_classes)
        return arr[0, :, int(class_index)]
    if isinstance(shap_out, list):
        return np.asarray(shap_out[int(class_index)])[0].reshape(-1)
    if arr.ndim == 2:
        return arr[0].reshape(-1)
    return arr.reshape(-1)


def top_embedding_contributions(
    shap_vector: np.ndarray, top_k: int = 12
) -> list[dict[str, float | int]]:
    """Return the embedding dimensions with largest absolute SHAP mass."""
    flat = np.asarray(shap_vector).reshape(-1)
    order = np.argsort(-np.abs(flat))[:top_k]
    return [{"embedding_dim": int(i), "shap_value": float(flat[i])} for i in order]


def explain_text(
    text: str, top_k: int = 12
) -> dict:
    """
    Embed ``text``, run SHAP for the predicted class, and return ranked latent dimensions.

    Returns
    -------
    dict
        predicted label, optional class index, SHAP summary for the predicted output,
        and a short note for report-style write-ups.
    """
    model, label_encoder = load_classifier()
    explainer = get_explainer()

    x = generate_embeddings([text])
    pred = model.predict(x)[0]
    label = label_encoder.inverse_transform(np.array([pred]))[0]

    shap_out = explainer.shap_values(x)
    vec = _shap_vector_for_class(shap_out, int(pred))

    return {
        "predicted_pramana": label,
        "predicted_class_index": int(pred),
        "top_embedding_contributions": top_embedding_contributions(vec, top_k=top_k),
        "note": (
            "SHAP values are with respect to MiniLM embedding dimensions (384-D latent space), "
            "not individual words."
        ),
    }


def explain_embedding(embedding_2d: np.ndarray, top_k: int = 12) -> dict:
    """Explain a precomputed embedding row of shape (1, n_features)."""
    model, label_encoder = load_classifier()
    explainer = get_explainer()

    x = np.asarray(embedding_2d, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    pred = model.predict(x)[0]
    label = label_encoder.inverse_transform(np.array([pred]))[0]
    shap_out = explainer.shap_values(x)
    vec = _shap_vector_for_class(shap_out, int(pred))

    return {
        "predicted_pramana": label,
        "predicted_class_index": int(pred),
        "top_embedding_contributions": top_embedding_contributions(vec, top_k=top_k),
        "note": (
            "SHAP values are with respect to MiniLM embedding dimensions (384-D latent space), "
            "not individual words."
        ),
    }


if __name__ == "__main__":
    sample = (
        "Smoke is rising from the hill, therefore there must be fire."
    )
    out = explain_text(sample)
    print("Example:", sample)
    for k, v in out.items():
        print(k, ":", v)
