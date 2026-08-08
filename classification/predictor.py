from pathlib import Path
import joblib
import numpy as np

from classification.embedder import generate_embeddings

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_model = None
_label_encoder = None


def _get_classifier():
    global _model, _label_encoder
    if _model is None:
        _model = joblib.load(_MODELS_DIR / "infer_model.pkl")
    if _label_encoder is None:
        _label_encoder = joblib.load(_MODELS_DIR / "label_encoder.pkl")
    return _model, _label_encoder


def _predict_core(text: str):
    model, label_encoder = _get_classifier()
    embedding = generate_embeddings([text])
    probabilities = model.predict_proba(embedding)[0]
    prediction = int(np.argmax(probabilities))
    label = label_encoder.inverse_transform(np.array([prediction]))[0]
    confidence = float(max(probabilities) * 100.0)
    classes = label_encoder.classes_.tolist()
    return label, confidence, embedding, probabilities, classes


def predict_pramana(text, return_embedding=False):
    label, confidence, embedding, _, _ = _predict_core(text)
    if return_embedding:
        return label, confidence, embedding
    return label, confidence


def predict_pramana_detailed(text: str) -> dict:
    """
    Return ML label, scalar confidence, embedding, full softmax, and class names.

    ``classes[i]`` aligns with ``probabilities[i]``.
    """
    label, confidence, embedding, proba, classes = _predict_core(text)
    return {
        "ml_label": label,
        "ml_confidence": confidence,
        "embedding": embedding,
        "probabilities": np.asarray(proba, dtype=np.float64),
        "classes": classes,
    }

