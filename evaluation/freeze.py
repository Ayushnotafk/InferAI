"""
Reproducible artifact freeze for InferAI experiments.

Copies the current trained model, label encoder, SHAP background, and symbolic
rule configuration into a timestamped directory with a manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MODELS = _ROOT / "models"
_RULES = _ROOT / "resources" / "rules_v2.yaml"
_FREEZE_ROOT = _ROOT / "models" / "frozen"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_current_artifacts(
    *,
    tag: str | None = None,
    note: str = "Pre-research-upgrade baseline freeze",
) -> Path:
    """
    Snapshot model artifacts and rule config for reproducible evaluation.

    Returns path to the freeze directory.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}_{tag}" if tag else stamp
    dest = _FREEZE_ROOT / name
    dest.mkdir(parents=True, exist_ok=False)

    artifacts = [
        _MODELS / "nyaya_model.pkl",
        _MODELS / "label_encoder.pkl",
        _MODELS / "shap_background.npy",
    ]
    manifest_files: dict[str, dict[str, str]] = {}

    for src in artifacts:
        if src.is_file():
            shutil.copy2(src, dest / src.name)
            manifest_files[src.name] = {"sha256": _sha256(dest / src.name)}

    if _RULES.is_file():
        shutil.copy2(_RULES, dest / "rules_v2.yaml")
        manifest_files["rules_v2.yaml"] = {"sha256": _sha256(dest / "rules_v2.yaml")}

    manifest = {
        "created_utc": stamp,
        "note": note,
        "default_alpha": 0.2,
        "rule_count": 77,
        "files": manifest_files,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest
