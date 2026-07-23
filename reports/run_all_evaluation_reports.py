"""
Run the full InferAI research evaluation report pipeline.

Order:
  1. Dataset statistics (no model required)
  2. Lexical diversity (no model required)
  3. Kappa analysis (no model required)
  4. Held-out evaluation (requires trained model)
  5. Confusion matrix analysis (requires trained model)
  6. Calibration report (requires trained model)
  7. Data leakage audit (no model required for exact; embeddings for near-dup)
  8. Professor review consolidation

Usage (from project root):
    python reports/run_all_evaluation_reports.py

Train first if model artifacts are missing:
    python classification/train_model.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    ("Dataset statistics", _ROOT / "reports" / "generate_dataset_statistics.py", False),
    ("Lexical diversity", _ROOT / "reports" / "generate_lexical_diversity.py", False),
    ("Kappa analysis", _ROOT / "calculate_kappa.py", False),
    ("Held-out evaluation", _ROOT / "reports" / "generate_heldout_evaluation.py", True),
    ("Confusion matrix analysis", _ROOT / "reports" / "generate_confusion_matrix_analysis.py", True),
    ("Calibration report", _ROOT / "reports" / "generate_calibration_report.py", True),
    ("Data leakage audit", _ROOT / "reports" / "check_data_leakage.py", False),
    ("Professor review response", _ROOT / "reports" / "generate_professor_review_response.py", False),
]


def _model_ready() -> bool:
    return (_ROOT / "models" / "nyaya_model.pkl").is_file()


def main() -> None:
    print("InferAI — research evaluation pipeline\n" + "=" * 50)
    if not _model_ready():
        print("Warning: models/nyaya_model.pkl not found. Model-dependent steps will be skipped.")
        print("Run: python classification/train_model.py\n")

    for name, script, needs_model in SCRIPTS:
        if needs_model and not _model_ready():
            print(f"[SKIP] {name} — model not trained")
            continue
        print(f"\n>>> {name}")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(_ROOT),
            check=False,
        )
        if result.returncode != 0:
            print(f"[FAIL] {name} exited with code {result.returncode}")
            sys.exit(result.returncode)

    print("\n" + "=" * 50)
    print("All reports generated under reports/")


if __name__ == "__main__":
    main()
