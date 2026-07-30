"""
Research-grade evaluation package for InferAI.

Modules remain independently importable; this package re-exports the most
common entry points only.
"""

from evaluation.benchmark import run_ablation_study, run_benchmark
from evaluation.freeze import freeze_current_artifacts

__all__ = [
    "run_benchmark",
    "run_ablation_study",
    "freeze_current_artifacts",
]
