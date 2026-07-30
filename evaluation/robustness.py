"""
Robustness evaluation under controlled text perturbations.

Perturbations: typos, grammar/case noise, word-order shuffles,
synonym swaps, and light paraphrases. Measures accuracy degradation
relative to clean hybrid predictions.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from classification.adaptive_router import DEFAULT_ALPHA
from evaluation.benchmark import predict_with_alpha

RNG = random.Random(42)

_SYNONYMS = {
    "therefore": "thus",
    "because": "since",
    "similar": "alike",
    "according": "per",
    "observed": "noticed",
    "suggests": "indicates",
    "evidence": "support",
    "study": "research",
    "important": "significant",
    "show": "demonstrate",
}


def _typo(text: str) -> str:
    chars = list(text)
    letters = [i for i, c in enumerate(chars) if c.isalpha()]
    if len(letters) < 2:
        return text
    i = RNG.choice(letters)
    j = RNG.choice(letters)
    chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def _grammar_noise(text: str) -> str:
    # Mild case / article noise without destroying meaning.
    out = text
    if RNG.random() < 0.5:
        out = out[0].lower() + out[1:] if out else out
    out = re.sub(r"\bThe\b", "the", out, count=1)
    out = re.sub(r"\ba\b", "an", out, count=1) if " a " in out else out
    return out


def _word_order(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    i = RNG.randrange(0, len(words) - 1)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def _synonym(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        key = w.lower()
        if key in _SYNONYMS:
            sub = _SYNONYMS[key]
            return sub.capitalize() if w[0].isupper() else sub
        return w

    return re.sub(r"\b\w+\b", repl, text)


def _paraphrase(text: str) -> str:
    # Lightweight template paraphrase (no external LLM).
    t = text.strip()
    replacements = [
        (r"\btherefore\b", "as a result"),
        (r"\bThus\b", "Consequently"),
        (r"\bbecause\b", "due to the fact that"),
        (r"\bsimilar to\b", "comparable to"),
        (r"\bAccording to\b", "As stated by"),
    ]
    for pat, rep in replacements:
        t2 = re.sub(pat, rep, t, count=1, flags=re.I)
        if t2 != t:
            return t2
    return "In short, " + t[0].lower() + t[1:] if t else t


PERTURBATIONS: dict[str, Callable[[str], str]] = {
    "typo": _typo,
    "grammar": _grammar_noise,
    "word_order": _word_order,
    "synonym": _synonym,
    "paraphrase": _paraphrase,
}


def run_robustness_evaluation(
    test_csv: str | Path,
    *,
    out_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate hybrid accuracy under each perturbation family."""
    df = pd.read_csv(test_csv)
    label_col = "pramana_label" if "pramana_label" in df.columns else "label"
    texts = df["text"].astype(str).tolist()
    y_true = df[label_col].astype(str).tolist()
    if max_samples is not None:
        texts, y_true = texts[:max_samples], y_true[:max_samples]

    clean_preds = []
    for t in texts:
        pred, _, _ = predict_with_alpha(t, alpha=DEFAULT_ALPHA, adaptive=False)
        clean_preds.append(pred)
    clean_acc = sum(p == y for p, y in zip(clean_preds, y_true)) / len(y_true)

    rows: list[dict[str, Any]] = [
        {
            "perturbation": "clean",
            "accuracy": clean_acc,
            "delta_accuracy": 0.0,
            "relative_drop_pct": 0.0,
        }
    ]
    detail_rows: list[dict[str, Any]] = []

    for name, fn in PERTURBATIONS.items():
        preds = []
        for t, y in zip(texts, y_true):
            pt = fn(t)
            pred, _, _ = predict_with_alpha(pt, alpha=DEFAULT_ALPHA, adaptive=False)
            preds.append(pred)
            detail_rows.append(
                {
                    "perturbation": name,
                    "original": t,
                    "perturbed": pt,
                    "y_true": y,
                    "y_pred": pred,
                    "correct": pred == y,
                }
            )
        acc = sum(p == y for p, y in zip(preds, y_true)) / len(y_true)
        delta = acc - clean_acc
        rows.append(
            {
                "perturbation": name,
                "accuracy": acc,
                "delta_accuracy": delta,
                "relative_drop_pct": (100.0 * (-delta) / clean_acc) if clean_acc else 0.0,
            }
        )

    result = {
        "clean_accuracy": clean_acc,
        "n_samples": len(texts),
        "results": rows,
        "worst_perturbation": min(
            (r for r in rows if r["perturbation"] != "clean"),
            key=lambda r: r["accuracy"],
        )["perturbation"],
        "mean_drop": float(
            sum(-r["delta_accuracy"] for r in rows if r["perturbation"] != "clean")
            / max(len(PERTURBATIONS), 1)
        ),
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "robustness_results.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(out / "robustness_examples.csv", index=False)
        (out / "robustness_report.md").write_text(_rob_md(result), encoding="utf-8")
    return result


def _rob_md(result: dict[str, Any]) -> str:
    lines = [
        "# Robustness Evaluation Report",
        "",
        f"Clean hybrid accuracy: **{result['clean_accuracy']:.3f}** (n={result['n_samples']}).",
        f"Worst perturbation: **{result['worst_perturbation']}** "
        f"(mean absolute drop={result['mean_drop']:.3f}).",
        "",
        "| Perturbation | Accuracy | Δ Acc | Relative drop % |",
        "|--------------|----------|-------|-----------------|",
    ]
    for r in result["results"]:
        lines.append(
            f"| {r['perturbation']} | {r['accuracy']:.3f} | {r['delta_accuracy']:+.3f} | "
            f"{r['relative_drop_pct']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Perturbations are heuristic and API-free (no LLM paraphraser). "
            "They stress lexical cue sensitivity of the symbolic channel and "
            "embedding stability of the ML head.",
            "",
        ]
    )
    return "\n".join(lines)
