#!/usr/bin/env python3
"""Freeze current InferAI model + symbolic rules for reproducible experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.freeze import freeze_current_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze InferAI artifacts")
    parser.add_argument("--tag", default="baseline", help="Optional freeze tag")
    parser.add_argument("--note", default="Pre-research-upgrade baseline freeze")
    args = parser.parse_args()
    dest = freeze_current_artifacts(tag=args.tag, note=args.note)
    print(f"Frozen to: {dest}")


if __name__ == "__main__":
    main()
