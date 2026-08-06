"""Shared utilities for InferAI evaluation scripts."""
from __future__ import annotations

import pandas as pd


def df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    """Convert a DataFrame to a markdown table string without requiring tabulate."""
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows_md = []
    for _, row in df.iterrows():
        rows_md.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows_md)
