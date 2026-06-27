import math
from pathlib import Path

import pandas as pd
import re

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

DATASET_PATH = Path("dataset/raw/nyaya_dataset_merged.csv")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------

def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    return df


# -------------------------------------------------------
# Tokenize
# -------------------------------------------------------

def tokenize(text):
    """
    Tokenize text into lowercase words.
    """

    return re.findall(r"\b\w+\b", str(text).lower())

# -------------------------------------------------------
# Token statistics
# -------------------------------------------------------

def get_all_tokens(texts):

    all_tokens = []

    sentence_lengths = []

    for sentence in texts:

        tokens = tokenize(sentence)

        all_tokens.extend(tokens)

        sentence_lengths.append(len(tokens))

    return all_tokens, sentence_lengths


# -------------------------------------------------------
# Global TTR
# -------------------------------------------------------

def calculate_ttr(tokens):

    if len(tokens) == 0:
        return 0

    vocabulary = len(set(tokens))

    return vocabulary / len(tokens)


# -------------------------------------------------------
# Guiraud's R
# -------------------------------------------------------

def calculate_guiraud(tokens):

    if len(tokens) == 0:
        return 0

    vocabulary = len(set(tokens))

    return vocabulary / math.sqrt(len(tokens))


# -------------------------------------------------------
# Moving Average TTR
# -------------------------------------------------------

def calculate_mattr(tokens, window_size=50):

    if len(tokens) < window_size:
        return calculate_ttr(tokens)

    values = []

    for i in range(len(tokens) - window_size + 1):

        window = tokens[i:i + window_size]

        values.append(calculate_ttr(window))

    return sum(values) / len(values)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    df = load_dataset()

    texts = df["text"].astype(str)

    tokens, sentence_lengths = get_all_tokens(texts)

    vocabulary = len(set(tokens))

    total_tokens = len(tokens)

    results = {

    "Documents": int(len(texts)),

    "Total Tokens": int(total_tokens),

    "Unique Word Types": int(vocabulary),

    "Global TTR": round(calculate_ttr(tokens), 4),

    "MATTR (Window=50)": round(calculate_mattr(tokens), 4),

    "Guiraud R": round(calculate_guiraud(tokens), 4),

    "Average Tokens/Sentence": round(sum(sentence_lengths) / len(sentence_lengths), 2),

    "Minimum Tokens": int(min(sentence_lengths)),

    "Maximum Tokens": int(max(sentence_lengths))

}

    print("=" * 60)

    print("LEXICAL DIVERSITY REPORT")

    print("=" * 60)

    for key, value in results.items():

        print(f"{key:<30}: {value}")

    report_df = pd.DataFrame(
        list(results.items()),
        columns=["Metric", "Value"]
    )

    report_df.to_csv(
        REPORT_DIR / "lexical_diversity.csv",
        index=False
    )

    with open(REPORT_DIR / "lexical_diversity.md", "w", encoding="utf-8") as f:

        f.write("# Lexical Diversity Report\n\n")

        for key, value in results.items():

            f.write(f"- **{key}:** {value}\n")

    print("\nReports saved inside reports/")


if __name__ == "__main__":

    main()