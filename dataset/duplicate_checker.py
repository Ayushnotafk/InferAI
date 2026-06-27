from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz

# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = Path("dataset/raw/nyaya_dataset_merged.csv")
TEST_PATH = Path("dataset/test_set.csv")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

SIMILARITY_THRESHOLD = 95

# ============================================================
# Load datasets
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_texts = train_df["text"].astype(str).str.strip().tolist()
test_texts = test_df["text"].astype(str).str.strip().tolist()

# ============================================================
# Exact duplicates inside training dataset
# ============================================================

exact_duplicates = train_df.duplicated(subset=["text"]).sum()

# ============================================================
# Exact overlap between train and held-out
# ============================================================

train_set = set(train_texts)
test_set = set(test_texts)

exact_overlap = train_set.intersection(test_set)

# ============================================================
# Near duplicates
# ============================================================

near_duplicates = []

remaining_train = train_set - exact_overlap
remaining_test = test_set - exact_overlap

for train_sentence in remaining_train:

    for test_sentence in remaining_test:

        similarity = fuzz.ratio(train_sentence, test_sentence)

        if similarity >= SIMILARITY_THRESHOLD:

            near_duplicates.append({
                "Train Sentence": train_sentence,
                "Test Sentence": test_sentence,
                "Similarity": similarity
            })

# ============================================================
# Console Output
# ============================================================

print("=" * 60)
print("DUPLICATE ANALYSIS REPORT")
print("=" * 60)

print(f"Training Samples               : {len(train_df)}")
print(f"Held-out Samples               : {len(test_df)}")
print(f"Exact Duplicates (Training)    : {exact_duplicates}")
print(f"Exact Train-Test Overlap       : {len(exact_overlap)}")
print(f"Near Duplicates (>=95%)        : {len(near_duplicates)}")

# ============================================================
# Save Summary CSV
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Training Samples",

        "Held-out Samples",

        "Exact Duplicates (Training)",

        "Exact Train-Test Overlap",

        "Near Duplicates (>=95%)"

    ],

    "Value": [

        len(train_df),

        len(test_df),

        int(exact_duplicates),

        len(exact_overlap),

        len(near_duplicates)

    ]

})

summary.to_csv(
    REPORT_DIR / "duplicate_summary.csv",
    index=False
)

# ============================================================
# Save Near Duplicate Details
# ============================================================

if near_duplicates:

    pd.DataFrame(near_duplicates).to_csv(

        REPORT_DIR / "near_duplicates.csv",

        index=False

    )

# ============================================================
# Markdown Report
# ============================================================

with open(REPORT_DIR / "duplicate_report.md", "w", encoding="utf-8") as file:

    file.write("# Duplicate Analysis Report\n\n")

    file.write(f"- Training Samples: {len(train_df)}\n")

    file.write(f"- Held-out Samples: {len(test_df)}\n")

    file.write(f"- Exact Duplicates (Training): {exact_duplicates}\n")

    file.write(f"- Exact Train-Test Overlap: {len(exact_overlap)}\n")

    file.write(f"- Near Duplicates (>=95%): {len(near_duplicates)}\n")

print("\nReports generated successfully inside reports/")