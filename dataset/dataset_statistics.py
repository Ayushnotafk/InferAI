import pandas as pd
from pathlib import Path

# ==========================================================
# Load merged dataset
# ==========================================================

dataset_path = Path("dataset/raw/nyaya_dataset_merged.csv")

df = pd.read_csv(dataset_path)

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

total_samples = len(df)
unique_samples = df["text"].nunique()
duplicate_samples = total_samples - unique_samples

print(f"Total Samples          : {total_samples}")
print(f"Unique Text Samples    : {unique_samples}")
print(f"Duplicate Text Samples : {duplicate_samples}")

print("\n")

# ==========================================================
# Pramana Distribution
# ==========================================================

print("=" * 60)
print("PRAMANA LABEL DISTRIBUTION")
print("=" * 60)

pramana = (
    df["pramana_label"]
    .value_counts()
    .rename_axis("Pramana")
    .reset_index(name="Count")
)

pramana["Percentage"] = (
    pramana["Count"] / total_samples * 100
).round(2)

print(pramana)

print("\n")

# ==========================================================
# Strength Distribution
# ==========================================================

print("=" * 60)
print("STRENGTH LABEL DISTRIBUTION")
print("=" * 60)

strength = (
    df["strength_label"]
    .value_counts()
    .rename_axis("Strength")
    .reset_index(name="Count")
)

strength["Percentage"] = (
    strength["Count"] / total_samples * 100
).round(2)

print(strength)

print("\n")

# ==========================================================
# Save reports
# ==========================================================

reports = Path("reports")
reports.mkdir(exist_ok=True)

summary = pd.DataFrame({
    "Metric": [
        "Total Samples",
        "Unique Text Samples",
        "Duplicate Text Samples"
    ],
    "Value": [
        total_samples,
        unique_samples,
        duplicate_samples
    ]
})

summary.to_csv(
    reports / "dataset_summary.csv",
    index=False
)

pramana.to_csv(
    reports / "pramana_distribution.csv",
    index=False
)

strength.to_csv(
    reports / "strength_distribution.csv",
    index=False
)

print("=" * 60)
print("Reports Generated Successfully")
print("=" * 60)

print("dataset_summary.csv")
print("pramana_distribution.csv")
print("strength_distribution.csv")