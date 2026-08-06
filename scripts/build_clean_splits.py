"""
Build a clean 3-way split (train / val / test) with group-disjoint constraints
and remove near-duplicate template rows.

Reviewer fixes addressed
------------------------
C-01 (CRITICAL): fusion weight alpha selected on VAL only; final test untouched.
C-03 (CRITICAL): near-duplicate template pairs isolated; group keys enforced.
MISS-2:          frozen split manifests written to dataset/split_manifests/.

Split strategy
--------------
REAL-WORLD rows (AAEC + IBM):
    Group-disjoint: GroupShuffleSplit on (essay_id | topic_group).
    Sizes: 70% train, 15% val, 15% final_test (approximate, group-level rounding).

SYNTHETIC rows:
    Group-disjoint: GroupShuffleSplit on template_family.
    BUT: rows tagged "generic_pad_augmentation" (near-duplicate padding vignettes)
    are isolated into a separate sidecar file and excluded from the main splits.
    Sizes: 70% train, 15% val, 15% final_test.

Both splits are then concatenated to form the final three-way split.
The final_test partition is written to dataset/clean_final_test.csv and should
NEVER be used for hyperparameter selection, alpha tuning, or calibration.

Usage (from project root):
    python scripts/build_clean_splits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "dataset" / "processed"
MANIFESTS = ROOT / "dataset" / "split_manifests"
MANIFESTS.mkdir(parents=True, exist_ok=True)

MERGED_WITH_GROUPS = PROCESSED / "merged_corpus_with_groups.csv"

# Output paths
OUT_TRAIN = PROCESSED / "split_train.csv"
OUT_VAL = PROCESSED / "split_val.csv"
OUT_TEST = ROOT / "dataset" / "clean_final_test.csv"
OUT_SIDECAR = PROCESSED / "near_dup_sidecar.csv"

RANDOM_STATE = 42
VAL_FRAC = 0.15   # of the non-test portion
TEST_FRAC = 0.15  # of all group-aware data

PRAMANA_LABELS = ["Pratyaksha", "Anumana", "Upamana", "Shabda"]
PAD_FAMILY = "generic_pad_augmentation"


def _group_key(row: pd.Series) -> str:
    """One consistent group key per row regardless of source."""
    if row["source"] == "aaec_reviewed" and row["essay_id"]:
        return "essay__" + str(row["essay_id"])
    if row["source"] == "ibm_reviewed" and row["topic_group"]:
        return "topic__" + str(row["topic_group"])
    if row["source"] == "nyaya_synthetic" and row["template_family"]:
        return "tmpl__" + str(row["template_family"])
    return "other__" + str(row.name)


def group_split(
    df: pd.DataFrame,
    group_col: str,
    test_frac: float,
    val_frac: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Group-disjoint 3-way split.
    Returns (train, val, test) DataFrames.
    """
    groups = df[group_col].values
    indices = np.arange(len(df))

    # Split off test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(indices, groups=groups))

    # Split off val from trainval
    trainval_groups = groups[trainval_idx]
    val_frac_of_trainval = val_frac / (1.0 - test_frac)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=random_state + 1)
    rel_train_idx, rel_val_idx = next(gss_val.split(trainval_idx, groups=trainval_groups))
    train_idx = trainval_idx[rel_train_idx]
    val_idx = trainval_idx[rel_val_idx]

    return (
        df.iloc[train_idx].copy(),
        df.iloc[val_idx].copy(),
        df.iloc[test_idx].copy(),
    )


def main() -> None:
    if not MERGED_WITH_GROUPS.is_file():
        raise FileNotFoundError(
            f"Missing: {MERGED_WITH_GROUPS}\n"
            "Run scripts/add_group_keys.py first."
        )

    df = pd.read_csv(MERGED_WITH_GROUPS)
    print(f"Loaded merged corpus: {len(df)} rows")

    # --- 1. Assign unified group key ---
    df["group_key"] = df.apply(_group_key, axis=1)

    # --- 2. Isolate near-duplicate padding vignettes ---
    pad_mask = df["template_family"] == PAD_FAMILY
    sidecar = df[pad_mask].copy()
    clean = df[~pad_mask].copy().reset_index(drop=True)
    print(f"\nNear-duplicate padding rows isolated: {len(sidecar)}")
    print(f"Clean corpus (no pads): {len(clean)}")
    sidecar.to_csv(OUT_SIDECAR, index=False)
    print(f"  Sidecar saved: {OUT_SIDECAR.relative_to(ROOT)}")

    # --- 3. Real-world and synthetic ---
    real = clean[clean["source"].isin(["aaec_reviewed", "ibm_reviewed"])].copy()
    synth = clean[clean["source"] == "nyaya_synthetic"].copy()
    print(f"\nReal-world rows: {len(real)}")
    print(f"Synthetic rows:  {len(synth)}")

    # --- 4. Group-disjoint splits ---
    r_train, r_val, r_test = group_split(real, "group_key", TEST_FRAC, VAL_FRAC, RANDOM_STATE)
    s_train, s_val, s_test = group_split(synth, "group_key", TEST_FRAC, VAL_FRAC, RANDOM_STATE)

    train = pd.concat([r_train, s_train], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)
    val = pd.concat([r_val, s_val], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)
    test = pd.concat([r_test, s_test], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)

    # --- 5. Save splits ---
    train.to_csv(OUT_TRAIN, index=False)
    val.to_csv(OUT_VAL, index=False)
    test.to_csv(OUT_TEST, index=False)

    print("\n" + "=" * 70)
    print("Split summary")
    print("=" * 70)
    for name, part in [("TRAIN", train), ("VAL", val), ("TEST (UNTOUCHED)", test)]:
        cls_dist = part["pramana_label"].value_counts()
        groups_n = part["group_key"].nunique()
        print(f"\n  {name}: {len(part)} rows, {groups_n} groups")
        for lbl in PRAMANA_LABELS:
            n = cls_dist.get(lbl, 0)
            print(f"    {lbl:12s}: {n}")

    # Verify no group overlap between splits
    train_groups = set(train["group_key"])
    val_groups = set(val["group_key"])
    test_groups = set(test["group_key"])
    tv_overlap = train_groups & val_groups
    tt_overlap = train_groups & test_groups
    vt_overlap = val_groups & test_groups
    print(f"\n  Group overlaps (must all be 0):")
    print(f"    train ∩ val:  {len(tv_overlap)}")
    print(f"    train ∩ test: {len(tt_overlap)}")
    print(f"    val ∩ test:   {len(vt_overlap)}")

    # --- 6. Write frozen manifests ---
    manifest_rows = []
    for split_name, part in [("train", train), ("val", val), ("test", test), ("sidecar_pad", sidecar)]:
        for _, row in part.iterrows():
            manifest_rows.append({
                "split": split_name,
                "source": row.get("source", ""),
                "group_key": row.get("group_key", ""),
                "pramana_label": row.get("pramana_label", ""),
                "text_hash": str(abs(hash(str(row.get("text", "")) [:80]))),
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFESTS / "full_split_manifest.csv", index=False)

    # Summary per source per split
    summary = manifest_df.groupby(["split", "source"])["pramana_label"].count().reset_index()
    summary.columns = ["split", "source", "rows"]
    summary.to_csv(MANIFESTS / "split_summary.csv", index=False)

    # Save registry metadata
    registry = {
        "split_script": "scripts/build_clean_splits.py",
        "random_state": RANDOM_STATE,
        "test_frac": TEST_FRAC,
        "val_frac_of_total": VAL_FRAC,
        "group_disjoint": True,
        "near_dup_pad_isolated": True,
        "rows": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
            "sidecar_pad": int(len(sidecar)),
        },
        "group_overlaps": {
            "train_val": int(len(tv_overlap)),
            "train_test": int(len(tt_overlap)),
            "val_test": int(len(vt_overlap)),
        },
        "alpha_selection_partition": "val",
        "final_evaluation_partition": "test",
        "note": (
            "The test partition (dataset/clean_final_test.csv) must NOT be used "
            "for hyperparameter selection, alpha tuning, router thresholds, "
            "calibration fitting, or model selection. Evaluate exactly once."
        ),
    }
    (MANIFESTS / "experiment_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )

    print(f"\n  Manifests saved to: {MANIFESTS.relative_to(ROOT)}/")
    print("    full_split_manifest.csv")
    print("    split_summary.csv")
    print("    experiment_registry.json")
    print("\n✅ Clean group-disjoint splits complete.")
    print(f"   UNTOUCHED test set: {OUT_TEST.relative_to(ROOT)}")
    print(f"   Use VAL ({OUT_VAL.relative_to(ROOT)}) for all hyperparameter selection.")


if __name__ == "__main__":
    main()
