"""
Official InferAI dataset counts (canonical project values).

Used by documentation generators for repository-wide consistency.
"""

from __future__ import annotations

SYNTHETIC_RAW = 4200
SYNTHETIC_USED = 611
AAEC_REVIEWED = 650
IBM_REVIEWED = 600
MERGED_CORPUS = 1861
REAL_WORLD_REVIEWED = AAEC_REVIEWED + IBM_REVIEWED  # 1250
RETRAIN_DUPES_REMOVED = 3589
MERGE_RAW_BEFORE_DEDUP = MERGED_CORPUS + RETRAIN_DUPES_REMOVED  # 5450
TRAIN_VAL_RANDOM_STATE = 42
TRAIN_FRACTION = 0.8
