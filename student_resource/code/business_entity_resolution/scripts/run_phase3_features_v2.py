"""Phase-3 v2 feature computation: identical logic to run_phase3_features.py
(reused directly, not duplicated), pointed at the v2 (P3/unpruned + hard-
negative-mixed) labeled candidate files from run_phase3_prune_label_v2.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3_features import build_train_lookup_tables, process_file, train_name_rarity_counts  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PRUNED_DIR = PROJECT_ROOT / "experiments" / "phase3" / "pruned_v2"
OUT_DIR = PROJECT_ROOT / "experiments" / "phase3" / "features_v2"


def main() -> None:
    t_start = time.time()
    s1_attrs, target_attrs = build_train_lookup_tables()
    name_rarity_counts = train_name_rarity_counts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ("calibration_holdout", "dev_eval", "matcher_train"):
        in_path = PRUNED_DIR / f"{split_name}.parquet"
        out_path = OUT_DIR / f"{split_name}.parquet"
        process_file(in_path, out_path, s1_attrs, target_attrs, name_rarity_counts)
    print(f"[features_v2] ALL DONE in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
