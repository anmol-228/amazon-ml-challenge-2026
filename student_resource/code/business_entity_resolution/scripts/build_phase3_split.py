"""Phase-3 precondition: matcher_train / dev_eval / calibration_holdout split.

Built strictly INSIDE validation_v1's `development` partition (never touches
`validation`). Entity-level, stratified by (country, singleton, match_bucket),
same algorithm as validation_split.py's assign_split but split three ways
instead of two. Seed 42 (same project-wide convention).

Fractions: 70% matcher_train / 15% dev_eval / 15% calibration_holdout of
`development` (1,765,456 entities), chosen to leave the majority of labeled
data for GBDT training while keeping dev_eval and calibration_holdout large
enough (~265k entities each) for a stable macro-F0.5 read given the ~2.27e-6
per-entity score granularity documented in VALIDATION_DESIGN.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from blocking_data import load_train_data  # noqa: E402
from validation_split import build_strata, match_bucket  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = PROJECT_ROOT / "experiments" / "splits"
OUT_PATH = OUT_DIR / "phase3_split_v1.tsv"
META_PATH = OUT_DIR / "phase3_split_v1_metadata.md"

SEED = 42
FRACTIONS = {"matcher_train": 0.70, "dev_eval": 0.15, "calibration_holdout": 0.15}


def three_way_assign(strata_df: pd.DataFrame, seed: int, fractions: dict[str, float]) -> pd.DataFrame:
    assert abs(sum(fractions.values()) - 1.0) < 1e-9
    rng = np.random.RandomState(seed)
    assignments: dict[str, str] = {}
    for stratum, idx in strata_df.groupby("stratum").groups.items():
        ids = strata_df.loc[idx, "source1_entity_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_dev_eval = round(n * fractions["dev_eval"])
        n_calib = round(n * fractions["calibration_holdout"])
        dev_eval_ids = ids[:n_dev_eval]
        calib_ids = ids[n_dev_eval : n_dev_eval + n_calib]
        train_ids = ids[n_dev_eval + n_calib :]
        for eid in dev_eval_ids:
            assignments[eid] = "dev_eval"
        for eid in calib_ids:
            assignments[eid] = "calibration_holdout"
        for eid in train_ids:
            assignments[eid] = "matcher_train"
    out = strata_df.copy()
    out["phase3_split"] = out["source1_entity_id"].map(assignments)
    return out[["source1_entity_id", "phase3_split", "stratum"]]


def manifest_sha256(df: pd.DataFrame) -> str:
    ordered = df.sort_values("source1_entity_id")
    payload = "\n".join(f"{r.source1_entity_id}\t{r.phase3_split}" for r in ordered.itertuples())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    data = load_train_data()
    dev_s1 = data.source1.loc[data.source1["split"] == "development"].copy()
    dev_ids = set(dev_s1["entity_id"])

    gt = data.ground_truth.copy()
    gt = gt[gt["source1_entity_id"].isin(dev_ids)].copy()
    gt["match_count"] = gt["source1_entity_id"].map(
        lambda e: len(data.truth_by_s1.get(e, frozenset()))
    )
    # Every development S1 entity must appear in the strata table, including
    # rows that have no ground_truth.tsv row at all (true singletons with an
    # empty match set are still listed in ground_truth per Phase-1 audit, but
    # guard defensively rather than assume).
    missing = dev_ids - set(gt["source1_entity_id"])
    if missing:
        extra = pd.DataFrame({"source1_entity_id": list(missing), "match_count": 0})
        gt = pd.concat([gt, extra], axis=0, ignore_index=True)

    strata = build_strata(gt, data.source1)
    assert set(strata["source1_entity_id"]) == dev_ids

    split_df = three_way_assign(strata, SEED, FRACTIONS)
    assert set(split_df["source1_entity_id"]) == dev_ids
    assert split_df["source1_entity_id"].is_unique

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_df[["source1_entity_id", "phase3_split"]].to_csv(OUT_PATH, sep="\t", index=False)

    counts = split_df["phase3_split"].value_counts().to_dict()
    by_stratum = (
        split_df.groupby(["stratum", "phase3_split"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    sha = manifest_sha256(split_df)

    meta = {
        "seed": SEED,
        "fractions": FRACTIONS,
        "n_development_entities": len(dev_ids),
        "counts": counts,
        "manifest_sha256": sha,
        "by_stratum_sample": dict(list(by_stratum.items())[:10]),
        "n_strata": len(by_stratum),
    }
    with open(OUT_DIR / "phase3_split_v1_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    with open(META_PATH, "w", encoding="utf-8") as fh:
        fh.write("# Phase-3 split (matcher_train / dev_eval / calibration_holdout)\n\n")
        fh.write(f"Built strictly inside validation_v1's `development` partition "
                 f"({len(dev_ids):,} entities); `validation` untouched.\n\n")
        fh.write(f"Seed: {SEED}. Fractions: {FRACTIONS}.\n\n")
        fh.write(f"Counts: {counts}\n\n")
        fh.write(f"Manifest SHA-256 (order-independent): `{sha}`\n\n")
        fh.write(f"Stratification key: (country, singleton, match_bucket), same as validation_v1. "
                 f"{len(by_stratum)} strata.\n")

    print("counts:", counts)
    print("sha256:", sha)
    print("wrote:", OUT_PATH)


if __name__ == "__main__":
    main()
