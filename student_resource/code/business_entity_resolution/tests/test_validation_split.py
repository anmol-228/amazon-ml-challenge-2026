"""Unit tests for the frozen validation split builder (src/validation_split.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from validation_split import (  # noqa: E402
    SplitConfig,
    assign_split,
    build_strata,
    manifest_sha256,
    match_bucket,
)


def _synthetic(n_per_stratum: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_gt = []
    rows_s1 = []
    i = 0
    for country in ("US", "India"):
        for count in (0, 1, 2, 3, 5):
            for _ in range(n_per_stratum):
                sid = f"S1-{i:06d}"
                rows_s1.append({"entity_id": sid, "country": country})
                rows_gt.append({"source1_entity_id": sid, "match_count": count})
                i += 1
    return pd.DataFrame(rows_gt), pd.DataFrame(rows_s1)


def test_match_bucket():
    assert match_bucket(0) == "0"
    assert match_bucket(1) == "1"
    assert match_bucket(2) == "2+"
    assert match_bucket(11) == "2+"


def test_every_entity_appears_exactly_once():
    gt, s1 = _synthetic()
    strata = build_strata(gt, s1)
    split = assign_split(strata, SplitConfig(seed=1))
    assert len(split) == len(gt)
    assert split["source1_entity_id"].is_unique


def test_development_validation_partition_is_complete_and_disjoint():
    gt, s1 = _synthetic()
    strata = build_strata(gt, s1)
    split = assign_split(strata, SplitConfig(seed=1))

    dev_ids = set(split.loc[split["split"] == "development", "source1_entity_id"])
    val_ids = set(split.loc[split["split"] == "validation", "source1_entity_id"])
    all_ids = set(gt["source1_entity_id"])

    assert dev_ids & val_ids == set()
    assert dev_ids | val_ids == all_ids


def test_deterministic_under_fixed_seed():
    gt, s1 = _synthetic()
    strata = build_strata(gt, s1)
    split_a = assign_split(strata, SplitConfig(seed=7))
    split_b = assign_split(strata, SplitConfig(seed=7))
    merged = split_a.merge(split_b, on="source1_entity_id", suffixes=("_a", "_b"))
    assert (merged["split_a"] == merged["split_b"]).all()


def test_different_seeds_can_differ():
    gt, s1 = _synthetic()
    strata = build_strata(gt, s1)
    split_a = assign_split(strata, SplitConfig(seed=1))
    split_b = assign_split(strata, SplitConfig(seed=2))
    merged = split_a.merge(split_b, on="source1_entity_id", suffixes=("_a", "_b"))
    assert not (merged["split_a"] == merged["split_b"]).all()


def test_manifest_hash_reproducible_and_order_independent():
    gt, s1 = _synthetic()
    strata = build_strata(gt, s1)
    split = assign_split(strata, SplitConfig(seed=3))
    h1 = manifest_sha256(split)
    h2 = manifest_sha256(split.sample(frac=1.0, random_state=99))  # shuffled rows
    assert h1 == h2

    split_again = assign_split(strata, SplitConfig(seed=3))
    assert manifest_sha256(split_again) == h1


def test_stratum_proportions_close_to_target_fraction():
    gt, s1 = _synthetic(n_per_stratum=200)
    strata = build_strata(gt, s1)
    split = assign_split(strata, SplitConfig(seed=5, validation_fraction=0.20))
    for stratum, group in split.groupby("stratum"):
        val_frac = (group["split"] == "validation").mean()
        assert abs(val_frac - 0.20) < 0.03  # small groups -> allow rounding slack


def test_component_map_keeps_grouped_entities_together():
    gt, s1 = _synthetic()
    strata = build_strata(gt, s1)
    ids = strata["source1_entity_id"].tolist()
    # force the first two entities into one artificial component
    component_map = pd.Series({ids[0]: "COMP-0", ids[1]: "COMP-0"})
    split = assign_split(strata, SplitConfig(seed=11), component_map=component_map)
    row0 = split.loc[split["source1_entity_id"] == ids[0], "split"].iloc[0]
    row1 = split.loc[split["source1_entity_id"] == ids[1], "split"].iloc[0]
    assert row0 == row1
