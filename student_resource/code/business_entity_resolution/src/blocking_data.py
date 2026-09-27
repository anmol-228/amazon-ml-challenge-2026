"""Shared data-loading helpers for Phase-2 blocking experiments.

Thin wrappers around `data_io.py` that add the one thing every blocking
experiment needs and none of the Phase-1 code provides: the frozen
`validation_v1` split joined onto Source-1, and a `s1_entity_id ->
frozenset(true target ids)` map built from `train_ground_truth.tsv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_io import TRAIN_PATHS, load_ground_truth, load_source_table, parse_matched_ids

PROJECT_ROOT = Path(__file__).resolve().parents[4]
VALIDATION_SPLIT_PATH = PROJECT_ROOT / "experiments" / "splits" / "validation_v1.tsv"


@dataclass
class TrainData:
    source1: pd.DataFrame  # entity_id, business_name, business_address, country, split
    source2: pd.DataFrame
    source3: pd.DataFrame
    ground_truth: pd.DataFrame  # source1_entity_id, matched_entity_ids
    truth_by_s1: dict[str, frozenset[str]]


def load_train_data() -> TrainData:
    s1 = load_source_table(TRAIN_PATHS["source1"])
    s2 = load_source_table(TRAIN_PATHS["source2"])
    s3 = load_source_table(TRAIN_PATHS["source3"])
    gt = load_ground_truth(TRAIN_PATHS["ground_truth"])

    split_df = pd.read_csv(VALIDATION_SPLIT_PATH, sep="\t", dtype=str)
    s1 = s1.merge(split_df, left_on="entity_id", right_on="source1_entity_id", how="left")
    s1 = s1.drop(columns=["source1_entity_id"])

    truth_by_s1 = {
        row.source1_entity_id: frozenset(parse_matched_ids(row.matched_entity_ids))
        for row in gt.itertuples()
    }

    return TrainData(source1=s1, source2=s2, source3=s3, ground_truth=gt, truth_by_s1=truth_by_s1)


def development_s1_ids(data: TrainData) -> list[str]:
    return data.source1.loc[data.source1["split"] == "development", "entity_id"].tolist()


def validation_s1_ids(data: TrainData) -> list[str]:
    return data.source1.loc[data.source1["split"] == "validation", "entity_id"].tolist()


def deterministic_pilot_sample(
    data: TrainData, n: int, seed: int = 42, split: str = "development"
) -> list[str]:
    """A fixed, stratified (country x singleton x match-bucket-coarse)
    pilot sample of S1 entity IDs from the requested split, for EXP-B000.
    """
    import numpy as np

    from validation_split import match_bucket

    s1 = data.source1.loc[data.source1["split"] == split, ["entity_id", "country"]].copy()
    s1["match_count"] = s1["entity_id"].map(
        lambda e: len(data.truth_by_s1.get(e, frozenset()))
    )
    s1["singleton"] = s1["match_count"] == 0
    s1["bucket"] = s1["match_count"].map(match_bucket)
    s1["stratum"] = (
        s1["country"].astype(str) + "|" + s1["singleton"].astype(str) + "|" + s1["bucket"].astype(str)
    )

    rng = np.random.RandomState(seed)
    per_stratum = max(1, n // max(1, s1["stratum"].nunique()))
    sampled_parts = []
    for _, group in s1.groupby("stratum"):
        take = min(per_stratum, len(group))
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        sampled_parts.append(s1.loc[idx])
    sample = pd.concat(sampled_parts, axis=0)
    if len(sample) > n:
        idx = rng.choice(sample.index.to_numpy(), size=n, replace=False)
        sample = sample.loc[idx]
    return sample["entity_id"].tolist()
