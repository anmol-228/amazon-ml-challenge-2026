"""Frozen, leakage-safe train/validation split construction.

Phase 1 measured (see experiments/audit_cache/target_reuse_stats.json) that
no Source-2/Source-3 target ID is referenced by more than one Source-1
ground-truth entity in the training set (max_s1_degree_for_any_target == 1).
There is therefore no target-sharing leakage between Source-1 entities, and
a stratified split at the Source-1-entity level is leakage-safe on its own -
no connected-component grouping is required. This module still accepts an
optional `component_map` so a future re-audit that finds reuse can force
grouped entities into the same split without changing the call sites.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    seed: int = 42
    validation_fraction: float = 0.20
    strata_columns: tuple[str, ...] = ("country", "singleton", "match_bucket")


def match_bucket(match_count: int) -> str:
    if match_count == 0:
        return "0"
    if match_count == 1:
        return "1"
    return "2+"


def build_strata(
    ground_truth: pd.DataFrame,
    source1: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per Source-1 entity with its stratification key.

    `ground_truth` needs columns [source1_entity_id, match_count].
    `source1` needs columns [entity_id, country].
    """
    s1_country = source1.set_index("entity_id")["country"]
    df = ground_truth[["source1_entity_id", "match_count"]].copy()
    df["country"] = df["source1_entity_id"].map(s1_country)
    df["singleton"] = df["match_count"] == 0
    df["match_bucket"] = df["match_count"].map(match_bucket)
    df["stratum"] = (
        df["country"].astype(str)
        + "|"
        + df["singleton"].astype(str)
        + "|"
        + df["match_bucket"].astype(str)
    )
    return df


def assign_split(
    strata_df: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
    component_map: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns [source1_entity_id, split].

    Grouping unit is the Source-1 entity itself unless `component_map`
    (source1_entity_id -> component_id) is supplied, in which case every
    member of a component is assigned to the same split (see module
    docstring: not needed given the measured zero-target-reuse result, but
    kept so a future re-audit can enable it without an API change).
    """
    df = strata_df.copy()
    if component_map is not None:
        df["group_id"] = df["source1_entity_id"].map(component_map).fillna(df["source1_entity_id"])
    else:
        df["group_id"] = df["source1_entity_id"]

    # One representative stratum per group (components are expected to be
    # trivial/absent here; when present, use the first member's stratum).
    group_stratum = df.groupby("group_id")["stratum"].first()

    rng = np.random.RandomState(config.seed)
    assignments: dict[str, str] = {}
    for stratum, group_ids in group_stratum.groupby(group_stratum).groups.items():
        ids = list(group_ids)
        rng.shuffle(ids)
        n_val = round(len(ids) * config.validation_fraction)
        val_ids = set(ids[:n_val])
        for gid in ids:
            assignments[gid] = "validation" if gid in val_ids else "development"

    df["split"] = df["group_id"].map(assignments)
    return df[["source1_entity_id", "split", "stratum"]]


def manifest_sha256(split_df: pd.DataFrame) -> str:
    """Deterministic hash of the frozen split (order-independent)."""
    ordered = split_df.sort_values("source1_entity_id")
    payload = "\n".join(f"{r.source1_entity_id}\t{r.split}" for r in ordered.itertuples())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
