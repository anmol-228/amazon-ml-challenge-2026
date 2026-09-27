"""Deterministic exact-match sanity route (Phase-2 mission spec §12).

Cheap, high-specificity candidate generation from normalized-field equality
only -- no domain dictionaries, no country-specific mappings, no automatic
promotion to a final match decision. Two relationships, each independently
flagged so downstream analysis can tell which fired:

  - `route_exact_name_address`: normalized (name, address) exact equality.
  - `route_exact_name`: normalized name exact equality alone (address
    ignored), which is far less specific and mainly useful as a sanity/
    collision-size check, not a trusted candidate source on its own.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from blocking_normalization import normalize_field


def _grouped_postings(entity_ids: list[str], keys: list[str]) -> dict[str, list[str]]:
    postings: dict[str, list[str]] = defaultdict(list)
    for eid, key in zip(entity_ids, keys):
        if key:
            postings[key].append(eid)
    return postings


def exact_candidates(
    s1_ids: list[str],
    s1_names: list[str],
    s1_addresses: list[str],
    target_ids: list[str],
    target_names: list[str],
    target_addresses: list[str],
) -> pd.DataFrame:
    """Return candidate edges with per-relationship boolean flags.

    Columns: s1_entity_id, target_entity_id, route_exact_name_address,
    route_exact_name.
    """
    s1_name_norm = [normalize_field(n) for n in s1_names]
    s1_addr_norm = [normalize_field(a) for a in s1_addresses]
    target_name_norm = [normalize_field(n) for n in target_names]
    target_addr_norm = [normalize_field(a) for a in target_addresses]

    s1_name_addr_key = [f"{n}\x1f{a}" for n, a in zip(s1_name_norm, s1_addr_norm)]
    target_name_addr_key = [f"{n}\x1f{a}" for n, a in zip(target_name_norm, target_addr_norm)]

    name_addr_postings = _grouped_postings(target_ids, target_name_addr_key)
    name_postings = _grouped_postings(target_ids, target_name_norm)

    rows_s1: list[str] = []
    rows_target: list[str] = []
    rows_na: list[bool] = []
    rows_n: list[bool] = []

    for s1_id, na_key, n_key in zip(s1_ids, s1_name_addr_key, s1_name_norm):
        matched: dict[str, list[bool]] = {}
        for tid in name_addr_postings.get(na_key, ()):
            matched.setdefault(tid, [False, False])[0] = True
        if n_key:
            for tid in name_postings.get(n_key, ()):
                matched.setdefault(tid, [False, False])[1] = True
        for tid, (na_flag, n_flag) in matched.items():
            rows_s1.append(s1_id)
            rows_target.append(tid)
            rows_na.append(na_flag)
            rows_n.append(n_flag)

    return pd.DataFrame(
        {
            "s1_entity_id": rows_s1,
            "target_entity_id": rows_target,
            "route_exact_name_address": rows_na,
            "route_exact_name": rows_n,
        }
    )


def collision_group_sizes(postings: dict[str, list[str]]) -> pd.Series:
    return pd.Series({key: len(ids) for key, ids in postings.items()}).sort_values(ascending=False)
