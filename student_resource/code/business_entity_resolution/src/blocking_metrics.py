"""Authoritative Phase-2 blocking metrics (mission spec section 8).

All metric definitions here are diagnostics computed before any matcher/
threshold decision exists. They do NOT touch or redefine the frozen official
evaluator (`evaluation.py`, DEC-010). Every function operates on plain
Python dicts of `{s1_entity_id: frozenset(target_entity_id)}` so it has no
dependency on any particular candidate-table storage format, and can be unit
tested with tiny hand-computed examples.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def build_id_sets_by_s1(
    df: pd.DataFrame, s1_col: str = "s1_entity_id", id_col: str = "target_entity_id"
) -> dict[str, frozenset[str]]:
    """Group candidate edges into {s1_entity_id: frozenset(target_ids)}.

    Deliberately NOT `df.groupby(s1_col)[id_col].apply(lambda s: frozenset(s))`
    -- that pattern instantiates one Python object per group via pandas'
    generic (non-vectorized) apply path, which is both slow and, at
    `development` scale (~1.7M distinct S1 groups over tens of millions of
    candidate rows), memory-heavy enough to raise MemoryError in practice
    (observed directly in this project on EXP-B003's candidate table). A
    plain single-pass dict-of-sets accumulation is both faster and far
    lighter here.
    """
    if df.empty:
        return {}
    result: dict[str, set[str]] = defaultdict(set)
    for s1, tid in zip(df[s1_col].to_numpy(), df[id_col].to_numpy()):
        result[s1].add(tid)
    return {k: frozenset(v) for k, v in result.items()}


def _load_stats(sizes: np.ndarray) -> dict[str, float]:
    if sizes.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
        "p95": float(np.percentile(sizes, 95)),
        "p99": float(np.percentile(sizes, 99)),
        "max": float(np.max(sizes)),
    }


@dataclass
class BlockingMetrics:
    link_recall: float
    macro_candidate_recall_non_singleton: float
    complete_true_link_coverage_non_singleton: float
    singleton_candidate_exposure_rate: float
    singleton_candidate_load: dict[str, float]
    overall_candidate_load: dict[str, float]
    total_candidate_edges: int
    global_reduction_ratio: float | None = None
    routing_aware_reduction_ratio: float | None = None
    n_entities: int = 0
    n_true_singletons: int = 0
    n_true_non_singletons: int = 0
    n_true_links: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "link_recall": self.link_recall,
            "macro_candidate_recall_non_singleton": self.macro_candidate_recall_non_singleton,
            "complete_true_link_coverage_non_singleton": self.complete_true_link_coverage_non_singleton,
            "singleton_candidate_exposure_rate": self.singleton_candidate_exposure_rate,
            "singleton_candidate_load": self.singleton_candidate_load,
            "overall_candidate_load": self.overall_candidate_load,
            "total_candidate_edges": self.total_candidate_edges,
            "global_reduction_ratio": self.global_reduction_ratio,
            "routing_aware_reduction_ratio": self.routing_aware_reduction_ratio,
            "n_entities": self.n_entities,
            "n_true_singletons": self.n_true_singletons,
            "n_true_non_singletons": self.n_true_non_singletons,
            "n_true_links": self.n_true_links,
            "extra": self.extra,
        }


def compute_blocking_metrics(
    all_s1_ids: list[str],
    truth_by_s1: dict[str, frozenset[str]],
    candidates_by_s1: dict[str, frozenset[str]],
    n_s2_total: int | None = None,
    n_s3_total: int | None = None,
    routing_aware_denominator: int | None = None,
) -> BlockingMetrics:
    """Compute the seven authoritative blocking metrics over `all_s1_ids`.

    `truth_by_s1` and `candidates_by_s1` may omit an entity entirely, which
    is treated identically to mapping it to an empty frozenset.
    """
    n_entities = len(all_s1_ids)
    tp_sum = 0
    truth_size_sum = 0
    per_entity_recall_non_singleton: list[float] = []
    complete_coverage_flags: list[bool] = []
    singleton_exposed_flags: list[bool] = []
    singleton_sizes: list[int] = []
    overall_sizes: list[int] = []
    n_true_singletons = 0
    n_true_non_singletons = 0

    for s1 in all_s1_ids:
        truth = truth_by_s1.get(s1, frozenset())
        cand = candidates_by_s1.get(s1, frozenset())
        overall_sizes.append(len(cand))

        if len(truth) == 0:
            n_true_singletons += 1
            singleton_sizes.append(len(cand))
            singleton_exposed_flags.append(len(cand) > 0)
            continue

        n_true_non_singletons += 1
        tp = len(truth & cand)
        tp_sum += tp
        truth_size_sum += len(truth)
        recall = tp / len(truth)
        per_entity_recall_non_singleton.append(recall)
        complete_coverage_flags.append(truth.issubset(cand))

    link_recall = (tp_sum / truth_size_sum) if truth_size_sum else float("nan")
    macro_candidate_recall = (
        float(np.mean(per_entity_recall_non_singleton)) if per_entity_recall_non_singleton else float("nan")
    )
    complete_coverage = (
        float(np.mean(complete_coverage_flags)) if complete_coverage_flags else float("nan")
    )
    singleton_exposure = (
        float(np.mean(singleton_exposed_flags)) if singleton_exposed_flags else float("nan")
    )

    total_edges = int(sum(overall_sizes))

    global_rr = None
    if n_s2_total is not None and n_s3_total is not None and n_entities > 0:
        unrestricted = n_entities * (n_s2_total + n_s3_total)
        global_rr = 1.0 - (total_edges / unrestricted) if unrestricted else None

    routing_rr = None
    if routing_aware_denominator is not None and routing_aware_denominator > 0:
        routing_rr = 1.0 - (total_edges / routing_aware_denominator)

    return BlockingMetrics(
        link_recall=link_recall,
        macro_candidate_recall_non_singleton=macro_candidate_recall,
        complete_true_link_coverage_non_singleton=complete_coverage,
        singleton_candidate_exposure_rate=singleton_exposure,
        singleton_candidate_load=_load_stats(np.array(singleton_sizes)),
        overall_candidate_load=_load_stats(np.array(overall_sizes)),
        total_candidate_edges=total_edges,
        global_reduction_ratio=global_rr,
        routing_aware_reduction_ratio=routing_rr,
        n_entities=n_entities,
        n_true_singletons=n_true_singletons,
        n_true_non_singletons=n_true_non_singletons,
        n_true_links=truth_size_sum,
    )


def classify_true_links_by_name_overlap(
    truth_by_s1: dict[str, frozenset[str]],
    s1_name_tokens: dict[str, frozenset[str]],
    target_name_tokens: dict[str, frozenset[str]],
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Split every S1's true-target set into two truth dicts: targets with
    nonzero normalized-business-name token overlap with that S1, and
    targets with zero overlap. Used for EXP-B001 diagnostics (B)/(C)/(D).
    """
    nonzero: dict[str, frozenset[str]] = {}
    zero: dict[str, frozenset[str]] = {}
    for s1, targets in truth_by_s1.items():
        if not targets:
            continue
        s1_tokens = s1_name_tokens.get(s1, frozenset())
        nz, z = set(), set()
        for t in targets:
            t_tokens = target_name_tokens.get(t, frozenset())
            if s1_tokens & t_tokens:
                nz.add(t)
            else:
                z.add(t)
        if nz:
            nonzero[s1] = frozenset(nz)
        if z:
            zero[s1] = frozenset(z)
    return nonzero, zero


def zero_overlap_residual_links(
    zero_overlap_truth_by_s1: dict[str, frozenset[str]],
    candidates_by_s1: dict[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    """The B001 diagnostic (D) complement: zero-name-overlap true links that
    are NOT present in the given candidate set (i.e., not address-rescued
    either). This is EXP-B002's actual target population per the mission
    spec's round-3 correction, not the raw zero-overlap population.
    """
    residual: dict[str, frozenset[str]] = {}
    for s1, targets in zero_overlap_truth_by_s1.items():
        cand = candidates_by_s1.get(s1, frozenset())
        missing = targets - cand
        if missing:
            residual[s1] = frozenset(missing)
    return residual


def route_incremental_value(
    all_s1_ids: list[str],
    truth_by_s1: dict[str, frozenset[str]],
    baseline_candidates_by_s1: dict[str, frozenset[str]],
    union_candidates_by_s1: dict[str, frozenset[str]],
) -> dict:
    """What does the union add that the baseline (without the new route)
    did not already have? Returns unique-true-links-rescued and newly-
    completely-covered-entity counts, per Phase-2 mission spec section 8.9.
    """
    new_links_rescued = 0
    newly_complete = 0
    for s1 in all_s1_ids:
        truth = truth_by_s1.get(s1, frozenset())
        if not truth:
            continue
        base = baseline_candidates_by_s1.get(s1, frozenset())
        union = union_candidates_by_s1.get(s1, frozenset())
        base_tp = truth & base
        union_tp = truth & union
        new_links_rescued += len(union_tp - base_tp)
        if truth.issubset(union) and not truth.issubset(base):
            newly_complete += 1
    return {"new_true_links_rescued": new_links_rescued, "newly_complete_s1_entities": newly_complete}


# ---------------------------------------------------------------------------
# Scalable metric path: computes the exact same seven metrics as
# `compute_blocking_metrics` without ever materializing a per-entity
# Python frozenset or concatenating full candidate tables into one giant
# union. Necessary once candidate tables reach tens/hundreds of millions of
# rows (EXP-B001R's r-grid union crashed on a `pandas.concat` +
# `drop_duplicates` + dict-of-sets rebuild at that scale). Ground truth is
# comparatively small (a few million true links even at full `development`
# scale), so every metric here is computed either by (a) a hash-join between
# the small ground-truth edge table and a candidate table, or (b) a
# vectorized `groupby(...).nunique()`/`groupby(...).size()` -- never by
# building Python-level sets over candidate rows.
#
# Semantics are identical to `compute_blocking_metrics` by construction and
# are verified equal on synthetic data in
# tests/test_blocking_infrastructure.py; this is a performance rewrite, not
# a metric redefinition.
# ---------------------------------------------------------------------------


def truth_edges_dataframe(truth_by_s1: dict[str, frozenset[str]]) -> pd.DataFrame:
    """One row per true (s1_entity_id, target_entity_id) link. Small: a few
    million rows even at full `development` scale, versus tens/hundreds of
    millions of candidate rows -- this is why joins against this table, not
    against the candidate table's own size, are the cheap direction.
    """
    s1_list: list[str] = []
    target_list: list[str] = []
    for s1, targets in truth_by_s1.items():
        for t in targets:
            s1_list.append(s1)
            target_list.append(t)
    return pd.DataFrame({"s1_entity_id": s1_list, "target_entity_id": target_list})


def candidate_count_per_s1(
    df: pd.DataFrame,
    all_s1_ids: list[str],
    s1_col: str = "s1_entity_id",
    id_col: str = "target_entity_id",
) -> pd.Series:
    """Distinct candidate count per S1 entity, reindexed over `all_s1_ids`
    (entities absent from `df` get 0). Uses pandas' vectorized
    `groupby().nunique()`, not a Python-level set per group.
    """
    if df.empty:
        return pd.Series(0, index=all_s1_ids, dtype="int64")
    counts = df.groupby(s1_col)[id_col].nunique()
    return counts.reindex(all_s1_ids, fill_value=0).astype("int64")


def intersection_count_per_s1(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    all_s1_ids: list[str],
    s1_col: str = "s1_entity_id",
    id_col: str = "target_entity_id",
) -> pd.Series:
    """Per-S1 count of candidate edges present in BOTH `df_a` and `df_b`.

    NOT a `pandas.merge` on the raw (s1, target) STRING columns -- a plain
    two-column string merge between two genuinely large candidate tables
    (observed directly on this project: 117.7M and 185.1M rows) is itself
    expensive enough to `MemoryError` inside pandas' own hash-join
    factorization, even though neither table alone is a problem. Each
    (s1, target) pair is instead hashed into a single uint64 key
    (`pandas.util.hash_pandas_object`), turning the join into a single-
    column integer membership test. A 64-bit hash collision across even
    hundreds of millions of pairs is not a practically realistic risk.

    The membership test itself is a manual sort + chunked
    `numpy.searchsorted`, not `numpy.isin` -- `numpy.isin`'s automatic
    algorithm selection was observed to crash the whole process natively
    (a hard, non-Python-catchable exit, not a `MemoryError`) at this same
    100M+-element scale on this machine. Sorting the smaller side once and
    probing the larger side in bounded-size chunks is an explicit,
    well-understood algorithm with predictable peak memory per step,
    rather than relying on an opaque internal heuristic at a scale where
    it apparently breaks down.
    """
    if df_a.empty or df_b.empty:
        return pd.Series(0, index=all_s1_ids, dtype="int64")
    keys_a = df_a[[s1_col, id_col]].drop_duplicates()
    keys_b = df_b[[s1_col, id_col]].drop_duplicates()
    hash_a = pd.util.hash_pandas_object(keys_a, index=False).to_numpy()
    hash_b_sorted = np.sort(pd.util.hash_pandas_object(keys_b, index=False).to_numpy())

    chunk_size = 5_000_000
    in_b_parts: list[np.ndarray] = []
    for start in range(0, len(hash_a), chunk_size):
        chunk = hash_a[start : start + chunk_size]
        pos = np.searchsorted(hash_b_sorted, chunk)
        pos_clipped = np.clip(pos, 0, len(hash_b_sorted) - 1)
        in_b_parts.append(hash_b_sorted[pos_clipped] == chunk)
    in_b = np.concatenate(in_b_parts) if in_b_parts else np.zeros(0, dtype=bool)

    counts = keys_a.loc[in_b].groupby(s1_col).size()
    return counts.reindex(all_s1_ids, fill_value=0).astype("int64")


def union_count_per_s1(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    all_s1_ids: list[str],
    s1_col: str = "s1_entity_id",
    id_col: str = "target_entity_id",
) -> pd.Series:
    """|A ∪ B| per S1 = |A| + |B| - |A ∩ B|, computed from three cheap
    per-entity count Series -- the union's row-level table is never built.
    """
    a = candidate_count_per_s1(df_a, all_s1_ids, s1_col, id_col)
    b = candidate_count_per_s1(df_b, all_s1_ids, s1_col, id_col)
    inter = intersection_count_per_s1(df_a, df_b, all_s1_ids, s1_col, id_col)
    return a + b - inter


def link_found_mask(
    truth_edges_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    s1_col: str = "s1_entity_id",
    id_col: str = "target_entity_id",
) -> np.ndarray:
    """Boolean array aligned to `truth_edges_df` rows: is each true link
    present in `candidate_df`? A left hash-join of the SMALL truth-edge
    table against the candidate table, not a scan building per-entity sets.
    """
    if candidate_df.empty:
        return np.zeros(len(truth_edges_df), dtype=bool)
    cand_keys = candidate_df[[s1_col, id_col]].drop_duplicates().rename(
        columns={s1_col: "s1_entity_id", id_col: "target_entity_id"}
    )
    cand_keys = cand_keys.assign(_found=True)
    merged = truth_edges_df.merge(cand_keys, on=["s1_entity_id", "target_entity_id"], how="left")
    # The left join leaves `_found` as an object column (True mixed with
    # NaN for unmatched rows) before fillna, so an explicit bool cast is
    # required -- otherwise `.to_numpy()` yields a numpy `object` array of
    # plain Python bools, and a later `~mask` on that array inverts each
    # Python bool individually (a deprecated, and undesired, bitwise-int
    # operation) instead of vectorized boolean negation.
    return merged["_found"].fillna(False).astype(bool).to_numpy()


def compute_blocking_metrics_scalable(
    all_s1_ids: list[str],
    truth_edges_df: pd.DataFrame,
    found_mask: np.ndarray,
    candidate_counts_per_s1: pd.Series,
    n_s2_total: int | None = None,
    n_s3_total: int | None = None,
    routing_aware_denominator: int | None = None,
) -> BlockingMetrics:
    """Same seven metrics as `compute_blocking_metrics`, computed from a
    pre-joined truth-edge `found_mask` and pre-aggregated per-entity
    candidate counts, so the caller controls how those two inputs were
    produced (a single-table join, or an OR/sum-minus-intersection across
    two tables for a union -- see `union_count_per_s1`/`link_found_mask`
    called once per table and OR'd) without this function ever seeing a
    materialized union table itself.

    `found_mask` must align 1:1 with `truth_edges_df` rows.
    `candidate_counts_per_s1` must be indexed over (at least) `all_s1_ids`.
    """
    n_entities = len(all_s1_ids)
    n_true_links = len(truth_edges_df)
    tp_sum = int(found_mask.sum()) if n_true_links else 0
    link_recall = (tp_sum / n_true_links) if n_true_links else float("nan")

    if n_true_links:
        te = truth_edges_df[["s1_entity_id"]].copy()
        te["_found"] = found_mask
        grp = te.groupby("s1_entity_id")["_found"].agg(["sum", "count"])
        per_entity_recall = grp["sum"] / grp["count"]
        macro_candidate_recall = float(per_entity_recall.mean())
        complete_coverage = float((grp["sum"] == grp["count"]).mean())
        non_singleton_ids = set(grp.index)
    else:
        macro_candidate_recall = float("nan")
        complete_coverage = float("nan")
        non_singleton_ids = set()

    n_true_non_singletons = len(non_singleton_ids)
    n_true_singletons = n_entities - n_true_non_singletons

    counts = candidate_counts_per_s1.reindex(all_s1_ids, fill_value=0)
    is_singleton_mask = np.array([s1 not in non_singleton_ids for s1 in all_s1_ids])
    overall_sizes = counts.to_numpy()
    singleton_sizes = overall_sizes[is_singleton_mask]

    singleton_exposure = (
        float(np.mean(singleton_sizes > 0)) if singleton_sizes.size else float("nan")
    )
    total_edges = int(overall_sizes.sum())

    global_rr = None
    if n_s2_total is not None and n_s3_total is not None and n_entities > 0:
        unrestricted = n_entities * (n_s2_total + n_s3_total)
        global_rr = 1.0 - (total_edges / unrestricted) if unrestricted else None

    routing_rr = None
    if routing_aware_denominator is not None and routing_aware_denominator > 0:
        routing_rr = 1.0 - (total_edges / routing_aware_denominator)

    return BlockingMetrics(
        link_recall=link_recall,
        macro_candidate_recall_non_singleton=macro_candidate_recall,
        complete_true_link_coverage_non_singleton=complete_coverage,
        singleton_candidate_exposure_rate=singleton_exposure,
        singleton_candidate_load=_load_stats(singleton_sizes),
        overall_candidate_load=_load_stats(overall_sizes),
        total_candidate_edges=total_edges,
        global_reduction_ratio=global_rr,
        routing_aware_reduction_ratio=routing_rr,
        n_entities=n_entities,
        n_true_singletons=n_true_singletons,
        n_true_non_singletons=n_true_non_singletons,
        n_true_links=n_true_links,
    )


def route_incremental_value_scalable(
    truth_edges_df: pd.DataFrame,
    baseline_found_mask: np.ndarray,
    union_found_mask: np.ndarray,
) -> dict:
    """Scalable equivalent of `route_incremental_value`: new true links
    rescued = truth edges found in the union but not the baseline; newly
    complete entities = entities where every true link is now found but at
    least one was missing under the baseline. Computed purely from the two
    boolean masks (aligned to `truth_edges_df`) -- no per-entity set ever
    built, no candidate table touched.
    """
    newly_found = union_found_mask & ~baseline_found_mask
    new_links_rescued = int(newly_found.sum())

    te = truth_edges_df[["s1_entity_id"]].copy()
    te["_base"] = baseline_found_mask
    te["_union"] = union_found_mask
    grp = te.groupby("s1_entity_id")[["_base", "_union"]].agg(["sum", "count"])
    base_complete = grp[("_base", "sum")] == grp[("_base", "count")]
    union_complete = grp[("_union", "sum")] == grp[("_union", "count")]
    newly_complete = int((union_complete & ~base_complete).sum())

    return {"new_true_links_rescued": new_links_rescued, "newly_complete_s1_entities": newly_complete}
