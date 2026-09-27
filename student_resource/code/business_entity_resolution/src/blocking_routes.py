"""Route builders: turn fitted vectors + top-k retrieval into candidate
DataFrames with route provenance columns, for EXP-B001 (forward),
EXP-B001R (reverse), and EXP-B002 (character n-gram, same mechanism as
forward with a different vectorizer).

Forward and reverse are the same retrieval primitive
(`blocking_topk.batched_top_k`) with query/target roles swapped -- this
module is where that symmetry is made explicit and where each direction's
rank/score provenance is attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from blocking_topk import batched_top_k


def forward_route(
    s1_ids: list[str],
    s1_matrix: sparse.csr_matrix,
    target_ids: list[str],
    target_matrix: sparse.csr_matrix,
    k: int,
    batch_size: int = 2000,
    min_score: float = 0.0,
    route_col: str = "route_forward_word",
    rank_col: str = "forward_rank",
    score_col: str = "forward_score",
) -> pd.DataFrame:
    """S1 -> top-k target rows. One row per (s1, target) candidate edge."""
    result = batched_top_k(s1_matrix, target_matrix, k=k, batch_size=batch_size, min_score=min_score)
    target_ids_arr = np.asarray(target_ids)

    rows_s1: list[str] = []
    rows_target: list[str] = []
    rows_rank: list[int] = []
    rows_score: list[float] = []

    for i, s1_id in enumerate(s1_ids):
        idx, scores = result.row(i)
        if idx.size == 0:
            continue
        rows_s1.extend([s1_id] * idx.size)
        rows_target.extend(target_ids_arr[idx].tolist())
        rows_rank.extend(range(1, idx.size + 1))
        rows_score.extend(scores.tolist())

    df = pd.DataFrame(
        {
            "s1_entity_id": rows_s1,
            "target_entity_id": rows_target,
            rank_col: rows_rank,
            score_col: rows_score,
        }
    )
    df[route_col] = True
    return df


def reverse_route(
    s1_ids: list[str],
    s1_matrix: sparse.csr_matrix,
    target_ids: list[str],
    target_matrix: sparse.csr_matrix,
    r: int,
    batch_size: int = 2000,
    min_score: float = 0.0,
    route_col: str = "route_reverse_word",
    rank_col: str = "reverse_rank",
    score_col: str = "reverse_score",
) -> pd.DataFrame:
    """Each target document -> top-r nearest S1 entities, then inverted into
    S1 -> target candidate edges. Same primitive as `forward_route` with
    query/target swapped; the caller still gets an S1-indexed edge list.
    """
    result = batched_top_k(target_matrix, s1_matrix, k=r, batch_size=batch_size, min_score=min_score)
    s1_ids_arr = np.asarray(s1_ids)

    rows_s1: list[str] = []
    rows_target: list[str] = []
    rows_rank: list[int] = []
    rows_score: list[float] = []

    for i, target_id in enumerate(target_ids):
        idx, scores = result.row(i)
        if idx.size == 0:
            continue
        rows_s1.extend(s1_ids_arr[idx].tolist())
        rows_target.extend([target_id] * idx.size)
        rows_rank.extend(range(1, idx.size + 1))
        rows_score.extend(scores.tolist())

    df = pd.DataFrame(
        {
            "s1_entity_id": rows_s1,
            "target_entity_id": rows_target,
            rank_col: rows_rank,
            score_col: rows_score,
        }
    )
    df[route_col] = True
    return df
