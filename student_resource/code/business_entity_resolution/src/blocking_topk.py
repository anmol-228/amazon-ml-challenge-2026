"""Batched sparse top-k retrieval, shared by forward and reverse routes.

Forward (EXP-B001) and reverse (EXP-B001R) retrieval are the *same*
operation with query/target roles swapped: given a query sparse matrix and a
target sparse matrix (both L2-normalized TF-IDF rows, so a dot product is
exactly cosine similarity), return, for every query row, the top-k target
row indices and their scores. Implemented as batched sparse @ sparse.T
matrix multiplication (never a dense NxM matrix), with per-row top-k
extracted directly from each batch's CSR slice via `np.argpartition` (no
densification, no full sort where a partial one suffices).

Result storage is batch-flattened, not one ndarray object per query row.
This was a real, measured bug, not a theoretical concern: at `development`
scale (1,765,456 rows), storing one small ndarray per row for both indices
and scores meant retaining ~3.5 million individual small array objects for
the whole run. A direct diagnostic (see experiments/blocking/B001's run
history) showed this drove RSS from ~1.4GB up past 25GB and back down in a
sawtooth pattern as Python's allocator struggled to reclaim/reuse that many
small, variably-sized blocks -- and eventually a routine ~40-80MB batch
allocation failed with `std::bad_alloc` despite tens of GB of system memory
being free, consistent with heap fragmentation from that churn, not true
memory exhaustion. Concatenating each batch's per-row results into one flat
array immediately (instead of keeping ~1500 small arrays per batch alive)
cuts the retained-object count by roughly 1500x (~2,400 objects for the
whole run instead of ~3.5 million) and was verified to keep RSS flat across
a full 1177-batch dry run before this fix was trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class TopKResult:
    """Batch-flattened ragged top-k result.

    Row i's neighbors are `flat_indices[offsets[i]:offsets[i+1]]` with
    matching scores in `flat_scores`. `offsets` has length n_rows + 1.
    """

    flat_indices: np.ndarray
    flat_scores: np.ndarray
    offsets: np.ndarray

    @property
    def n_rows(self) -> int:
        return len(self.offsets) - 1

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.flat_indices[lo:hi], self.flat_scores[lo:hi]


def _top_k_from_csr_row(indices_row: np.ndarray, data_row: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if data_row.size == 0:
        return indices_row, data_row
    if data_row.size <= k:
        order = np.argsort(-data_row)
        return indices_row[order], data_row[order]
    part = np.argpartition(-data_row, k - 1)[:k]
    order = part[np.argsort(-data_row[part])]
    return indices_row[order], data_row[order]


def _oom_exception_types() -> tuple[type[BaseException], ...]:
    """MemoryError plus numpy's more specific array-allocation-failure type,
    resolved defensively since its module path has moved between numpy
    versions (`numpy.core._exceptions` pre-2.0, `numpy._core._exceptions`
    in 2.x, both private/internal).
    """
    types: list[type[BaseException]] = [MemoryError]
    for module_path in ("numpy._core._exceptions", "numpy.core._exceptions"):
        try:
            import importlib

            mod = importlib.import_module(module_path)
            types.append(mod._ArrayMemoryError)
            break
        except (ImportError, AttributeError):
            continue
    return tuple(types)


_OOM_EXCEPTIONS = _oom_exception_types()

MIN_SUB_BATCH_SIZE = 8


def _compute_batch_sims(batch: sparse.csr_matrix, target_t: sparse.csr_matrix):
    """`batch.dot(target_t)`, splitting into halves on an allocation
    failure instead of assuming a fixed batch_size is always small enough.
    Real data is not uniformly dense: a batch of rows with unusually common
    tokens can produce a far larger intermediate sparse product than a
    batch of the same size elsewhere, even when total system memory is
    ample (observed directly on this project). Halving is a robust,
    self-correcting response that does not depend on diagnosing the exact
    OS-level cause. Returns a list of (row_offset, csr_result) pieces so
    the caller can still map back to original row positions after a split.
    """
    n_rows = batch.shape[0]
    if n_rows == 0:
        return []
    try:
        return [(0, batch.dot(target_t).tocsr())]
    except _OOM_EXCEPTIONS:
        if n_rows <= MIN_SUB_BATCH_SIZE:
            raise
        mid = n_rows // 2
        left = _compute_batch_sims(batch[:mid], target_t)
        right = _compute_batch_sims(batch[mid:], target_t)
        return left + [(mid + off, s) for off, s in right]


def _process_batch(
    batch: sparse.csr_matrix,
    target_t: sparse.csr_matrix,
    k: int,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one batch's top-k, returning batch-flattened
    (flat_indices, flat_scores, row_lengths) -- never one array per row.
    """
    pieces = _compute_batch_sims(batch, target_t)
    n_rows = batch.shape[0]
    row_indices: list[np.ndarray] = [None] * n_rows  # type: ignore[list-item]
    row_scores: list[np.ndarray] = [None] * n_rows  # type: ignore[list-item]

    for row_offset, sims in pieces:
        indptr = sims.indptr
        col_idx = sims.indices
        data = sims.data
        for local_row in range(sims.shape[0]):
            lo, hi = indptr[local_row], indptr[local_row + 1]
            row_idx = col_idx[lo:hi]
            row_data = data[lo:hi]
            if min_score > 0.0 and row_data.size:
                keep = row_data >= min_score
                row_idx = row_idx[keep]
                row_data = row_data[keep]
            top_idx, top_score = _top_k_from_csr_row(row_idx, row_data, k)
            row_indices[row_offset + local_row] = top_idx
            row_scores[row_offset + local_row] = top_score

    lengths = np.fromiter((len(a) for a in row_indices), dtype=np.int64, count=n_rows)
    flat_indices = (
        np.concatenate(row_indices) if lengths.sum() > 0 else np.empty(0, dtype=np.int64)
    )
    flat_scores = (
        np.concatenate(row_scores) if lengths.sum() > 0 else np.empty(0, dtype=np.float32)
    )
    return flat_indices, flat_scores, lengths


def batched_top_k(
    query_matrix: sparse.csr_matrix,
    target_matrix: sparse.csr_matrix,
    k: int,
    batch_size: int = 2000,
    min_score: float = 0.0,
) -> TopKResult:
    """For every row of `query_matrix`, return top-k rows of `target_matrix`.

    Both inputs must already be L2-normalized (dot product == cosine sim).
    `target_matrix` is transposed once up front and reused across batches.
    Any single batch that fails to allocate is transparently retried as two
    half-sized sub-batches (see `_compute_batch_sims`), so `batch_size` is a
    starting point/throughput knob, not a hard memory-safety guarantee.
    """
    target_t = target_matrix.T.tocsr()
    n_queries = query_matrix.shape[0]

    batch_flat_indices: list[np.ndarray] = []
    batch_flat_scores: list[np.ndarray] = []
    batch_lengths: list[np.ndarray] = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_matrix[start:end]
        flat_idx, flat_scr, lengths = _process_batch(batch, target_t, k, min_score)
        batch_flat_indices.append(flat_idx)
        batch_flat_scores.append(flat_scr)
        batch_lengths.append(lengths)

    all_flat_indices = (
        np.concatenate(batch_flat_indices) if batch_flat_indices else np.empty(0, dtype=np.int64)
    )
    all_flat_scores = (
        np.concatenate(batch_flat_scores) if batch_flat_scores else np.empty(0, dtype=np.float32)
    )
    all_lengths = (
        np.concatenate(batch_lengths) if batch_lengths else np.empty(0, dtype=np.int64)
    )
    offsets = np.zeros(len(all_lengths) + 1, dtype=np.int64)
    np.cumsum(all_lengths, out=offsets[1:])

    return TopKResult(flat_indices=all_flat_indices, flat_scores=all_flat_scores, offsets=offsets)
