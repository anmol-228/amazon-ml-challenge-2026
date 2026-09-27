"""Candidate union / dedup across blocking routes (EXP-B004).

Candidate identity is the composite key (s1_entity_id, target_source,
target_entity_id) per `blocking_identity.py`. Every input route DataFrame
must have at least [s1_entity_id, target_entity_id]; `target_source` is
derived here (never trusted from the route itself) so a bug in one route
cannot mislabel a candidate's source. Route-specific columns are preserved
across the union by column-name convention:

  - columns named `route_*`            -> boolean, combined with OR
  - columns ending in `_rank`          -> combined with min (best rank wins)
  - columns ending in `_score`         -> combined with max (best score wins)
  - anything else                      -> first non-null value kept

This lets each route contribute whatever provenance columns it has without
the union module needing to know about every route in advance.

Two implementations share the exact same aggregation semantics
(`_aggregate_union_frame`):

  - `union_candidates`: simple in-memory union for small/moderate route
    tables (concatenates everything, then aggregates once).
  - `union_candidates_partitioned`: memory-safe union for large route
    tables. Reads each route's parquet file ONE AT A TIME (never holding
    more than one route table in memory simultaneously), scatters rows
    into `n_buckets` on-disk partitions by a stable hash of
    `s1_entity_id`, then aggregates each (small) partition and appends it
    straight to the output file -- the full unioned table is never held
    in memory at once, regardless of total input size. This exists
    because B001-B003's own route tables individually reach the tens/
    hundreds of millions of rows, and a canonical B004 union over several
    of them at once is exactly the scale where a single in-memory
    `pandas.concat` + `groupby` was judged unsafe under this machine's
    ~31.6GB RAM budget (see docs/PHASE_2_BLOCKING_REPORT.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking_identity import target_source_of


def _add_target_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_source"] = df["target_entity_id"].map(target_source_of)
    return df


def _aggregate_union_frame(combined: pd.DataFrame) -> pd.DataFrame:
    """Core aggregation: given a DataFrame that already contains every
    contributing route's rows (with `target_source` present), collapse to
    one row per (s1_entity_id, target_source, target_entity_id) using the
    route_*/[_rank]/[_score]/other column-name convention described in the
    module docstring. Shared by both the simple and partitioned union
    implementations so their semantics can never silently diverge.
    """
    key = ["s1_entity_id", "target_source", "target_entity_id"]
    route_bool_cols = [c for c in combined.columns if c.startswith("route_")]
    rank_cols = [c for c in combined.columns if c.endswith("_rank")]
    score_cols = [c for c in combined.columns if c.endswith("_score")]
    other_cols = [
        c for c in combined.columns if c not in key and c not in route_bool_cols + rank_cols + score_cols
    ]

    for c in route_bool_cols:
        combined[c] = combined[c].fillna(False).astype(bool)

    grouped = combined.groupby(key, sort=False)

    agg_frames = []
    if route_bool_cols:
        agg_frames.append(grouped[route_bool_cols].max())
    if rank_cols:
        agg_frames.append(grouped[rank_cols].min())
    if score_cols:
        agg_frames.append(grouped[score_cols].max())
    if other_cols:
        agg_frames.append(grouped[other_cols].first())

    if agg_frames:
        return pd.concat(agg_frames, axis=1).reset_index()
    return grouped.size().reset_index(name="_n")


def union_candidates(route_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Simple in-memory union. Only safe for route tables small/moderate
    enough that concatenating all of them at once fits comfortably in
    memory -- see `union_candidates_partitioned` otherwise.
    """
    if not route_frames:
        return pd.DataFrame(columns=["s1_entity_id", "target_source", "target_entity_id"])

    prepared = [_add_target_source(df) for df in route_frames if not df.empty]
    if not prepared:
        return pd.DataFrame(columns=["s1_entity_id", "target_source", "target_entity_id"])

    combined = pd.concat(prepared, axis=0, ignore_index=True, sort=False)
    return _aggregate_union_frame(combined)


def any_route_flag_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("route_")]


def _bucket_of(df: pd.DataFrame, n_buckets: int) -> pd.Series:
    return pd.util.hash_pandas_object(df[["s1_entity_id"]], index=False) % n_buckets


def union_candidates_partitioned(
    route_paths: list[Path],
    output_path: Path,
    n_buckets: int = 64,
    tmp_dir: Path | None = None,
) -> dict:
    """Memory-safe candidate union over one or more route PARQUET FILES.

    Scatter phase: each route file is read fully into memory ONE AT A TIME
    (safe -- every individual route table produced by this project's own
    scripts has already been proven to load without incident) and
    immediately scattered, by a stable hash of `s1_entity_id` modulo
    `n_buckets`, into `n_buckets` on-disk partition parquet files under
    `tmp_dir`. The route DataFrame is dropped before the next route is
    read, so peak memory during this phase is bounded by the single
    largest route table, never the sum of all of them.

    Gather phase: each (small, ~total_rows / n_buckets) partition file is
    read, aggregated with the exact same logic as `union_candidates`
    (`_aggregate_union_frame`), and appended as a new row group directly
    to `output_path` via a single open `pyarrow.parquet.ParquetWriter` --
    the full unioned table is never assembled in memory at once.

    Returns a small dict of diagnostics (row counts per phase), not the
    unioned table itself -- callers read `output_path` afterward.
    """
    tmp_dir = tmp_dir or (output_path.parent / f"{output_path.stem}_partitions_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    partition_paths = [tmp_dir / f"bucket_{b:04d}.parquet" for b in range(n_buckets)]
    writers: dict[int, pq.ParquetWriter] = {}
    total_input_rows = 0

    # Different routes legitimately have different columns (e.g.
    # `forward_rank`/`forward_score` vs `reverse_rank`/`reverse_score`) --
    # a bucket's ParquetWriter must be opened with the UNION of every
    # route's columns from the start, or a later route's table (with a
    # genuinely different column set, not just a type mismatch) cannot be
    # reconciled by `Table.cast`. Schemas are read WITHOUT loading any
    # route's data (`pq.ParquetFile(...).schema_arrow`), so this costs
    # nothing memory-wise.
    union_schema: dict[str, pa.DataType] = {}
    for route_path in route_paths:
        for field in pq.ParquetFile(route_path).schema_arrow:
            union_schema.setdefault(field.name, field.type)
    if "target_source" not in union_schema:
        union_schema["target_source"] = pa.string()

    # Building every scattered table against this ONE fixed target schema
    # (rather than letting `Table.from_pandas` infer types per call, then
    # trying to reconcile mismatches afterward) is what actually avoids
    # the type conflict: a padding column of all-None values, if left to
    # type inference, becomes pyarrow's special `null` type, and pyarrow
    # cannot cast a real `bool` column (from a route that DOES have that
    # column) backward into `null` -- only the reverse direction works.
    # Constructing with an explicit schema from the start sidesteps this
    # asymmetry entirely.
    target_schema = pa.schema(list(union_schema.items()))

    def _pad_to_union_columns(df: pd.DataFrame) -> pd.DataFrame:
        for col in union_schema:
            if col not in df.columns:
                df[col] = None
        return df[list(union_schema.keys())]

    try:
        # ---- scatter --------------------------------------------------
        for route_path in route_paths:
            df = pd.read_parquet(route_path)
            if df.empty:
                continue
            df = _add_target_source(df)
            df = _pad_to_union_columns(df)
            total_input_rows += len(df)
            bucket = _bucket_of(df, n_buckets).to_numpy()
            for b in range(n_buckets):
                mask = bucket == b
                if not mask.any():
                    continue
                table = pa.Table.from_pandas(df.loc[mask], schema=target_schema, preserve_index=False)
                if b not in writers:
                    writers[b] = pq.ParquetWriter(partition_paths[b], table.schema)
                writers[b].write_table(table)
            del df
    finally:
        for w in writers.values():
            w.close()

    # ---- gather ---------------------------------------------------------
    final_writer: pq.ParquetWriter | None = None
    total_output_rows = 0
    try:
        for b in range(n_buckets):
            path = partition_paths[b]
            if b not in writers or not path.exists():
                continue
            part_df = pd.read_parquet(path)
            if part_df.empty:
                continue
            result = _aggregate_union_frame(part_df)
            total_output_rows += len(result)
            table = pa.Table.from_pandas(result, preserve_index=False)
            if final_writer is None:
                final_writer = pq.ParquetWriter(output_path, table.schema)
            elif not table.schema.equals(final_writer.schema):
                table = table.cast(final_writer.schema)
            final_writer.write_table(table)
    finally:
        if final_writer is not None:
            final_writer.close()
    if final_writer is None:
        # No input rows at all -- still produce an (empty) valid output file.
        pq.ParquetWriter(
            output_path,
            pa.schema([("s1_entity_id", pa.string()), ("target_source", pa.string()), ("target_entity_id", pa.string())]),
        ).close()

    for path in partition_paths:
        if path.exists():
            path.unlink()
    tmp_dir.rmdir()

    return {
        "n_buckets": n_buckets,
        "total_input_rows": total_input_rows,
        "total_output_rows_before_cross_partition_merge": total_output_rows,
    }
