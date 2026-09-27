"""Phase-3 EXP-F001 lexical/numeric feature construction.

Reads the pruned+labeled candidate files written by run_phase3_prune_label.py
(experiments/phase3/pruned/{matcher_train,dev_eval,calibration_holdout}.parquet
-- see that script's docstring for the frozen pruning rule and provenance),
streams each in batches, joins normalized name/address text and precomputed
token sets by dict lookup (built once, in memory, from the raw train source
tables), and writes an augmented feature parquet per split under
experiments/phase3/features/. The identical function (`build_lookup_tables`,
`compute_batch_features`) is reused for TEST inference in
run_phase3_test_pipeline.py against the equivalent test-side lookup tables,
so dev and test feature computation share one code path.

Feature inventory: see docs/research/RESEARCH_TO_EXPERIMENT_PLAN.md
EXP-F001 for the full designed target list; this is the time-boxed subset
implemented for submission #1 (richer teammate-reconciliation features are
POST_SUBMISSION_HIGH_PRIORITY, not implemented here):
  - route_* (pass-through), forward_rank/reverse_rank/forward_score/reverse_score
    (pass-through, NaN filled with a sentinel), n_routes_support, best_rank
  - name_exact, address_exact: normalized-field equality
  - name_ratio, address_ratio: RapidFuzz Levenshtein ratio (0-1) on normalized text
  - name_jw, address_jw: RapidFuzz Jaro-Winkler similarity (0-1)
  - name_token_jaccard: word-token Jaccard on business_name
  - name_len_ratio, address_len_ratio: min(len)/max(len), 1.0 if both empty
  - numeric_jaccard: numeric-token-set Jaccard (address)
  - numeric_conflict: both sides have >=1 numeric token and they are fully disjoint
  - numeric_all_equal: numeric token sets identical and non-empty (redundant
    with route_numeric_exact_set but kept as an explicit float feature)
  - is_source2: target_source == "S2"
  - country_agree: normalized country strings equal
  - target_name_rarity: how many TRAIN target-universe rows share this
    exact normalized name (collision-group-size proxy, log1p-scaled).
    **Leakage-safety note (repair applied after independent review flagged
    it as a methodology ambiguity):** this count is frozen from TRAIN
    Source-2/3 only (`train_name_rarity_counts()`), looked up by normalized
    NAME STRING (not target_entity_id, whose namespace differs between
    train and test), and reused unchanged for matcher_train/dev_eval/
    calibration_holdout/test alike. A target name never seen in train's own
    S2/S3 population (this can only happen for test rows, since dev/train
    rows are themselves drawn from that same population) falls back to a
    count of 0 (log1p(0)=0), the most-rare bucket -- deterministic, and not
    a fitted parameter or a statistic computed from test's own frequency
    distribution.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from blocking_normalization import normalize_field, numeric_tokens, word_tokens  # noqa: E402
from data_io import TRAIN_PATHS, load_source_table  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PRUNED_DIR = PROJECT_ROOT / "experiments" / "phase3" / "pruned"
OUT_DIR = PROJECT_ROOT / "experiments" / "phase3" / "features"

BATCH_ROWS = 2_000_000
RANK_SENTINEL = 999.0
SCORE_SENTINEL = 0.0


def make_attr_dicts(df: pd.DataFrame):
    ids = df["entity_id"].tolist()
    names_norm = [normalize_field(n) for n in df["business_name"].tolist()]
    addrs_norm = [normalize_field(a) for a in df["business_address"].tolist()]
    countries = [normalize_field(c) for c in df["country"].tolist()]
    name_tok = [frozenset(word_tokens(n)) for n in names_norm]
    addr_num = [frozenset(numeric_tokens(a)) for a in addrs_norm]
    return {
        "name_norm": dict(zip(ids, names_norm)),
        "addr_norm": dict(zip(ids, addrs_norm)),
        "country": dict(zip(ids, countries)),
        "name_tok": dict(zip(ids, name_tok)),
        "addr_num": dict(zip(ids, addr_num)),
    }


def build_lookup_tables(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
    t0 = time.time()
    s1_attrs = make_attr_dicts(s1_df)
    target_df = pd.concat([s2_df, s3_df], axis=0, ignore_index=True)
    target_attrs = make_attr_dicts(target_df)

    print(f"[features] lookup tables built in {time.time()-t0:.1f}s: "
          f"s1={len(s1_attrs['name_norm']):,} targets={len(target_attrs['name_norm']):,}")
    return s1_attrs, target_attrs


def build_train_lookup_tables():
    s1 = load_source_table(TRAIN_PATHS["source1"])
    s2 = load_source_table(TRAIN_PATHS["source2"])
    s3 = load_source_table(TRAIN_PATHS["source3"])
    return build_lookup_tables(s1, s2, s3)


def train_name_rarity_counts() -> dict[str, int]:
    """Frozen name-collision-group-size lookup, keyed by normalized name
    string, built ONLY from train Source-2/3 (never test). Reused unchanged
    for every split's `target_name_rarity` feature -- see this module's
    docstring for the leakage-safety rationale."""
    from collections import Counter

    t0 = time.time()
    s2 = load_source_table(TRAIN_PATHS["source2"], usecols=["entity_id", "business_name"])
    s3 = load_source_table(TRAIN_PATHS["source3"], usecols=["entity_id", "business_name"])
    names = [normalize_field(n) for n in list(s2["business_name"]) + list(s3["business_name"])]
    counts = dict(Counter(names))
    print(f"[features] train_name_rarity_counts built in {time.time()-t0:.1f}s "
          f"({len(names):,} names, {len(counts):,} distinct)")
    return counts


def compute_batch_features(df: pd.DataFrame, s1_attrs, target_attrs, name_rarity_counts: dict[str, int]) -> pd.DataFrame:
    """Add lexical/numeric/context feature columns to `df` in place-ish
    (returns a new DataFrame with the original columns preserved plus the
    new feature columns). `df` must already have route/rank/score columns,
    plus `label` and `phase3_split` if present (both simply pass through
    unchanged -- test-side callers omit `label`)."""
    s1_ids = df["s1_entity_id"].to_numpy()
    target_ids = df["target_entity_id"].to_numpy()
    n = len(df)

    s1_name, s1_addr, s1_country = s1_attrs["name_norm"], s1_attrs["addr_norm"], s1_attrs["country"]
    s1_ntok, s1_anum = s1_attrs["name_tok"], s1_attrs["addr_num"]
    t_name, t_addr, t_country = target_attrs["name_norm"], target_attrs["addr_norm"], target_attrs["country"]
    t_ntok, t_anum = target_attrs["name_tok"], target_attrs["addr_num"]

    name_ratio = np.empty(n, dtype=np.float32)
    addr_ratio = np.empty(n, dtype=np.float32)
    name_jw = np.empty(n, dtype=np.float32)
    addr_jw = np.empty(n, dtype=np.float32)
    name_exact = np.empty(n, dtype=bool)
    addr_exact = np.empty(n, dtype=bool)
    name_jacc = np.empty(n, dtype=np.float32)
    name_len_ratio = np.empty(n, dtype=np.float32)
    addr_len_ratio = np.empty(n, dtype=np.float32)
    num_jacc = np.empty(n, dtype=np.float32)
    num_conflict = np.empty(n, dtype=bool)
    num_all_equal = np.empty(n, dtype=bool)
    country_agree = np.empty(n, dtype=bool)
    rarity = np.empty(n, dtype=np.float32)

    for i in range(n):
        sid = s1_ids[i]
        tid = target_ids[i]
        sn, tn = s1_name.get(sid, ""), t_name.get(tid, "")
        sa, ta = s1_addr.get(sid, ""), t_addr.get(tid, "")

        name_ratio[i] = fuzz.ratio(sn, tn) / 100.0
        addr_ratio[i] = fuzz.ratio(sa, ta) / 100.0
        name_jw[i] = JaroWinkler.similarity(sn, tn) if sn and tn else 0.0
        addr_jw[i] = JaroWinkler.similarity(sa, ta) if sa and ta else 0.0
        name_exact[i] = sn == tn and sn != ""
        addr_exact[i] = sa == ta and sa != ""

        stok, ttok = s1_ntok.get(sid, frozenset()), t_ntok.get(tid, frozenset())
        union = stok | ttok
        name_jacc[i] = (len(stok & ttok) / len(union)) if union else 0.0

        ls, lt = len(sn), len(tn)
        name_len_ratio[i] = (min(ls, lt) / max(ls, lt)) if max(ls, lt) > 0 else 1.0
        las, lat = len(sa), len(ta)
        addr_len_ratio[i] = (min(las, lat) / max(las, lat)) if max(las, lat) > 0 else 1.0

        snum, tnum = s1_anum.get(sid, frozenset()), t_anum.get(tid, frozenset())
        num_union = snum | tnum
        num_jacc[i] = (len(snum & tnum) / len(num_union)) if num_union else 0.0
        num_conflict[i] = bool(snum) and bool(tnum) and len(snum & tnum) == 0
        num_all_equal[i] = bool(snum) and snum == tnum

        country_agree[i] = s1_country.get(sid, "") == t_country.get(tid, "")
        rarity[i] = np.log1p(name_rarity_counts.get(tn, 0))

    feat = pd.DataFrame(
        {
            "name_exact": name_exact,
            "address_exact": addr_exact,
            "name_ratio": name_ratio,
            "address_ratio": addr_ratio,
            "name_jw": name_jw,
            "address_jw": addr_jw,
            "name_token_jaccard": name_jacc,
            "name_len_ratio": name_len_ratio,
            "address_len_ratio": addr_len_ratio,
            "numeric_jaccard": num_jacc,
            "numeric_conflict": num_conflict,
            "numeric_all_equal": num_all_equal,
            "is_source2": (df["target_source"] == "S2").to_numpy(),
            "country_agree": country_agree,
            "target_name_rarity": rarity,
        }
    )

    out = pd.concat([df.reset_index(drop=True), feat], axis=1)
    out["forward_rank"] = out["forward_rank"].fillna(RANK_SENTINEL).astype(np.float32)
    out["reverse_rank"] = out["reverse_rank"].fillna(RANK_SENTINEL).astype(np.float32)
    out["forward_score"] = out["forward_score"].fillna(SCORE_SENTINEL).astype(np.float32)
    out["reverse_score"] = out["reverse_score"].fillna(SCORE_SENTINEL).astype(np.float32)
    out["n_routes_support"] = (
        out["route_exact_name_address"].astype(np.int8)
        + out["route_forward_word"].astype(np.int8)
        + out["route_reverse_word"].astype(np.int8)
        + out["route_numeric_exact_set"].astype(np.int8)
    )
    out["best_rank"] = np.minimum(out["forward_rank"].values, out["reverse_rank"].values)
    return out


def process_file(in_path: Path, out_path: Path, s1_attrs, target_attrs, name_rarity_counts) -> None:
    t0 = time.time()
    pf = pq.ParquetFile(in_path)
    total_rows = pf.metadata.num_rows
    writer = None
    processed = 0
    for batch_i, batch in enumerate(pf.iter_batches(batch_size=BATCH_ROWS), start=1):
        t_b = time.time()
        df = batch.to_pandas()
        feats = compute_batch_features(df, s1_attrs, target_attrs, name_rarity_counts)
        table = pa.Table.from_pandas(feats, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        processed += len(df)
        print(f"[features:{in_path.stem}] batch {batch_i}: {len(df):,} rows in {time.time()-t_b:.1f}s "
              f"({processed:,}/{total_rows:,} = {100*processed/max(1,total_rows):.1f}%) "
              f"elapsed {time.time()-t0:.1f}s")
    if writer is not None:
        writer.close()
    print(f"[features:{in_path.stem}] done, {processed:,} rows, {time.time()-t0:.1f}s -> {out_path}")


def main() -> None:
    t_start = time.time()
    s1_attrs, target_attrs = build_train_lookup_tables()
    name_rarity_counts = train_name_rarity_counts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ("calibration_holdout", "dev_eval", "matcher_train"):
        in_path = PRUNED_DIR / f"{split_name}.parquet"
        out_path = OUT_DIR / f"{split_name}.parquet"
        process_file(in_path, out_path, s1_attrs, target_attrs, name_rarity_counts)
    print(f"[features] ALL DONE in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
