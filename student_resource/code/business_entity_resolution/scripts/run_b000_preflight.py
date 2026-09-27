"""EXP-B000 -- blocking scale / collision preflight.

Engineering gate only (Phase-2 mission spec section 13): measures resource
and collision risk before EXP-B001-B004 are run at full `development` scale.
As a side effect, this script also builds and caches the shared production
word-TF-IDF vectorizer + transformed matrices (dev-S1 and full-S2+S3) that
EXP-B001/EXP-B001R/EXP-B004 reuse, since fitting that vectorizer over the
real corpus is itself the most informative preflight measurement available
(a real document-frequency distribution, not a pilot extrapolation of one).

Run: .venv/Scripts/python.exe scripts/run_b000_preflight.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

from blocking_data import deterministic_pilot_sample, development_s1_ids, load_train_data  # noqa: E402
from blocking_io import write_manifest  # noqa: E402
from blocking_normalization import joint_text  # noqa: E402
from blocking_numeric import build_numeric_postings, numeric_collision_group_sizes  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402
from blocking_topk import batched_top_k  # noqa: E402
from blocking_vectorization import VectorizerConfig, fit_vectorizer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "blocking" / "B000"
CACHE_DIR = PROJECT_ROOT / "experiments" / "blocking" / "cache"

PILOT_QUERY_N = 3000
PILOT_BATCH_SIZE = 3000
WORD_MAX_DF = 0.0008
WORD_MIN_DF = 3
CHAR_NGRAM_RANGE = (3, 3)
CHAR_MAX_DF = 0.02
CHAR_MIN_DF = 2
CHAR_PILOT_DOC_N = 60000

MACHINE_TOTAL_RAM_MB = 32374.0


def top_df_tokens(vectorizer, matrix: sparse.csr_matrix, top_n: int = 20) -> list[dict]:
    df_counts = np.asarray((matrix > 0).sum(axis=0)).ravel()
    order = np.argsort(-df_counts)[:top_n]
    vocab_inv = {v: k for k, v in vectorizer.vocabulary_.items()}
    n_docs = matrix.shape[0]
    return [
        {"token": vocab_inv[i], "df_count": int(df_counts[i]), "df_fraction": float(df_counts[i] / n_docs)}
        for i in order
    ]


def main() -> None:
    report: dict = {"experiment": "B000", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print("[B000] loading train data...")
    with ResourceMonitor() as mon_load:
        data = load_train_data()
    report["load_train_data"] = mon_load.report.as_dict()

    dev_ids = development_s1_ids(data)
    report["n_development_s1"] = len(dev_ids)
    report["n_s2_total"] = len(data.source2)
    report["n_s3_total"] = len(data.source3)

    print(f"[B000] development S1 = {len(dev_ids):,}, S2 = {len(data.source2):,}, S3 = {len(data.source3):,}")

    # ---- Build the real production corpus (dev S1 + all S2 + all S3) -----
    print("[B000] building joint-text corpus (measured, not extrapolated)...")
    with ResourceMonitor() as mon_corpus:
        dev_s1_df = data.source1[data.source1["entity_id"].isin(set(dev_ids))]
        dev_s1_texts = [
            joint_text(n, a) for n, a in zip(dev_s1_df["business_name"], dev_s1_df["business_address"])
        ]
        dev_s1_ids_ordered = dev_s1_df["entity_id"].tolist()

        s2_texts = [
            joint_text(n, a) for n, a in zip(data.source2["business_name"], data.source2["business_address"])
        ]
        s3_texts = [
            joint_text(n, a) for n, a in zip(data.source3["business_name"], data.source3["business_address"])
        ]
        target_ids_ordered = data.source2["entity_id"].tolist() + data.source3["entity_id"].tolist()
        target_texts = s2_texts + s3_texts
        full_corpus_texts = dev_s1_texts + target_texts
    report["corpus_build"] = mon_corpus.report.as_dict()
    report["n_target_total"] = len(target_texts)

    # ---- Fit the real word vectorizer over the real corpus ---------------
    print("[B000] fitting word TF-IDF vectorizer over full corpus...")
    word_config = VectorizerConfig(
        representation="word",
        min_df=WORD_MIN_DF,
        max_df=WORD_MAX_DF,
        fit_id="b000_b001_shared_word_v1",
    )
    with ResourceMonitor() as mon_fit:
        fitted_word = fit_vectorizer(full_corpus_texts, word_config)
    report["word_vectorizer_fit"] = mon_fit.report.as_dict()
    report["word_vectorizer_provenance"] = fitted_word.provenance()

    with ResourceMonitor() as mon_transform:
        s1_dev_matrix = fitted_word.vectorizer.transform(dev_s1_texts).tocsr()
        target_matrix = fitted_word.vectorizer.transform(target_texts).tocsr()
    report["word_transform"] = mon_transform.report.as_dict()
    report["s1_dev_matrix_shape"] = list(s1_dev_matrix.shape)
    report["target_matrix_shape"] = list(target_matrix.shape)
    report["s1_dev_matrix_nnz"] = int(s1_dev_matrix.nnz)
    report["target_matrix_nnz"] = int(target_matrix.nnz)
    s1_row_nnz = np.diff(s1_dev_matrix.indptr)
    target_row_nnz = np.diff(target_matrix.indptr)
    report["s1_dev_empty_row_fraction"] = float(np.mean(s1_row_nnz == 0))
    report["target_empty_row_fraction"] = float(np.mean(target_row_nnz == 0))
    report["s1_dev_row_nnz_mean"] = float(np.mean(s1_row_nnz))
    report["target_row_nnz_mean"] = float(np.mean(target_row_nnz))
    est_bytes = (s1_dev_matrix.nnz + target_matrix.nnz) * 8  # float32 data + int32 index, approx
    report["word_matrices_estimated_bytes"] = est_bytes

    report["word_top_document_frequency_tokens"] = top_df_tokens(fitted_word.vectorizer, target_matrix, 20)

    # cache to disk for B001/B001R/B004 reuse
    print("[B000] caching fitted word vectorizer + matrices...")
    import pickle

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "word_vectorizer.pkl", "wb") as fh:
        pickle.dump(fitted_word.vectorizer, fh)
    sparse.save_npz(CACHE_DIR / "s1_dev_word_matrix.npz", s1_dev_matrix)
    sparse.save_npz(CACHE_DIR / "target_word_matrix.npz", target_matrix)
    write_manifest(
        {
            "dev_s1_ids": dev_s1_ids_ordered,
            "target_ids": target_ids_ordered,
            "n_s2": len(data.source2),
            "n_s3": len(data.source3),
            "provenance": fitted_word.provenance(),
        },
        CACHE_DIR / "word_cache_manifest.json",
    )

    # ---- Retrieval pilot: measure per-batch throughput/memory ------------
    print(f"[B000] retrieval pilot: {PILOT_QUERY_N} dev-S1 queries against full target matrix...")
    pilot_ids = deterministic_pilot_sample(data, n=PILOT_QUERY_N, split="development")
    dev_s1_pos_map = {eid: i for i, eid in enumerate(dev_s1_ids_ordered)}
    pilot_pos = [dev_s1_pos_map[pid] for pid in pilot_ids if pid in dev_s1_pos_map]
    pilot_matrix = s1_dev_matrix[pilot_pos]

    with ResourceMonitor() as mon_retrieval:
        result = batched_top_k(pilot_matrix, target_matrix, k=20, batch_size=PILOT_BATCH_SIZE)
    report["retrieval_pilot"] = mon_retrieval.report.as_dict()
    report["retrieval_pilot_n_queries"] = len(pilot_pos)
    avg_candidates = float(np.mean([len(idx) for idx in result.indices])) if result.indices else 0.0
    report["retrieval_pilot_avg_topk_candidates_per_query"] = avg_candidates

    seconds_per_query = mon_retrieval.report.wall_seconds / max(1, len(pilot_pos))
    est_full_forward_seconds = seconds_per_query * len(dev_ids)
    report["estimated_full_forward_seconds"] = est_full_forward_seconds
    report["estimated_full_forward_minutes"] = est_full_forward_seconds / 60.0

    # reverse cost estimate: queries = all targets (much larger than dev S1)
    seconds_per_target_query = seconds_per_query  # same per-query cost order, same representation
    est_full_reverse_seconds = seconds_per_target_query * len(target_texts)
    report["estimated_full_reverse_seconds"] = est_full_reverse_seconds
    report["estimated_full_reverse_minutes"] = est_full_reverse_seconds / 60.0

    # ---- Character n-gram footprint estimate (pilot doc sample) ----------
    print("[B000] estimating character n-gram footprint on a pilot doc sample...")
    rng = np.random.RandomState(42)
    char_pilot_idx = rng.choice(len(target_texts), size=min(CHAR_PILOT_DOC_N, len(target_texts)), replace=False)
    char_pilot_texts = [target_texts[i] for i in char_pilot_idx] + dev_s1_texts[: min(10000, len(dev_s1_texts))]
    char_config = VectorizerConfig(
        representation="char_ngram",
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=CHAR_MIN_DF,
        max_df=CHAR_MAX_DF,
        fit_id="b000_char_pilot_v1",
    )
    with ResourceMonitor() as mon_char:
        fitted_char = fit_vectorizer(char_pilot_texts, char_config)
        char_pilot_matrix = fitted_char.vectorizer.transform(char_pilot_texts)
    report["char_pilot_fit"] = mon_char.report.as_dict()
    report["char_pilot_provenance"] = fitted_char.provenance()
    report["char_pilot_matrix_nnz"] = int(char_pilot_matrix.nnz)
    nnz_per_doc = char_pilot_matrix.nnz / char_pilot_matrix.shape[0]
    est_char_full_nnz = nnz_per_doc * (len(dev_ids) + len(target_texts))
    report["char_estimated_full_nnz"] = est_char_full_nnz
    report["char_estimated_full_bytes"] = est_char_full_nnz * 8

    # ---- Numeric-token collision estimate (full, cheap) -------------------
    print("[B000] measuring numeric-token postings on full S2+S3 addresses...")
    with ResourceMonitor() as mon_numeric:
        postings = build_numeric_postings(target_ids_ordered, [
            a for a in list(data.source2["business_address"]) + list(data.source3["business_address"])
        ])
        sizes = numeric_collision_group_sizes(postings)
    report["numeric_postings"] = mon_numeric.report.as_dict()
    report["numeric_vocab_size"] = len(postings)
    report["numeric_largest_groups"] = sizes.head(20).to_dict()
    report["numeric_group_size_p50_p95_p99_max"] = {
        "p50": float(np.percentile(sizes.values, 50)),
        "p95": float(np.percentile(sizes.values, 95)),
        "p99": float(np.percentile(sizes.values, 99)),
        "max": float(sizes.values.max()),
    }

    # ---- Full-scale candidate-row / disk estimate --------------------------
    est_candidate_rows_forward = avg_candidates * len(dev_ids)
    report["estimated_forward_candidate_rows"] = est_candidate_rows_forward
    bytes_per_row_estimate = 60  # rough: ids + rank + score, parquet-compressed order of magnitude
    report["estimated_forward_disk_bytes"] = est_candidate_rows_forward * bytes_per_row_estimate

    # ---- GO / GO WITH MODIFICATIONS / BLOCKED per route --------------------
    peak_word_mb = max(
        report["load_train_data"]["peak_rss_mb"],
        report["corpus_build"]["peak_rss_mb"],
        report["word_vectorizer_fit"]["peak_rss_mb"],
        report["word_transform"]["peak_rss_mb"],
        report["retrieval_pilot"]["peak_rss_mb"],
    )
    headroom_fraction = peak_word_mb / MACHINE_TOTAL_RAM_MB

    def decide(headroom: float, est_minutes: float) -> str:
        if headroom > 0.85 or est_minutes > 240:
            return "BLOCKED"
        if headroom > 0.6 or est_minutes > 60:
            return "GO WITH MODIFICATIONS"
        return "GO"

    decisions = {
        "B001_forward_word": decide(headroom_fraction, report["estimated_full_forward_minutes"]),
        "B001R_reverse_word": decide(headroom_fraction, report["estimated_full_reverse_minutes"]),
        "B002_char_ngram": decide(
            report["char_pilot_fit"]["peak_rss_mb"] / MACHINE_TOTAL_RAM_MB,
            report["char_estimated_full_bytes"] / (1024 * 1024 * 60000),
        ),
        "B003_numeric": "GO",  # measured full-scale already, cheap by construction
    }
    report["route_decisions"] = decisions
    report["machine_total_ram_mb"] = MACHINE_TOTAL_RAM_MB
    report["peak_rss_mb_observed"] = peak_word_mb
    report["headroom_fraction_used"] = headroom_fraction

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(report, EXPERIMENT_DIR / "b000_results.json")
    print(json.dumps(decisions, indent=2))
    print(f"[B000] full report written to {EXPERIMENT_DIR / 'b000_results.json'}")


if __name__ == "__main__":
    main()
