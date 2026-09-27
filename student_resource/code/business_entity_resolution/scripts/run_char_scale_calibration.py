"""Phase 2.5 Part E -- character n-gram full-scale resource calibration.

EXP-B002's pilot used `max_df=0.05` for the char-trigram vectorizer -- the
same order of permissiveness that made the WORD vectorizer's own initial
`max_df=0.02` estimate full-scale forward retrieval at ~654 minutes
(BLOCKED) before retuning to `max_df=0.0008` cut that to ~11.5 minutes (a
57x measured speedup, see EXP-B000 round 2). This calibration checks
whether the same fix applies to the char vectorizer before concluding
full-scale character retrieval is infeasible:

1. Fit a char-trigram vectorizer on the REAL FULL corpus (all dev S1 +
   all S2 + all S3, ~12M documents) -- not a bounded pilot sample -- and
   measure its real document-frequency distribution and resource cost.
2. Transform the REAL full target matrix (10.32M rows, not a 660K-row
   bounded background pool).
3. Run a small retrieval pilot (a genuine random sample of dev S1 queries)
   against that REAL full target matrix, measuring true per-query cost
   against the real target universe -- the earlier 200.8-minute estimate
   extrapolated from a bounded pool and is not trusted here.
4. Extrapolate to full-`development`-scale retrieval time/memory from
   this measurement, and report a GO / GO WITH MODIFICATIONS / BLOCKED
   verdict, mirroring EXP-B000's own methodology.

This is measurement only -- no candidate table is unioned into anything
here; that is Part F's job, gated on this calibration's own verdict.

Run: .venv/Scripts/python.exe scripts/run_char_scale_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from blocking_data import deterministic_pilot_sample, development_s1_ids, load_train_data  # noqa: E402
from blocking_io import write_manifest  # noqa: E402
from blocking_normalization import joint_text  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402
from blocking_topk import batched_top_k  # noqa: E402
from blocking_vectorization import VectorizerConfig, fit_vectorizer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PHASE25_DIR = PROJECT_ROOT / "experiments" / "blocking" / "PHASE_2_5"

# Tried MOST-aggressive (safest) first, stopping at the first working
# configuration -- the reverse of "try the risky permissive setting at
# full scale first". 0.0008 matches the word route's own proven-successful
# retune; 0.05 (EXP-B002's pilot setting) is included only as a fallback
# if the aggressive end turns out to hurt recall too much to be useful,
# never as the first full-scale attempt.
CANDIDATE_MAX_DF_VALUES = [0.0008, 0.002, 0.01, 0.05]
RETRIEVAL_PILOT_N = 3000
RETRIEVAL_PILOT_K = 50
BATCH_SIZE = 500


def main() -> None:
    print("[CHAR-CAL] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)

    print("[CHAR-CAL] building full joint-text corpus (dev S1 + all S2 + all S3)...")
    with ResourceMonitor() as mon_corpus:
        dev_s1_df = data.source1[data.source1["entity_id"].isin(dev_set)]
        dev_s1_texts = [
            joint_text(n, a) for n, a in zip(dev_s1_df["business_name"], dev_s1_df["business_address"])
        ]
        dev_s1_ids_ordered = dev_s1_df["entity_id"].tolist()
        target_texts = [
            joint_text(n, a) for n, a in zip(data.source2["business_name"], data.source2["business_address"])
        ] + [
            joint_text(n, a) for n, a in zip(data.source3["business_name"], data.source3["business_address"])
        ]
        full_corpus = dev_s1_texts + target_texts
    print(f"[CHAR-CAL] corpus built in {mon_corpus.report.wall_seconds:.1f}s, "
          f"peak {mon_corpus.report.peak_rss_mb:.0f}MB, {len(full_corpus):,} documents")

    # ---- try candidate max_df values, measuring real vocab/nnz -------------
    calibration_results = []
    chosen = None
    for max_df in CANDIDATE_MAX_DF_VALUES:
        print(f"[CHAR-CAL] fitting char-trigram vectorizer at max_df={max_df}...")
        char_config = VectorizerConfig(
            representation="char_ngram", analyzer="char_wb", ngram_range=(3, 3),
            min_df=2, max_df=max_df, fit_id=f"phase25_char_full_maxdf{max_df}",
            corpus_scope="phase25_full_corpus",
        )
        try:
            with ResourceMonitor() as mon_fit:
                fitted = fit_vectorizer(full_corpus, char_config)
        except MemoryError:
            print(f"[CHAR-CAL] max_df={max_df}: MemoryError during fit -- skipping")
            calibration_results.append({"max_df": max_df, "status": "MEMORY_ERROR_FIT"})
            continue

        prov = fitted.provenance()
        entry = {
            "max_df": max_df,
            "vocabulary_size": prov["vocabulary_size"],
            "fit_resource": mon_fit.report.as_dict(),
        }
        print(f"[CHAR-CAL] max_df={max_df}: vocab={prov['vocabulary_size']:,}, "
              f"fit_time={mon_fit.report.wall_seconds:.1f}s, peak={mon_fit.report.peak_rss_mb:.0f}MB")

        # Transform target matrix (real full scale) + a query pilot sample.
        try:
            with ResourceMonitor() as mon_transform:
                target_matrix = fitted.vectorizer.transform(target_texts).tocsr()
        except MemoryError:
            print(f"[CHAR-CAL] max_df={max_df}: MemoryError during target transform -- skipping")
            entry["status"] = "MEMORY_ERROR_TRANSFORM"
            calibration_results.append(entry)
            continue

        entry["target_matrix_nnz"] = int(target_matrix.nnz)
        entry["transform_resource"] = mon_transform.report.as_dict()
        print(f"[CHAR-CAL] max_df={max_df}: target matrix nnz={target_matrix.nnz:,}, "
              f"transform_time={mon_transform.report.wall_seconds:.1f}s, peak={mon_transform.report.peak_rss_mb:.0f}MB")

        pilot_ids = deterministic_pilot_sample(data, n=RETRIEVAL_PILOT_N, split="development")
        pos_map = {eid: i for i, eid in enumerate(dev_s1_ids_ordered)}
        pilot_pos = [pos_map[pid] for pid in pilot_ids if pid in pos_map]
        pilot_texts = [dev_s1_texts[i] for i in pilot_pos]
        pilot_query_matrix = fitted.vectorizer.transform(pilot_texts).tocsr()

        try:
            with ResourceMonitor() as mon_retrieve:
                result = batched_top_k(
                    pilot_query_matrix, target_matrix, k=RETRIEVAL_PILOT_K, batch_size=BATCH_SIZE
                )
        except MemoryError:
            print(f"[CHAR-CAL] max_df={max_df}: MemoryError during retrieval pilot -- skipping")
            entry["status"] = "MEMORY_ERROR_RETRIEVAL"
            calibration_results.append(entry)
            continue

        avg_candidates = float(np.mean(np.diff(result.offsets)))
        seconds_per_query = mon_retrieve.report.wall_seconds / max(1, len(pilot_pos))
        est_full_minutes = (seconds_per_query * len(dev_ids)) / 60.0

        entry["retrieval_pilot_resource"] = mon_retrieve.report.as_dict()
        entry["retrieval_pilot_n_queries"] = len(pilot_pos)
        entry["avg_topk_candidates_per_query"] = avg_candidates
        entry["estimated_full_scale_retrieval_minutes_REAL_full_target_universe"] = est_full_minutes
        entry["status"] = "MEASURED"

        peak_mb = max(
            mon_fit.report.peak_rss_mb, mon_transform.report.peak_rss_mb, mon_retrieve.report.peak_rss_mb
        )
        if peak_mb > 28000 or est_full_minutes > 180:
            verdict = "BLOCKED"
        elif peak_mb > 20000 or est_full_minutes > 60:
            verdict = "GO_WITH_MODIFICATIONS"
        else:
            verdict = "GO"
        entry["verdict"] = verdict
        print(f"[CHAR-CAL] max_df={max_df}: REAL full-scale estimate={est_full_minutes:.1f} min, "
              f"peak_rss={peak_mb:.0f}MB, verdict={verdict}")

        calibration_results.append(entry)
        del target_matrix, pilot_query_matrix
        if verdict in ("GO", "GO_WITH_MODIFICATIONS"):
            chosen = entry
            print(f"[CHAR-CAL] max_df={max_df} verdict={verdict} -- stopping search here "
                  f"(most-aggressive-first order; not trying more permissive values).")
            break

    results = {
        "candidate_max_df_values_tried": CANDIDATE_MAX_DF_VALUES,
        "calibration_results": calibration_results,
        "chosen_config": chosen,
        "final_verdict": chosen["verdict"] if chosen else "BLOCKED",
    }
    PHASE25_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(results, PHASE25_DIR / "char_scale_calibration.json")
    print(f"[CHAR-CAL] done. Final verdict: {results['final_verdict']}")
    if chosen:
        print(f"[CHAR-CAL] chosen max_df={chosen['max_df']}, "
              f"estimated full-scale retrieval={chosen.get('estimated_full_scale_retrieval_minutes_REAL_full_target_universe', 'n/a')} min")


if __name__ == "__main__":
    main()
