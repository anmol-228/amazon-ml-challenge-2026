"""Phase 2.5 Part E/F -- full-`development`-scale character n-gram forward
retrieval.

Justified by run_char_scale_calibration.py's measured verdict: GO at
max_df=0.0008 (vocabulary 73,448, real full-target-universe retrieval
estimated at 13.8 minutes -- not the earlier 200.8-minute estimate, which
used max_df=0.05 and a bounded 660K-row background pool). Part D found no
viable observable targeting rule (best rule captured only 6.18% of the
needs_char population at 0.81% flagged), so this runs broadly over ALL
dev-S1 entities against the REAL FULL target universe, exactly like every
other route in this project (no oracle gating -- truth is never used to
decide who gets this route).

This script does NOT touch or overwrite the existing EXP-B004 artifact.
Its output is a new, separate candidate table with its own `route_char`
provenance, to be unioned alongside B004 (not instead of it) by a
follow-up script.

Run: .venv/Scripts/python.exe scripts/run_char_full_scale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from blocking_data import development_s1_ids, load_train_data  # noqa: E402
from blocking_io import write_candidates, write_manifest  # noqa: E402
from blocking_normalization import joint_text  # noqa: E402
from blocking_resource import ResourceMonitor  # noqa: E402
from blocking_routes import forward_route  # noqa: E402
from blocking_vectorization import VectorizerConfig, fit_vectorizer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PHASE25_DIR = PROJECT_ROOT / "experiments" / "blocking" / "PHASE_2_5"

MAX_DF = 0.0008  # calibrated: run_char_scale_calibration.py measured GO here
MIN_DF = 2
CHAR_K = 50
BATCH_SIZE = 1500


def main() -> None:
    print("[CHAR-FULL] loading train data...")
    data = load_train_data()
    dev_ids = development_s1_ids(data)
    dev_set = set(dev_ids)

    print("[CHAR-FULL] building full joint-text corpus...")
    with ResourceMonitor() as mon_corpus:
        dev_s1_df = data.source1[data.source1["entity_id"].isin(dev_set)]
        dev_s1_texts = [
            joint_text(n, a) for n, a in zip(dev_s1_df["business_name"], dev_s1_df["business_address"])
        ]
        dev_s1_ids_ordered = dev_s1_df["entity_id"].tolist()
        target_ids_ordered = data.source2["entity_id"].tolist() + data.source3["entity_id"].tolist()
        target_texts = [
            joint_text(n, a) for n, a in zip(data.source2["business_name"], data.source2["business_address"])
        ] + [
            joint_text(n, a) for n, a in zip(data.source3["business_name"], data.source3["business_address"])
        ]
        full_corpus = dev_s1_texts + target_texts
    print(f"[CHAR-FULL] corpus built in {mon_corpus.report.wall_seconds:.1f}s, "
          f"peak {mon_corpus.report.peak_rss_mb:.0f}MB, {len(full_corpus):,} documents")

    print(f"[CHAR-FULL] fitting char-trigram vectorizer at max_df={MAX_DF} (calibrated)...")
    char_config = VectorizerConfig(
        representation="char_ngram", analyzer="char_wb", ngram_range=(3, 3),
        min_df=MIN_DF, max_df=MAX_DF, fit_id="phase25_char_full_scale_v1",
        corpus_scope="phase25_full_corpus",
    )
    with ResourceMonitor() as mon_fit:
        fitted = fit_vectorizer(full_corpus, char_config)
    print(f"[CHAR-FULL] fit done in {mon_fit.report.wall_seconds:.1f}s, peak {mon_fit.report.peak_rss_mb:.0f}MB, "
          f"vocab={fitted.provenance()['vocabulary_size']:,}")

    print("[CHAR-FULL] transforming S1 dev matrix + target matrix...")
    with ResourceMonitor() as mon_transform:
        s1_dev_matrix = fitted.vectorizer.transform(dev_s1_texts).tocsr()
        target_matrix = fitted.vectorizer.transform(target_texts).tocsr()
    print(f"[CHAR-FULL] transform done in {mon_transform.report.wall_seconds:.1f}s, "
          f"peak {mon_transform.report.peak_rss_mb:.0f}MB, "
          f"s1_nnz={s1_dev_matrix.nnz:,}, target_nnz={target_matrix.nnz:,}")
    del full_corpus, dev_s1_texts, target_texts

    print(f"[CHAR-FULL] running FULL forward retrieval: k={CHAR_K} over {len(dev_s1_ids_ordered):,} "
          f"queries x {len(target_ids_ordered):,} targets (the real full target universe)...")
    with ResourceMonitor() as mon_retrieve:
        char_df = forward_route(
            dev_s1_ids_ordered, s1_dev_matrix, target_ids_ordered, target_matrix, k=CHAR_K,
            batch_size=BATCH_SIZE, route_col="route_char", rank_col="char_rank", score_col="char_score",
        )
    print(f"[CHAR-FULL] retrieval done in {mon_retrieve.report.wall_seconds:.1f}s "
          f"({mon_retrieve.report.wall_seconds/60:.1f} min), peak {mon_retrieve.report.peak_rss_mb:.0f}MB, "
          f"rows={len(char_df):,}")

    PHASE25_DIR.mkdir(parents=True, exist_ok=True)
    write_candidates(char_df, PHASE25_DIR / "char_full_candidates.parquet")

    results = {
        "max_df": MAX_DF,
        "vocabulary_size": fitted.provenance()["vocabulary_size"],
        "corpus_resource": mon_corpus.report.as_dict(),
        "fit_resource": mon_fit.report.as_dict(),
        "transform_resource": mon_transform.report.as_dict(),
        "retrieval_resource": mon_retrieve.report.as_dict(),
        "n_candidate_rows": len(char_df),
        "n_queries": len(dev_s1_ids_ordered),
        "n_targets": len(target_ids_ordered),
    }
    write_manifest(results, PHASE25_DIR / "char_full_scale_results.json")
    print(f"[CHAR-FULL] done. Candidates written to "
          f"{PHASE25_DIR / 'char_full_candidates.parquet'}")


if __name__ == "__main__":
    main()
