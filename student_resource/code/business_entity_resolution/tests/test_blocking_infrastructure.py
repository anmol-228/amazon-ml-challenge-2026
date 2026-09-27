"""Unit tests for the Phase-2 blocking infrastructure (src/blocking_*.py).

Uses tiny synthetic data with manually-obvious expected answers throughout,
per the Phase-2 mission spec section 26 ("no metric should be trusted simply
because it returns a plausible number").
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import sparse  # noqa: E402

from blocking_exact import exact_candidates  # noqa: E402
from blocking_identity import (  # noqa: E402
    assert_no_cross_source_collision,
    candidate_key,
    target_source_of,
)
from blocking_io import read_candidates, read_manifest, write_candidates, write_manifest  # noqa: E402
from blocking_metrics import (  # noqa: E402
    build_id_sets_by_s1,
    candidate_count_per_s1,
    compute_blocking_metrics,
    compute_blocking_metrics_scalable,
    intersection_count_per_s1,
    link_found_mask,
    route_incremental_value,
    route_incremental_value_scalable,
    truth_edges_dataframe,
    union_count_per_s1,
)
from blocking_normalization import joint_text, normalize_field, numeric_tokens, word_tokens  # noqa: E402
from blocking_numeric import (  # noqa: E402
    exact_numeric_signature_candidates,
    numeric_candidates,
    numeric_collision_group_sizes,
    numeric_signature,
)
from blocking_routes import forward_route, reverse_route  # noqa: E402
from blocking_topk import batched_top_k  # noqa: E402
from blocking_union import union_candidates, union_candidates_partitioned  # noqa: E402
from blocking_vectorization import VectorizerConfig, fit_vectorizer  # noqa: E402


# --- 1. normalization compatibility -----------------------------------------

def test_normalize_field_basic():
    assert normalize_field("  Café   BAKERY ") == "café bakery"
    assert normalize_field("") == ""
    assert normalize_field(None) == ""


def test_joint_text_handles_missing_address():
    assert joint_text("Acme Inc", "") == "acme inc"
    assert joint_text("", "12 Main St") == "12 main st"
    assert joint_text("", "") == ""


def test_word_and_numeric_tokens():
    assert word_tokens(normalize_field("12 Main St, Suite 4")) == ["12", "main", "st", "suite", "4"]
    assert numeric_tokens("12 Main St, Suite 4") == ["12", "4"]


# --- 2/19/20. top-k retrieval, chunk invariance, determinism ----------------

def _toy_matrix(rows: list[list[float]]) -> sparse.csr_matrix:
    arr = np.array(rows, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return sparse.csr_matrix(arr)


def test_batched_top_k_known_answer():
    # 3 queries, 4 targets, obvious nearest neighbor by construction.
    queries = _toy_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    targets = _toy_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    result = batched_top_k(queries, targets, k=2, batch_size=2)
    idx0, scr0 = result.row(0)
    idx1, _ = result.row(1)
    idx2, _ = result.row(2)
    assert idx0[0] == 0  # query 0 best match is target 0
    assert idx1[0] == 1
    assert idx2[0] == 2
    assert np.isclose(scr0[0], 1.0)


def test_batched_top_k_chunk_invariant():
    rng = np.random.RandomState(0)
    dense = (rng.rand(37, 15) > 0.7).astype("float32")
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms
    mat = sparse.csr_matrix(dense)

    r1 = batched_top_k(mat, mat, k=5, batch_size=3)
    r2 = batched_top_k(mat, mat, k=5, batch_size=37)
    for i in range(37):
        idx1, scr1 = r1.row(i)
        idx2, scr2 = r2.row(i)
        assert list(idx1) == list(idx2)
        assert np.allclose(scr1, scr2)


# --- 3/4. forward retrieval correctness, reverse inversion ------------------

def test_forward_and_reverse_route_roundtrip():
    s1_ids = ["S1-1", "S1-2"]
    target_ids = ["S2-1", "S3-1", "S2-2"]
    s1_matrix = _toy_matrix([[1, 0], [0, 1]])
    target_matrix = _toy_matrix([[1, 0], [0, 1], [1, 1]])

    fwd = forward_route(s1_ids, s1_matrix, target_ids, target_matrix, k=2)
    assert set(fwd.columns) >= {"s1_entity_id", "target_entity_id", "forward_rank", "forward_score", "route_forward_word"}
    s1_1_targets = set(fwd[fwd.s1_entity_id == "S1-1"].target_entity_id)
    assert "S2-1" in s1_1_targets  # exact direction match must be retrieved

    rev = reverse_route(s1_ids, s1_matrix, target_ids, target_matrix, r=1)
    assert set(rev.columns) >= {"s1_entity_id", "target_entity_id", "reverse_rank", "reverse_score", "route_reverse_word"}
    # S2-1 (== [1,0]) must invert back to S1-1.
    row = rev[rev.target_entity_id == "S2-1"]
    assert row.s1_entity_id.iloc[0] == "S1-1"


# --- 5/6/22. composite identity, cross-source collision, source validity ---

def test_target_source_of_and_candidate_key():
    assert target_source_of("S2-123") == "S2"
    assert target_source_of("S3-999") == "S3"
    with pytest.raises(ValueError):
        target_source_of("S1-1")
    assert candidate_key("S1-1", "S2", "S2-9") == ("S1-1", "S2", "S2-9")
    with pytest.raises(ValueError):
        candidate_key("S1-1", "S1", "S1-9")


def test_no_cross_source_collision_detection():
    assert_no_cross_source_collision({"S2-1", "S2-2"}, {"S3-1", "S3-2"})
    with pytest.raises(ValueError):
        assert_no_cross_source_collision({"S2-1", "DUP-1"}, {"DUP-1", "S3-2"})


# --- 7. exact sanity route --------------------------------------------------

def test_exact_candidates_route():
    df = exact_candidates(
        s1_ids=["S1-1", "S1-2"],
        s1_names=["Acme Inc", "Totally Different"],
        s1_addresses=["12 Main St", "99 Other Rd"],
        target_ids=["S2-1", "S2-2"],
        target_names=["ACME   inc", "Totally Different"],
        target_addresses=["12   main st", "1 Elsewhere Ave"],
    )
    row1 = df[df.s1_entity_id == "S1-1"]
    assert row1.iloc[0]["route_exact_name_address"] is True or row1.iloc[0]["route_exact_name_address"] == True
    assert row1.iloc[0]["route_exact_name"] == True
    row2 = df[(df.s1_entity_id == "S1-2") & (df.target_entity_id == "S2-2")]
    assert row2.iloc[0]["route_exact_name"] == True
    assert row2.iloc[0]["route_exact_name_address"] == False


# --- numeric route + collision groups ---------------------------------------

def test_numeric_candidates_and_collisions():
    df = numeric_candidates(
        s1_ids=["S1-1"],
        s1_addresses=["12 Main St"],
        target_ids=["S2-1", "S2-2"],
        target_addresses=["12 Elsewhere Rd", "99 Nowhere Ave"],
    )
    assert list(df.target_entity_id) == ["S2-1"]

    postings = {"12": ["S2-1", "S2-2", "S2-3"], "99": ["S2-4"]}
    sizes = numeric_collision_group_sizes(postings)
    assert sizes.iloc[0] == 3
    assert sizes.index[0] == "12"


def test_numeric_signature_order_and_duplicate_insensitive():
    # "12 MG Road Building 4" vs "Building 4, 12 M.G. Rd" -- same numeric
    # tokens {12, 4} in different order/punctuation -- must produce the
    # identical signature.
    sig_a = numeric_signature("12 MG Road Building 4")
    sig_b = numeric_signature("Building 4, 12 M.G. Rd")
    assert sig_a == sig_b == ("12", "4")
    assert numeric_signature("no numbers here") is None
    assert numeric_signature("") is None
    # duplicate numeric tokens collapse in the signature
    assert numeric_signature("4 4 4 Main St") == ("4",)


def test_exact_numeric_signature_candidates_route():
    df = exact_numeric_signature_candidates(
        s1_ids=["S1-1", "S1-2"],
        s1_addresses=["12 MG Road Building 4", "5 Elm St"],
        target_ids=["S2-1", "S2-2", "S2-3"],
        target_addresses=[
            "Building 4, 12 M.G. Rd",  # same signature as S1-1 -> should match
            "12 MG Road",  # signature ("12",) only -> does not match S1-1's ("12","4")
            "5 Elm St",  # exact match for S1-2
        ],
    )
    matched_for_s1_1 = set(df[df.s1_entity_id == "S1-1"].target_entity_id)
    assert matched_for_s1_1 == {"S2-1"}
    matched_for_s1_2 = set(df[df.s1_entity_id == "S1-2"].target_entity_id)
    assert matched_for_s1_2 == {"S2-3"}

    # max_group_size prunes an oversized signature group
    df_pruned = exact_numeric_signature_candidates(
        s1_ids=["S1-1"],
        s1_addresses=["12 MG Road Building 4"],
        target_ids=["S2-1", "S2-2"],
        target_addresses=["Building 4, 12 M.G. Rd", "12 4 Another Match"],
        max_group_size=1,
    )
    assert df_pruned.empty  # group size 2 > cap of 1 -> dropped entirely


# --- 8/9/10. union dedup, route provenance, score/rank retention -----------

def test_union_dedup_and_provenance():
    fwd = pd.DataFrame(
        {
            "s1_entity_id": ["S1-1", "S1-1"],
            "target_entity_id": ["S2-1", "S3-1"],
            "route_forward_word": [True, True],
            "forward_rank": [1, 2],
            "forward_score": [0.9, 0.5],
        }
    )
    rev = pd.DataFrame(
        {
            "s1_entity_id": ["S1-1"],
            "target_entity_id": ["S2-1"],
            "route_reverse_word": [True],
            "reverse_rank": [1],
            "reverse_score": [0.99],
        }
    )
    union = union_candidates([fwd, rev])
    assert len(union) == 2  # S2-1 appears once, not twice, despite two routes
    row = union[union.target_entity_id == "S2-1"].iloc[0]
    assert row["route_forward_word"] == True
    assert row["route_reverse_word"] == True
    assert row["forward_score"] == 0.9
    assert row["reverse_score"] == 0.99
    assert row["target_source"] == "S2"

    row2 = union[union.target_entity_id == "S3-1"].iloc[0]
    assert row2["route_reverse_word"] == False  # never touched by reverse route


def test_union_candidates_partitioned_matches_simple(tmp_path):
    # A richer synthetic scenario: 3 routes, 5 S1 entities, several targets
    # shared across routes with differing rank/score provenance, so the
    # partitioned (bucketed, disk-backed) path and the simple in-memory
    # path have real aggregation work to agree on, not just a trivial case.
    fwd = pd.DataFrame(
        {
            "s1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3", "S1-4", "S1-5"],
            "target_entity_id": ["S2-1", "S3-1", "S2-2", "S3-2", "S2-3", "S3-3"],
            "route_forward_word": [True] * 6,
            "forward_rank": [1, 2, 1, 1, 3, 1],
            "forward_score": [0.9, 0.5, 0.8, 0.7, 0.3, 0.6],
        }
    )
    rev = pd.DataFrame(
        {
            "s1_entity_id": ["S1-1", "S1-3", "S1-4"],
            "target_entity_id": ["S2-1", "S3-2", "S2-9"],
            "route_reverse_word": [True, True, True],
            "reverse_rank": [1, 2, 1],
            "reverse_score": [0.99, 0.4, 0.55],
        }
    )
    numeric = pd.DataFrame(
        {
            "s1_entity_id": ["S1-2", "S1-4", "S1-5"],
            "target_entity_id": ["S2-2", "S2-3", "S3-9"],
            "route_numeric_exact_set": [True, True, True],
        }
    )

    simple = union_candidates([fwd, rev, numeric])
    simple_sorted = simple.sort_values(["s1_entity_id", "target_source", "target_entity_id"]).reset_index(drop=True)

    paths = []
    for name, df in [("fwd", fwd), ("rev", rev), ("numeric", numeric)]:
        p = tmp_path / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)

    output_path = tmp_path / "union_output.parquet"
    diag = union_candidates_partitioned(paths, output_path, n_buckets=4, tmp_dir=tmp_path / "parts")
    assert diag["total_input_rows"] == len(fwd) + len(rev) + len(numeric)

    partitioned = pd.read_parquet(output_path)
    partitioned_sorted = partitioned.sort_values(
        ["s1_entity_id", "target_source", "target_entity_id"]
    ).reset_index(drop=True)

    # Column order can differ (bucket processing order) -- compare as sets
    # of columns and row-for-row values, not literal frame equality.
    assert set(simple_sorted.columns) == set(partitioned_sorted.columns)
    common_cols = sorted(simple_sorted.columns)
    pd.testing.assert_frame_equal(
        simple_sorted[common_cols], partitioned_sorted[common_cols], check_dtype=False
    )


# --- 11-18. blocking metrics -------------------------------------------------

def test_blocking_metrics_hand_computed():
    all_s1 = ["S1-A", "S1-B", "S1-C", "S1-D"]
    # A: non-singleton, fully covered. B: non-singleton, half covered.
    # C: true singleton, exposed (false candidate). D: true singleton, clean.
    truth = {
        "S1-A": frozenset({"S2-1", "S2-2"}),
        "S1-B": frozenset({"S2-3", "S2-4"}),
    }
    candidates = {
        "S1-A": frozenset({"S2-1", "S2-2", "S2-9"}),
        "S1-B": frozenset({"S2-3"}),
        "S1-C": frozenset({"S2-8"}),
    }

    m = compute_blocking_metrics(
        all_s1_ids=all_s1,
        truth_by_s1=truth,
        candidates_by_s1=candidates,
        n_s2_total=10,
        n_s3_total=10,
    )

    # link recall = (2 + 1) / (2 + 2) = 0.75
    assert np.isclose(m.link_recall, 0.75)
    # macro candidate recall over {A,B} = mean(1.0, 0.5) = 0.75
    assert np.isclose(m.macro_candidate_recall_non_singleton, 0.75)
    # complete coverage: only A is fully covered -> 1/2 = 0.5
    assert np.isclose(m.complete_true_link_coverage_non_singleton, 0.5)
    # singleton exposure: C exposed, D not -> 1/2 = 0.5
    assert np.isclose(m.singleton_candidate_exposure_rate, 0.5)
    assert m.singleton_candidate_load["max"] == 1
    assert m.n_true_singletons == 2
    assert m.n_true_non_singletons == 2
    assert m.total_candidate_edges == 3 + 1 + 1 + 0
    # global reduction ratio: unrestricted = 4 * (10+10) = 80; edges = 5
    assert np.isclose(m.global_reduction_ratio, 1 - 5 / 80)


def test_route_incremental_value():
    all_s1 = ["S1-A"]
    truth = {"S1-A": frozenset({"S2-1", "S2-2"})}
    baseline = {"S1-A": frozenset({"S2-1"})}
    union = {"S1-A": frozenset({"S2-1", "S2-2"})}
    delta = route_incremental_value(all_s1, truth, baseline, union)
    assert delta["new_true_links_rescued"] == 1
    assert delta["newly_complete_s1_entities"] == 1


def test_build_id_sets_by_s1():
    df = pd.DataFrame(
        {"s1_entity_id": ["S1-1", "S1-1", "S1-2"], "target_entity_id": ["S2-1", "S2-1", "S3-9"]}
    )
    sets = build_id_sets_by_s1(df)
    assert sets["S1-1"] == frozenset({"S2-1"})
    assert sets["S1-2"] == frozenset({"S3-9"})


# --- 21. candidate serialization/deserialization ----------------------------

def test_candidate_and_manifest_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "s1_entity_id": ["S1-1"],
            "target_source": ["S2"],
            "target_entity_id": ["S2-1"],
            "route_forward_word": [True],
        }
    )
    out = tmp_path / "candidates.parquet"
    write_candidates(df, out)
    back = read_candidates(out)
    pd.testing.assert_frame_equal(df, back)

    manifest_path = tmp_path / "manifest.json"
    manifest = {"experiment": "TEST", "config_hash": "abc123", "n_rows": 1}
    write_manifest(manifest, manifest_path)
    back_manifest = read_manifest(manifest_path)
    assert back_manifest == manifest


# --- TF-IDF fitting provenance ----------------------------------------------

def test_fit_vectorizer_provenance():
    config = VectorizerConfig(representation="word", min_df=1, max_df=1.0, fit_id="unit-test")
    fitted = fit_vectorizer(["acme bakery", "acme grocery", "totally unrelated shop"], config)
    prov = fitted.provenance()
    assert prov["representation"] == "word"
    assert prov["n_documents_fit"] == 3
    assert prov["vocabulary_size"] > 0
    assert len(prov["config_hash"]) == 16


# --- scalable metrics == simple metrics, on synthetic data ------------------
# Proves the join/groupby-based scalable path (used once candidate tables
# are too large to build per-entity Python sets from, e.g. EXP-B001R's
# r-grid union) gives EXACTLY the same answer as the original
# build_id_sets_by_s1 + compute_blocking_metrics path, for both a single
# candidate table and a two-table union.

def _synthetic_blocking_scenario():
    all_s1 = ["S1-A", "S1-B", "S1-C", "S1-D", "S1-E"]
    truth = {
        "S1-A": frozenset({"S2-1", "S2-2", "S3-9"}),
        "S1-B": frozenset({"S2-3"}),
        "S1-D": frozenset({"S3-7", "S3-8"}),
        # S1-C, S1-E are true singletons (absent from truth => empty).
    }
    # Route A ("forward-like"): decent recall, some noise, exposes a singleton.
    df_a = pd.DataFrame(
        {
            "s1_entity_id": ["S1-A", "S1-A", "S1-B", "S1-D", "S1-C", "S1-E"],
            "target_entity_id": ["S2-1", "S2-9", "S2-3", "S3-7", "S2-8", "S3-1"],
        }
    )
    # Route B ("reverse-like"): rescues S1-A's S3-9 and S1-D's S3-8, overlaps
    # with route A on S2-1 (S1-A) and S3-7 (S1-D), also exposes S1-E again
    # via a different id and S1-C not at all.
    df_b = pd.DataFrame(
        {
            "s1_entity_id": ["S1-A", "S1-A", "S1-D", "S1-D", "S1-E"],
            "target_entity_id": ["S2-1", "S3-9", "S3-7", "S3-8", "S3-2"],
        }
    )
    return all_s1, truth, df_a, df_b


def test_scalable_metrics_match_simple_single_table():
    all_s1, truth, df_a, _ = _synthetic_blocking_scenario()

    cand_by_s1 = build_id_sets_by_s1(df_a)
    simple = compute_blocking_metrics(all_s1, truth, cand_by_s1, n_s2_total=9, n_s3_total=9)

    truth_edges = truth_edges_dataframe(truth)
    mask = link_found_mask(truth_edges, df_a)
    counts = candidate_count_per_s1(df_a, all_s1)
    scalable = compute_blocking_metrics_scalable(
        all_s1, truth_edges, mask, counts, n_s2_total=9, n_s3_total=9
    )

    assert simple.as_dict() == scalable.as_dict()


def test_scalable_metrics_match_simple_two_table_union():
    all_s1, truth, df_a, df_b = _synthetic_blocking_scenario()

    cand_a = build_id_sets_by_s1(df_a)
    cand_b = build_id_sets_by_s1(df_b)
    union_by_s1 = {
        s1: cand_a.get(s1, frozenset()) | cand_b.get(s1, frozenset()) for s1 in all_s1
    }
    simple_union = compute_blocking_metrics(all_s1, truth, union_by_s1, n_s2_total=9, n_s3_total=9)

    truth_edges = truth_edges_dataframe(truth)
    mask_a = link_found_mask(truth_edges, df_a)
    mask_b = link_found_mask(truth_edges, df_b)
    union_mask = mask_a | mask_b
    union_counts = union_count_per_s1(df_a, df_b, all_s1)
    scalable_union = compute_blocking_metrics_scalable(
        all_s1, truth_edges, union_mask, union_counts, n_s2_total=9, n_s3_total=9
    )

    assert simple_union.as_dict() == scalable_union.as_dict()

    # Intersection sanity: S1-A shares S2-1, S1-D shares S3-7 -> 2 total.
    inter = intersection_count_per_s1(df_a, df_b, all_s1)
    assert inter.to_dict() == {"S1-A": 1, "S1-B": 0, "S1-C": 0, "S1-D": 1, "S1-E": 0}


def test_route_incremental_value_scalable_matches_simple():
    all_s1, truth, df_a, df_b = _synthetic_blocking_scenario()

    cand_a = build_id_sets_by_s1(df_a)
    cand_b = build_id_sets_by_s1(df_b)
    union_by_s1 = {
        s1: cand_a.get(s1, frozenset()) | cand_b.get(s1, frozenset()) for s1 in all_s1
    }
    simple_delta = route_incremental_value(all_s1, truth, cand_a, union_by_s1)

    truth_edges = truth_edges_dataframe(truth)
    mask_a = link_found_mask(truth_edges, df_a)
    mask_b = link_found_mask(truth_edges, df_b)
    scalable_delta = route_incremental_value_scalable(truth_edges, mask_a, mask_a | mask_b)

    assert simple_delta == scalable_delta
    # S1-A's S3-9 and S1-D's S3-8 are rescued only by route B -> 2 new links.
    assert scalable_delta["new_true_links_rescued"] == 2
    # S1-D becomes fully covered only once route B adds S3-8 -> 1 newly complete.
    assert scalable_delta["newly_complete_s1_entities"] == 1
