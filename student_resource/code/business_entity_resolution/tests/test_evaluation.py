"""Unit tests for the official metric implementation (src/evaluation.py).

Zero tolerance for error: these tests cover the documented branch rules,
the Amazon worked example, and macro-averaging behavior. Run with:
    python -m pytest student_resource/code/business_entity_resolution/tests -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import evaluate, f_beta_from_counts, trivial_empty_baseline  # noqa: E402


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# 1. truth empty, prediction empty -> 1.0
def test_singleton_correct():
    result = evaluate({"S1-1": []}, {"S1-1": []})
    assert approx(result.macro_f_beta, 1.0)
    assert result.entity_scores[0].f_beta == 1.0


# 2. truth empty, prediction non-empty -> 0.0
def test_singleton_false_merge():
    result = evaluate({"S1-1": []}, {"S1-1": ["S2-1"]})
    assert approx(result.macro_f_beta, 0.0)


# 3. truth non-empty, prediction empty -> 0.0
def test_missed_entity_entirely():
    result = evaluate({"S1-1": ["S2-1"]}, {"S1-1": []})
    assert approx(result.macro_f_beta, 0.0)


# 4. single true match predicted exactly -> 1.0
def test_single_match_exact():
    result = evaluate({"S1-1": ["S2-1"]}, {"S1-1": ["S2-1"]})
    assert approx(result.macro_f_beta, 1.0)
    assert result.entity_scores[0].exact_match is True


# 5. multi-match truth predicted exactly -> 1.0
def test_multi_match_exact():
    truth = {"S1-1": ["S2-1", "S3-1", "S3-2"]}
    pred = {"S1-1": ["S3-2", "S2-1", "S3-1"]}  # order irrelevant
    result = evaluate(truth, pred)
    assert approx(result.macro_f_beta, 1.0)


# 6. false-positive case: one correct + one spurious
def test_false_positive_case():
    result = evaluate({"S1-1": ["S2-1"]}, {"S1-1": ["S2-1", "S2-2"]})
    # TP=1, P=2, T=1 -> precision=0.5, recall=1
    # F0.5 = (1.25*0.5*1)/(0.25*0.5+1) = 0.625/1.125
    expected = (1.25 * 0.5 * 1) / (0.25 * 0.5 + 1)
    assert approx(result.macro_f_beta, expected)


# 7. false-negative case: only recovers part of the truth
def test_false_negative_case():
    result = evaluate({"S1-1": ["S2-1", "S2-2"]}, {"S1-1": ["S2-1"]})
    # TP=1, P=1, T=2 -> precision=1, recall=0.5
    expected = (1.25 * 1 * 0.5) / (0.25 * 1 + 0.5)
    assert approx(result.macro_f_beta, expected)


# 8. mixed TP/FP/FN in one entity
def test_mixed_tp_fp_fn():
    result = evaluate({"S1-1": ["S2-1", "S2-2", "S3-1"]}, {"S1-1": ["S2-1", "S2-9"]})
    # T={S2-1,S2-2,S3-1}, P={S2-1,S2-9}, TP=1, precision=1/2, recall=1/3
    precision, recall = 0.5, 1 / 3
    expected = (1.25 * precision * recall) / (0.25 * precision + recall)
    assert approx(result.macro_f_beta, expected)


# 9. ordering irrelevant (ID lists given in different order score identically)
def test_ordering_irrelevant():
    r1 = evaluate({"S1-1": ["S2-1", "S3-1"]}, {"S1-1": ["S3-1", "S2-1"]})
    r2 = evaluate({"S1-1": ["S3-1", "S2-1"]}, {"S1-1": ["S2-1", "S3-1"]})
    assert approx(r1.macro_f_beta, r2.macro_f_beta)
    assert approx(r1.macro_f_beta, 1.0)


# 10. duplicate IDs handled per the documented evaluator contract (set dedup)
def test_duplicate_ids_deduplicated():
    result = evaluate({"S1-1": ["S2-1", "S2-1", "S3-1"]}, {"S1-1": ["S2-1"]})
    entity = result.entity_scores[0]
    assert entity.truth_size == 2  # {S2-1, S3-1}, duplicate collapsed
    assert entity.pred_size == 1


# 11. macro averaging across singleton-correct / singleton-wrong / exact / partial
def test_macro_averaging_across_mixed_entities():
    truth = {
        "S1-a": [],  # true singleton
        "S1-b": [],  # true singleton
        "S1-c": ["S2-1"],  # exact match
        "S1-d": ["S2-1", "S2-2"],  # partial match
    }
    pred = {
        "S1-a": [],  # correct empty -> 1.0
        "S1-b": ["S2-9"],  # false merge on singleton -> 0.0
        "S1-c": ["S2-1"],  # exact -> 1.0
        "S1-d": ["S2-1"],  # partial: TP=1,P=1,T=2 -> precision=1,recall=0.5
    }
    result = evaluate(truth, pred)
    partial_f = (1.25 * 1 * 0.5) / (0.25 * 1 + 0.5)
    expected_macro = (1.0 + 0.0 + 1.0 + partial_f) / 4
    assert approx(result.macro_f_beta, expected_macro)
    assert result.n_true_singletons == 2
    assert result.n_true_non_singletons == 2
    assert approx(result.singleton_accuracy, 0.5)  # one of two singletons correct
    assert approx(result.non_singleton_macro_f_beta, (1.0 + partial_f) / 2)


# 12. Amazon documented worked example
def test_amazon_worked_example():
    truth = {"S1-x": ["S2-00047", "S3-00812"]}
    pred = {"S1-x": ["S2-00047", "S2-00193", "S3-00812"]}
    result = evaluate(truth, pred)
    entity = result.entity_scores[0]
    assert entity.tp == 2
    assert approx(entity.precision, 2 / 3)
    assert approx(entity.recall, 1.0)
    assert approx(result.macro_f_beta, 0.714285714285714, tol=1e-9)


def test_f_beta_from_counts_matches_worked_example_directly():
    f = f_beta_from_counts(tp=2, pred_size=3, truth_size=2, beta=0.5)
    assert approx(f, 0.714285714285714, tol=1e-9)


def test_f_beta_zero_true_positive_but_nonempty_sets_scores_zero():
    # T and P both non-empty but disjoint -> TP=0 -> denom nonzero but numerator 0
    f = f_beta_from_counts(tp=0, pred_size=2, truth_size=2, beta=0.5)
    assert f == 0.0


def test_missing_prediction_key_treated_as_empty():
    # Contract: an S1 key present in truth but absent from pred is scored as
    # an empty prediction, not skipped.
    result = evaluate({"S1-1": ["S2-1"]}, {})
    assert result.n_entities == 1
    assert approx(result.macro_f_beta, 0.0)


def test_trivial_empty_baseline():
    truth = {
        "S1-a": [],
        "S1-b": [],
        "S1-c": ["S2-1"],
        "S1-d": ["S2-1", "S2-2"],
    }
    # 2 of 4 entities are true singletons -> baseline macro F0.5 = 0.5
    assert approx(trivial_empty_baseline(truth), 0.5)


def test_exact_set_accuracy():
    truth = {"S1-1": ["S2-1"], "S1-2": ["S2-2"]}
    pred = {"S1-1": ["S2-1"], "S1-2": ["S2-9"]}
    result = evaluate(truth, pred)
    assert approx(result.exact_set_accuracy, 0.5)
