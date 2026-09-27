"""Exact implementation of the official competition metric.

Official rule (per docs/OFFICIAL_REQUIREMENTS.md, SCORING section):

For each Source-1 entity, let T be the truth set of matched target IDs and
P be the predicted set of matched target IDs.

    T empty, P empty     -> entity score = 1.0   (singleton correctly left empty)
    T empty, P non-empty -> entity score = 0.0   (false merge on a true singleton)
    T non-empty, P empty -> entity score = 0.0   (missed every true match)
    otherwise:
        TP = |T ∩ P|
        precision = TP / |P|
        recall    = TP / |T|
        F0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
               (with F0.5 = 0.0 if that denominator is 0, i.e. TP == 0)

Macro score = mean(entity F0.5 over all Source-1 entities being evaluated).

Evaluator contract (documented, not implicit):
  - Truth/prediction ID lists are converted to sets before scoring, so
    duplicate IDs within one entity's list are deduplicated and ordering is
    irrelevant.
  - Every key in `truth` must be scored. A key missing from `pred` is treated
    as an empty prediction for that entity (contract choice; a real
    submission that omits a required row is a separate validator failure).
  - Keys present in `pred` but absent from `truth` are ignored by this
    function. In the real competition, `truth` covers every required
    Source-1 entity, so this only matters for ad hoc/partial evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

Prediction = Mapping[str, Iterable[str]]


def f_beta_from_counts(tp: int, pred_size: int, truth_size: int, beta: float = 0.5) -> float:
    """Entity-level F-beta score under the exact official branch rules."""
    if truth_size == 0 and pred_size == 0:
        return 1.0
    if truth_size == 0 and pred_size > 0:
        return 0.0
    if truth_size > 0 and pred_size == 0:
        return 0.0
    precision = tp / pred_size
    recall = tp / truth_size
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


@dataclass
class EntityScore:
    entity_id: str
    truth_size: int
    pred_size: int
    tp: int
    precision: float | None
    recall: float | None
    f_beta: float
    exact_match: bool


@dataclass
class EvaluationResult:
    macro_f_beta: float
    mean_precision: float | None
    mean_recall: float | None
    singleton_accuracy: float | None
    non_singleton_macro_f_beta: float | None
    exact_set_accuracy: float
    n_entities: int
    n_true_singletons: int
    n_true_non_singletons: int
    n_predicted_singleton: int
    n_predicted_non_singleton: int
    entity_scores: list[EntityScore] = field(default_factory=list)


def evaluate(
    truth: Prediction,
    pred: Prediction,
    beta: float = 0.5,
    keep_entity_scores: bool = True,
) -> EvaluationResult:
    """Compute the official macro F-beta score plus diagnostics.

    `truth` and `pred` map source1_entity_id -> iterable of matched target IDs.
    Only `truth`'s keys are scored (see module docstring contract).
    """
    f_scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    exact_matches = 0
    n_true_singletons = 0
    n_true_non_singletons = 0
    n_predicted_singleton = 0
    n_predicted_non_singleton = 0
    entity_scores: list[EntityScore] = []

    for entity_id, truth_ids in truth.items():
        t = frozenset(truth_ids)
        p = frozenset(pred.get(entity_id, ()))
        tp = len(t & p)
        truth_size = len(t)
        pred_size = len(p)

        f = f_beta_from_counts(tp, pred_size, truth_size, beta)
        f_scores.append(f)

        precision = (tp / pred_size) if pred_size > 0 else None
        recall = (tp / truth_size) if truth_size > 0 else None
        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)

        if truth_size == 0:
            n_true_singletons += 1
        else:
            n_true_non_singletons += 1
        if pred_size == 0:
            n_predicted_singleton += 1
        else:
            n_predicted_non_singleton += 1

        exact = t == p
        if exact:
            exact_matches += 1

        if keep_entity_scores:
            entity_scores.append(
                EntityScore(
                    entity_id=entity_id,
                    truth_size=truth_size,
                    pred_size=pred_size,
                    tp=tp,
                    precision=precision,
                    recall=recall,
                    f_beta=f,
                    exact_match=exact,
                )
            )

    n_entities = len(f_scores)
    macro_f_beta = sum(f_scores) / n_entities if n_entities else float("nan")
    mean_precision = sum(precisions) / len(precisions) if precisions else None
    mean_recall = sum(recalls) / len(recalls) if recalls else None

    singleton_scores = [
        f for f, t_size in zip(f_scores, _truth_sizes(truth)) if t_size == 0
    ]
    non_singleton_scores = [
        f for f, t_size in zip(f_scores, _truth_sizes(truth)) if t_size > 0
    ]
    singleton_accuracy = (
        sum(singleton_scores) / len(singleton_scores) if singleton_scores else None
    )
    non_singleton_macro_f_beta = (
        sum(non_singleton_scores) / len(non_singleton_scores)
        if non_singleton_scores
        else None
    )
    exact_set_accuracy = exact_matches / n_entities if n_entities else float("nan")

    return EvaluationResult(
        macro_f_beta=macro_f_beta,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        singleton_accuracy=singleton_accuracy,
        non_singleton_macro_f_beta=non_singleton_macro_f_beta,
        exact_set_accuracy=exact_set_accuracy,
        n_entities=n_entities,
        n_true_singletons=n_true_singletons,
        n_true_non_singletons=n_true_non_singletons,
        n_predicted_singleton=n_predicted_singleton,
        n_predicted_non_singleton=n_predicted_non_singleton,
        entity_scores=entity_scores,
    )


def _truth_sizes(truth: Prediction) -> Sequence[int]:
    return [len(frozenset(v)) for v in truth.values()]


def trivial_empty_baseline(truth: Prediction) -> float:
    """TRIVIAL TRAINING-ONLY REFERENCE.

    Macro F-beta of the degenerate "always predict empty" system. This is
    NOT a deployable test score and NOT a leaderboard claim - it exists only
    to quantify how much score is structurally attributable to singletons.
    Independent of beta: an empty prediction scores 1.0 against an empty
    truth set and 0.0 against any non-empty truth set, for every beta.
    """
    n = 0
    total = 0.0
    for truth_ids in truth.values():
        n += 1
        total += 1.0 if len(frozenset(truth_ids)) == 0 else 0.0
    return total / n if n else float("nan")
