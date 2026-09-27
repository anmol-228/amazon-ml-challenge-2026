"""Vectorized macro F-beta under the official branch rules (evaluation.py
f_beta_from_counts), for fast threshold / decoder sweeps over candidate rows.

Entities are integer codes 0..n_entities-1 covering the FULL evaluation
population; truth_size[e] is the true match count of entity e. Candidate rows
carry (entity code, label, selected). Each (entity, target) pair must appear at
most once among rows.
"""

from __future__ import annotations

import numpy as np


def per_entity_f(ent: np.ndarray, label: np.ndarray, selected: np.ndarray, truth_size: np.ndarray,
                 beta: float = 0.5) -> np.ndarray:
    n = len(truth_size)
    sel = selected.astype(bool)
    pred = np.bincount(ent[sel], minlength=n).astype(np.float64)
    tp = np.bincount(ent[sel & label.astype(bool)], minlength=n).astype(np.float64)
    truth = truth_size.astype(np.float64)
    b2 = beta * beta
    f = np.zeros(n, dtype=np.float64)
    f[(truth == 0) & (pred == 0)] = 1.0
    both = (truth > 0) & (pred > 0)
    p = np.where(both, tp / np.maximum(pred, 1), 0.0)
    r = np.where(both, tp / np.maximum(truth, 1), 0.0)
    denom = b2 * p + r
    ok = both & (denom > 0)
    f[ok] = (1 + b2) * p[ok] * r[ok] / denom[ok]
    return f


def macro_f(ent, label, selected, truth_size, beta: float = 0.5) -> float:
    return float(per_entity_f(ent, label, selected, truth_size, beta).mean())


def target_exclusive(tgt: np.ndarray, prob: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Keep, for each target code, only the selected row with the highest prob
    (ties -> lowest row index). Rows not selected stay unselected."""
    idx = np.flatnonzero(selected)
    if len(idx) == 0:
        return selected.copy()
    order = idx[np.lexsort((idx, -prob[idx], tgt[idx]))]
    first = np.ones(len(order), dtype=bool)
    first[1:] = tgt[order][1:] != tgt[order][:-1]
    out = np.zeros_like(selected, dtype=bool)
    out[order[first]] = True
    return out
