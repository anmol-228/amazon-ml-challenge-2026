"""Stage-2 (collective) features from stage-1 match probabilities.

Input: arrays over ONE candidate table covering every S1 of the corpus being
resolved (train corpus with out-of-fold probabilities for training S1; test
corpus at inference): s1 code, target code, stage-1 probability p.
Output: per-row features describing how this (S1, target) pair competes with
the target's other possible owners and with the S1's other candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STAGE2_FEATURES = [
    "p1", "t_p_other_max", "t_margin", "t_n_conf", "t_p_sum", "t_p_rank",
    "s_p_sum", "s_n_conf", "s_p_max", "s_p_rank", "s_p_gap_next",
]


def _top2(key: np.ndarray, p: np.ndarray):
    """Per-group best and second-best of p for integer group keys."""
    order = np.lexsort((-p, key))
    k, v = key[order], p[order]
    first = np.ones(len(k), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    starts = np.flatnonzero(first)
    best = v[starts]
    has2 = np.zeros(len(starts), dtype=bool)
    nxt = starts + 1
    has2[:-1] = nxt[:-1] < starts[1:]
    has2[-1] = nxt[-1] < len(k)
    second = np.where(has2, v[np.minimum(nxt, len(v) - 1)], 0.0)
    rank = np.empty(len(k), dtype=np.int64)
    rank[order] = np.arange(len(k)) - np.repeat(starts, np.diff(np.append(starts, len(k))))
    return k[starts], best, second, rank + 1


def stage2_features(s1: np.ndarray, t: np.ndarray, p: np.ndarray, conf: float = 0.5) -> pd.DataFrame:
    p = p.astype(np.float32)
    out = {"p1": p}
    # target side
    tk, tbest, tsec, trank = _top2(t.astype(np.int64), p)
    pos = np.searchsorted(tk, t)
    best, sec = tbest[pos], tsec[pos]
    other_max = np.where(trank == 1, sec, best).astype(np.float32)
    out["t_p_other_max"] = other_max
    out["t_margin"] = (p - other_max).astype(np.float32)
    out["t_n_conf"] = np.bincount(t[p >= conf], minlength=int(t.max()) + 1)[t].astype(np.float32)
    out["t_p_sum"] = np.bincount(t, weights=p, minlength=int(t.max()) + 1)[t].astype(np.float32)
    out["t_p_rank"] = trank.astype(np.float32)
    # S1 side
    sk, sbest, ssec, srank = _top2(s1.astype(np.int64), p)
    spos = np.searchsorted(sk, s1)
    out["s_p_sum"] = np.bincount(s1, weights=p, minlength=int(s1.max()) + 1)[s1].astype(np.float32)
    out["s_n_conf"] = np.bincount(s1[p >= conf], minlength=int(s1.max()) + 1)[s1].astype(np.float32)
    out["s_p_max"] = sbest[spos].astype(np.float32)
    out["s_p_rank"] = srank.astype(np.float32)
    # gap between this row's p and the next lower p within the S1 (0 if last)
    order = np.lexsort((-p, s1))
    ps, ss = p[order], s1[order]
    gap = np.zeros(len(p), dtype=np.float32)
    same = ss[1:] == ss[:-1]
    g = np.zeros(len(p), dtype=np.float32)
    g[:-1] = np.where(same, ps[:-1] - ps[1:], ps[:-1])
    g[-1] = ps[-1]
    gap[order] = g
    out["s_p_gap_next"] = gap
    return pd.DataFrame(out)
