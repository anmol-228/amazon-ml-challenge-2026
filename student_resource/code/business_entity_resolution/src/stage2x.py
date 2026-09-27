"""Stage-2 extra collective features (EXP-S2X), computed from the same inputs as src/stage2.py plus the
target source (S2 / S3) of every row.

- source-split S1 support: confident candidates of the same S1 from the SAME target source (excluding the
  row itself) and from the OTHER source; best competing probability in each.
- soft ownership shares: the row's probability as a share of the target's and of the S1's probability mass.

All values are functions of stage-1 probabilities over the table being resolved (no labels), identical in
train (cross-fitted probabilities) and test (main-model probabilities).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stage2 import _top2

STAGE2X_FEATURES = ["s_n_conf_same", "s_n_conf_oth", "s_pmax_same_x", "s_pmax_oth", "t_share", "s_share"]


def _group_best(key: np.ndarray, p: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Best p of group `query` (0 when the group is empty)."""
    k, best, _, _ = _top2(key, p)
    pos = np.searchsorted(k, query)
    pos_c = np.minimum(pos, len(k) - 1)
    return np.where(k[pos_c] == query, best[pos_c], 0.0).astype(np.float32)


def stage2x_features(s1: np.ndarray, t: np.ndarray, p: np.ndarray, t_src: np.ndarray,
                     conf: float = 0.5) -> pd.DataFrame:
    p = p.astype(np.float32)
    src = t_src.astype(np.int64)
    s1 = s1.astype(np.int64)
    key_same = s1 * 2 + src
    key_oth = s1 * 2 + (1 - src)
    is_conf = p >= conf
    cnt = np.bincount(key_same[is_conf], minlength=int(key_same.max()) + 2)
    out = {}
    out["s_n_conf_same"] = (cnt[key_same] - is_conf).astype(np.float32)
    out["s_n_conf_oth"] = cnt[key_oth].astype(np.float32)
    k, best, sec, rank = _top2(key_same, p)
    pos = np.searchsorted(k, key_same)
    out["s_pmax_same_x"] = np.where(rank == 1, sec[pos], best[pos]).astype(np.float32)
    out["s_pmax_oth"] = _group_best(key_same, p, key_oth)
    t_sum = np.bincount(t, weights=p, minlength=int(t.max()) + 1)[t]
    s_sum = np.bincount(s1, weights=p, minlength=int(s1.max()) + 1)[s1]
    out["t_share"] = (p / np.maximum(t_sum, 1e-6)).astype(np.float32)
    out["s_share"] = (p / np.maximum(s_sum, 1e-6)).astype(np.float32)
    return pd.DataFrame(out)


STAGE2T_FEATURES = ["t_twin_n", "t_twin_pmax_x", "t_twin_rank"]


def stage2_twin_features(s1: np.ndarray, t: np.ndarray, p: np.ndarray, s1_name_code: np.ndarray) -> pd.DataFrame:
    """Target-side exact-name twin competition: among the target's candidate S1s, those whose v2-normalized
    name equals this row's S1 name (s1_name_code indexed by s1). Count (excluding the row), best competing
    twin probability, and the row's probability rank inside its (target, S1-name) group."""
    p = p.astype(np.float32)
    nc = s1_name_code[s1].astype(np.int64)
    key = t.astype(np.int64) * (int(nc.max()) + 1) + nc
    k, best, sec, rank = _top2(key, p)
    pos = np.searchsorted(k, key)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return pd.DataFrame({"t_twin_n": (cnt[inv] - 1).astype(np.float32),
                         "t_twin_pmax_x": np.where(rank == 1, sec[pos], best[pos]).astype(np.float32),
                         "t_twin_rank": rank.astype(np.float32)})


OPENSET_T_FEATURES = ["t_owner_top_share", "t_owner_entropy", "t_owner_eff_n"]


def target_open_set_features(t: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Target-side stage-1 ownership concentration (EXP-S2O): top-owner share of the target's probability mass,
    entropy and effective number of owners of the within-target mass distribution. Diffuse ownership is the
    signature of a target whose true owner is absent (open-set / orphan distractor). Label-free."""
    t = t.astype(np.int64)
    p = np.maximum(p.astype(np.float64), 0.0)
    n_t = int(t.max()) + 1
    mass = np.bincount(t, weights=p, minlength=n_t)
    denom = np.maximum(mass[t], 1e-12)
    q = p / denom
    q2 = np.bincount(t, weights=q * q, minlength=n_t)
    ent = np.bincount(t, weights=-q * np.log(np.maximum(q, 1e-12)), minlength=n_t)
    best = np.zeros(n_t, dtype=np.float64)
    np.maximum.at(best, t, p)
    return pd.DataFrame({"t_owner_top_share": (best[t] / denom).astype(np.float32),
                         "t_owner_entropy": ent[t].astype(np.float32),
                         "t_owner_eff_n": (1.0 / np.maximum(q2[t], 1e-12)).astype(np.float32)})
