"""Stage-2 target-target corroboration features (EXP-S2C).

For each candidate row (S1 s, target t) the "corroborator" is the same S1's highest-probability OTHER candidate
target (overall, and separately the best candidate from the other target source). The features compare the row's
target record with its corroborator record (v2-normalized strings): name token-set ratio and address ratio, plus the
corroborator's stage-1 probability. Rationale: several independent listings of one business are corrupted copies of
the same entity; a confidently matched sibling listing carries evidence (address, name variant) that an ambiguous or
address-less target lacks, and an unrelated (orphan) target tends to disagree with the S1's confident listings.
Inputs are stage-1 probabilities over the table being resolved and the supplied records (no labels); identical code
for train (cross-fitted probabilities) and test (main-model probabilities).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

STAGE2C_FEATURES = ["c_p", "c_name_sim", "c_addr_sim", "c_p_oth", "c_name_sim_oth", "c_addr_sim_oth"]


def _best_two(key: np.ndarray, p: np.ndarray):
    """Sorted order by (key, -p); per sorted position the row index of its group's best and second-best rows (-1)."""
    order = np.lexsort((-p, key))
    k = key[order]
    first = np.ones(len(k), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.append(starts, len(k)))
    gs = np.repeat(starts, sizes)
    best = order[gs]
    has2 = np.repeat(sizes >= 2, sizes)
    second = np.where(has2, order[np.minimum(gs + 1, len(k) - 1)], -1)
    return order, k, starts, best, second


def _other_best_same_group(key: np.ndarray, p: np.ndarray) -> np.ndarray:
    order, _, _, best, second = _best_two(key, p)
    other = np.where(best == order, second, best)
    out = np.empty(len(p), dtype=np.int64)
    out[order] = other
    return out


def _best_of_group(key: np.ndarray, p: np.ndarray, query: np.ndarray) -> np.ndarray:
    order, k, starts, best, _ = _best_two(key, p)
    gk, gbest = k[starts], best[starts]
    pos = np.minimum(np.searchsorted(gk, query), len(gk) - 1)
    return np.where(gk[pos] == query, gbest[pos], -1)


def _sims(rows: np.ndarray, other: np.ndarray, t: np.ndarray, t_name: np.ndarray, t_addr: np.ndarray):
    n = len(rows)
    name_sim = np.full(n, -1.0, dtype=np.float32)
    addr_sim = np.full(n, -1.0, dtype=np.float32)
    ok = other >= 0
    if ok.any():
        a, b = t[rows[ok]], t[other[ok]]
        name_sim[ok] = process.cpdist(t_name[a].tolist(), t_name[b].tolist(), scorer=fuzz.token_set_ratio,
                                      workers=-1).astype(np.float32) / 100.0
        aa, ab = t_addr[a], t_addr[b]
        both = (aa != "") & (ab != "")
        tmp = np.full(len(a), -1.0, dtype=np.float32)
        if both.any():
            tmp[both] = process.cpdist(aa[both].tolist(), ab[both].tolist(), scorer=fuzz.ratio,
                                       workers=-1).astype(np.float32) / 100.0
        addr_sim[ok] = tmp
    return name_sim, addr_sim


def corroboration_features(s1: np.ndarray, t: np.ndarray, p: np.ndarray, t_src: np.ndarray, t_name: np.ndarray,
                           t_addr: np.ndarray, keep: np.ndarray) -> pd.DataFrame:
    """s1, t, p, t_src: per-row arrays over the full table; t_name / t_addr: per-target normalized strings (object
    arrays indexed by target code); keep: rows for which features are computed (others get the defaults)."""
    p = p.astype(np.float32)
    s1 = s1.astype(np.int64)
    src = t_src.astype(np.int64)
    n = len(p)
    out = {c: np.full(n, -1.0, dtype=np.float32) for c in STAGE2C_FEATURES}
    out["c_p"][:] = 0.0
    out["c_p_oth"][:] = 0.0
    other = _other_best_same_group(s1, p)
    other_oth = _best_of_group(s1 * 2 + src, p, s1 * 2 + (1 - src))
    rows = np.flatnonzero(keep)
    for tag, oth in (("", other), ("_oth", other_oth)):
        o = oth[rows]
        out["c_p" + tag][rows] = np.where(o >= 0, p[np.maximum(o, 0)], 0.0)
        ns, as_ = _sims(rows, o, t, t_name, t_addr)
        out["c_name_sim" + tag][rows] = ns
        out["c_addr_sim" + tag][rows] = as_
    return pd.DataFrame(out)


def target_strings(ids, table: pd.DataFrame):
    """v2-normalized (name, address) object arrays aligned to target codes `ids` (entity_id order)."""
    from normalization_v2 import normalize_address_v2, normalize_name_v2
    tb = table.set_index("entity_id")
    names = pd.Series(ids).map(tb["business_name"]).fillna("").map(normalize_name_v2).to_numpy(dtype=object)
    addrs = pd.Series(ids).map(tb["business_address"]).fillna("").map(normalize_address_v2).to_numpy(dtype=object)
    return names, addrs


STAGE2R_FEATURES = ["r_c_p", "r_c_name_sim", "r_c_addr_sim", "c_anchor_t_share", "c_anchor_t_margin"]


def rival_anchor_features(s1: np.ndarray, t: np.ndarray, p: np.ndarray, corr: pd.DataFrame) -> pd.DataFrame:
    """(a) rival corroboration: the S2C evidence of the target's best competing owner row;
    (b) anchor reliability: ownership share / margin of the row's same-S1 anchor target."""
    from stage2 import _top2
    p = p.astype(np.float32)
    s1 = s1.astype(np.int64)
    t = t.astype(np.int64)
    rival = _other_best_same_group(t, p)
    ok = rival >= 0
    out = {}
    for col, name, default in (("c_p", "r_c_p", 0.0), ("c_name_sim", "r_c_name_sim", -1.0),
                               ("c_addr_sim", "r_c_addr_sim", -1.0)):
        x = corr[col].to_numpy(dtype=np.float32)
        v = np.full(len(p), default, dtype=np.float32)
        v[ok] = x[rival[ok]]
        out[name] = v
    anchor = _other_best_same_group(s1, p)
    tk, tbest, tsec, trank = _top2(t, p)
    pos = np.searchsorted(tk, t)
    t_other = np.where(trank == 1, tsec[pos], tbest[pos]).astype(np.float32)
    t_sum = np.bincount(t, weights=p, minlength=int(t.max()) + 1)[t]
    share = (p / np.maximum(t_sum, 1e-6)).astype(np.float32)
    margin = (p - t_other).astype(np.float32)
    oka = anchor >= 0
    a_share = np.zeros(len(p), dtype=np.float32)
    a_margin = np.full(len(p), -1.0, dtype=np.float32)
    a_share[oka] = share[anchor[oka]]
    a_margin[oka] = margin[anchor[oka]]
    out["c_anchor_t_share"] = a_share
    out["c_anchor_t_margin"] = a_margin
    return pd.DataFrame(out)
