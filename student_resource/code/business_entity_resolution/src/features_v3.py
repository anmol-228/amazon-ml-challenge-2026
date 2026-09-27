"""Vectorized pair features for the R1-candidate matcher (M003).

All record-level preprocessing (normalization, token hashing, document
frequencies) is computed over the input record set being resolved (train
records for development, test records for test), with no labels, using the
same code path for both. Pair features are computed in batches with RapidFuzz
cpdist (multi-threaded C++) and sparse row-wise products, so no per-pair Python
loop is needed.

Competition features (per target: best/second-best owner score, margin; per
S1: rank among its candidates) are computed from the candidate table itself.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from scipy import sparse

from blocking_normalization import normalize_field

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_DIGITS_RE = re.compile(r"^\d+$")
_NONALNUM_RE = re.compile(r"[\W_]+", re.UNICODE)
N_HASH = 2**24


def _num(t: str) -> str:
    s = t.lstrip("0")
    return s if s else "0"


def _tokens(name: str, addr: str):
    ntoks = []
    for t in _TOKEN_RE.findall(name):
        if _DIGITS_RE.match(t):
            t = _num(t)
        if t not in ntoks:
            ntoks.append(t)
    alpha, nums = [], []
    for t in _TOKEN_RE.findall(addr):
        if _DIGITS_RE.match(t):
            t = _num(t)
            if t not in nums:
                nums.append(t)
        elif t not in alpha:
            alpha.append(t)
    return ntoks, alpha, nums


class RecordTable:
    """Normalized strings + hashed token matrices for a set of records."""

    def __init__(self, ids, names, addrs, norm: str = "v1"):
        from sklearn.feature_extraction import FeatureHasher

        if norm == "v2":
            from normalization_v2 import normalize_address_v2 as fa, normalize_name_v2 as fn
        else:
            fn = fa = normalize_field
        self.ids = np.asarray(ids, dtype=object)
        self.name = np.array([fn(x) for x in names], dtype=object)
        self.addr = np.array([fa(x) for x in addrs], dtype=object)
        self.name_compact = np.array([_NONALNUM_RE.sub("", x) for x in self.name], dtype=object)
        toks = [_tokens(n, a) for n, a in zip(self.name, self.addr)]
        fh = FeatureHasher(n_features=N_HASH, input_type="string", alternate_sign=False, dtype=np.float32)

        def mat(lists):
            m = fh.transform(lists).tocsr()
            m.data[:] = 1.0
            return m

        self.ntok = mat([t[0] for t in toks])
        self.atok = mat([t[1] for t in toks])
        self.num = mat([t[2] for t in toks])
        self.first_num = np.array([t[2][0] if t[2] else "" for t in toks], dtype=object)
        self.n_ntok = np.diff(self.ntok.indptr).astype(np.float32)
        self.n_atok = np.diff(self.atok.indptr).astype(np.float32)
        self.n_num = np.diff(self.num.indptr).astype(np.float32)
        self.name_len = np.array([len(x) for x in self.name], dtype=np.float32)
        self.addr_len = np.array([len(x) for x in self.addr], dtype=np.float32)
        self.name_ascii = np.array([sum(c.isascii() for c in x) / len(x) if x else 1.0 for x in self.name],
                                   dtype=np.float32)


def corpus_idf(tables: list[RecordTable], attr: str) -> np.ndarray:
    """Per-hash-bucket IDF over the given record tables (the input corpus)."""
    df = np.zeros(N_HASH, dtype=np.int64)
    n = 0
    for t in tables:
        m = getattr(t, attr)
        df += np.bincount(m.indices, minlength=N_HASH)
        n += m.shape[0]
    return (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)


IDF_REF_N = 200_000


def corpus_idf_rows(parts: list[tuple[RecordTable, np.ndarray]], attr: str, ref_n: float = IDF_REF_N) -> np.ndarray:
    """Scale-invariant IDF over a row subset of each table (one country partition): n and df are both
    counted inside the subset and the ratio is capped at ref_n, so a token's IDF depends only on its
    relative frequency (rare tokens share the cap) and not on the size of the corpus being resolved.
    ref_n must not exceed the smallest partition size."""
    df = np.zeros(N_HASH, dtype=np.int64)
    n = 0
    for t, rows in parts:
        m = getattr(t, attr)[rows]
        df += np.bincount(m.indices, minlength=N_HASH)
        n += m.shape[0]
    assert n >= ref_n, (n, ref_n)
    return (np.log(np.minimum((n + 1) / (df + 1), ref_n)) + 1.0).astype(np.float32)


def _rowdot(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)


def _weighted(m: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    m = m.tocsr().copy()
    m.data = (m.data * idf[m.indices]).astype(np.float32)
    return m


def _wsum(m: sparse.csr_matrix, idf: np.ndarray) -> np.ndarray:
    return np.asarray(_weighted(m, idf).sum(axis=1)).ravel().astype(np.float32)


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def pair_features(S: RecordTable, T: RecordTable, si: np.ndarray, ti: np.ndarray,
                  idf_n: np.ndarray, idf_a: np.ndarray, workers: int = -1) -> dict[str, np.ndarray]:
    sn, tn = S.name[si], T.name[ti]
    sa, ta = S.addr[si], T.addr[ti]
    f: dict[str, np.ndarray] = {}
    f["name_ratio"] = _cp(sn, tn, fuzz.ratio, workers) / 100
    f["name_tsort"] = _cp(sn, tn, fuzz.token_sort_ratio, workers) / 100
    f["name_tset"] = _cp(sn, tn, fuzz.token_set_ratio, workers) / 100
    f["name_partial"] = _cp(sn, tn, fuzz.partial_ratio, workers) / 100
    f["name_jw"] = _cp(sn, tn, JaroWinkler.normalized_similarity, workers)
    f["name_compact_ratio"] = _cp(S.name_compact[si], T.name_compact[ti], fuzz.ratio, workers) / 100
    f["addr_ratio"] = _cp(sa, ta, fuzz.ratio, workers) / 100
    f["addr_tset"] = _cp(sa, ta, fuzz.token_set_ratio, workers) / 100
    f["addr_partial"] = _cp(sa, ta, fuzz.partial_ratio, workers) / 100

    Ns, Nt = S.ntok[si], T.ntok[ti]
    As, At = S.atok[si], T.atok[ti]
    Xs, Xt = S.num[si], T.num[ti]
    n_sh = _rowdot(Ns, Nt)
    a_sh = _rowdot(As, At)
    x_sh = _rowdot(Xs, Xt)
    ns_c, nt_c = S.n_ntok[si], T.n_ntok[ti]
    as_c, at_c = S.n_atok[si], T.n_atok[ti]
    xs_c, xt_c = S.n_num[si], T.n_num[ti]
    f["name_tok_shared"] = n_sh
    f["name_tok_jacc"] = n_sh / np.maximum(ns_c + nt_c - n_sh, 1)
    f["addr_tok_shared"] = a_sh
    f["addr_tok_jacc"] = a_sh / np.maximum(as_c + at_c - a_sh, 1)
    f["num_shared"] = x_sh
    f["num_jacc"] = x_sh / np.maximum(xs_c + xt_c - x_sh, 1)
    f["num_conflict"] = ((xs_c > 0) & (xt_c > 0) & (x_sh == 0)).astype(np.float32)
    f["num_s_only"] = xs_c - x_sh
    f["num_t_only"] = xt_c - x_sh
    fs, ft = S.first_num[si], T.first_num[ti]
    f["first_num_eq"] = np.where((fs == "") | (ft == ""), np.float32(-1), (fs == ft).astype(np.float32)).astype(np.float32)

    # IDF-weighted overlaps and unmatched mass (contradiction evidence).
    for tag, A, B, idf in (("name", Ns, Nt, idf_n), ("addr", As, At, idf_a)):
        wa = _wsum(A, idf)
        wb = _wsum(B, idf)
        shw = _weighted(A.multiply(B).tocsr(), idf)
        wsh = np.asarray(shw.sum(axis=1)).ravel()
        mx = np.zeros(shw.shape[0], dtype=np.float32)
        nz = np.diff(shw.indptr) > 0
        if nz.any():
            mx[nz] = np.maximum.reduceat(shw.data, shw.indptr[:-1][nz]).astype(np.float32)
        f[f"{tag}_idf_jacc"] = (wsh / np.maximum(wa + wb - wsh, 1e-6)).astype(np.float32)
        f[f"{tag}_idf_shared_max"] = mx
        f[f"{tag}_idf_s_only"] = (wa - wsh).astype(np.float32)
        f[f"{tag}_idf_t_only"] = (wb - wsh).astype(np.float32)

    f["name_exact"] = (sn == tn).astype(np.float32)
    f["name_compact_eq"] = (S.name_compact[si] == T.name_compact[ti]).astype(np.float32)
    f["name_len_s"], f["name_len_t"] = S.name_len[si], T.name_len[ti]
    f["addr_len_s"], f["addr_len_t"] = S.addr_len[si], T.addr_len[ti]
    f["t_addr_missing"] = (T.addr_len[ti] == 0).astype(np.float32)
    f["t_name_ascii"] = T.name_ascii[ti]
    f["s_name_ascii"] = S.name_ascii[si]
    return f


def competition_features(cand: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Target-side and S1-side competition features from a candidate table
    containing s1_entity_id, target_entity_id and a retrieval score column."""
    c = cand[["s1_entity_id", "target_entity_id", score_col]].copy()
    c[score_col] = c[score_col].fillna(0).astype(np.float32)
    g = c.groupby("target_entity_id", sort=False)[score_col]
    best = g.transform("max")
    n_t = g.transform("size")
    rank_t = g.rank(ascending=False, method="min")
    # second best per target: max of scores excluding one instance of the best
    srt = c.sort_values(["target_entity_id", score_col], ascending=[True, False])
    second = srt.groupby("target_entity_id", sort=False)[score_col].nth(1)
    sec_map = pd.Series(second.values, index=srt.loc[second.index, "target_entity_id"].values)
    sec = c["target_entity_id"].map(sec_map).fillna(0).astype(np.float32)
    gs = c.groupby("s1_entity_id", sort=False)[score_col]
    out = pd.DataFrame({
        "t_best_score": best.astype(np.float32),
        "t_n_owners": n_t.astype(np.float32),
        "t_rank": rank_t.astype(np.float32),
        "t_margin_to_best": (c[score_col] - best).astype(np.float32),
        "t_margin_to_second": np.where(rank_t == 1, c[score_col] - sec, c[score_col] - best).astype(np.float32),
        "s_best_score": gs.transform("max").astype(np.float32),
        "s_n_cand": gs.transform("size").astype(np.float32),
        "s_rank": gs.rank(ascending=False, method="min").astype(np.float32),
    }, index=cand.index)
    out["s_margin_to_best"] = (c[score_col] - out["s_best_score"]).astype(np.float32)
    return out
