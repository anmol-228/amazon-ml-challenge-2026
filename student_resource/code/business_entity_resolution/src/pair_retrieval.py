"""Pair-shingle sparse retrieval (candidate route R1).

Motivation (measured): the B004 word route used max_df=0.0008 to stay
tractable, which removes every token appearing in more than ~9.7k of ~12M
documents (city names, street types, legal suffixes, common name words). True
links whose distinctive evidence is made of individually common tokens
("alpha" + "infotech", "42" + "avenue" + "kolkata") are then unreachable.

This route keeps tractability differently: each record is represented by
*combination* keys (token pairs, number x token), which are rare even when the
individual tokens are common, plus single tokens. Keys are hashed into a 2^31
space, document frequency is computed per country partition over the input
records themselves (S1 + target records of that country; no labels), keys with
df above an absolute cap or present on only one side are dropped, and records
are TF-IDF weighted and L2 normalized over the remaining keys. Retrieval is
exact sparse cosine within the country partition: forward (S1 -> top-k
targets) and reverse (target -> top-k S1 owners).

Country is used as an open-set partition key (whatever country strings are
observed); there is no country-specific logic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from blocking_normalization import normalize_field

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_DIGITS_RE = re.compile(r"^\d+$")

N_FEATURES = 2**31 - 1


@dataclass(frozen=True)
class KeyConfig:
    max_name_tokens: int = 6
    max_addr_alpha: int = 6
    max_addr_nums: int = 2
    norm: str = "v1"


def _num(tok: str) -> str:
    s = tok.lstrip("0")
    return s if s else "0"


def record_keys(name: str, address: str, cfg: KeyConfig = KeyConfig()) -> list[str]:
    """Combination keys for one record. Inputs are raw field strings."""
    if cfg.norm == "v2":
        from normalization_v2 import normalize_address_v2, normalize_name_v2
        n = normalize_name_v2(name)
        a = normalize_address_v2(address)
    else:
        n = normalize_field(name)
        a = normalize_field(address)
    ntoks: list[str] = []
    for t in _TOKEN_RE.findall(n):
        if _DIGITS_RE.match(t):
            t = _num(t)
        elif len(t) < 2:
            continue
        if t not in ntoks:
            ntoks.append(t)
    atoks = _TOKEN_RE.findall(a)
    nums: list[str] = []
    alpha: list[str] = []
    for t in atoks:
        if _DIGITS_RE.match(t):
            t = _num(t)
            if t not in nums:
                nums.append(t)
        elif len(t) >= 2 and t not in alpha:
            alpha.append(t)
    nt = ntoks[: cfg.max_name_tokens]
    al = alpha[: cfg.max_addr_alpha]
    nu = nums[: cfg.max_addr_nums]

    keys = ["n:" + t for t in nt]
    for i in range(len(nt)):
        for j in range(i + 1, len(nt)):
            x, y = (nt[i], nt[j]) if nt[i] < nt[j] else (nt[j], nt[i])
            keys.append("nn:" + x + "|" + y)
    keys.extend("a:" + w for w in al)
    for i in range(len(al)):
        for j in range(i + 1, len(al)):
            x, y = (al[i], al[j]) if al[i] < al[j] else (al[j], al[i])
            keys.append("aa:" + x + "|" + y)
    for d in nu:
        keys.append("x:" + d)
        keys.extend("ax:" + d + "|" + w for w in al)
        keys.extend("nx:" + d + "|" + t for t in nt[:4])
    return keys


def hash_records(names: list[str], addrs: list[str], cfg: KeyConfig = KeyConfig()) -> sparse.csr_matrix:
    from sklearn.feature_extraction import FeatureHasher

    fh = FeatureHasher(n_features=N_FEATURES, input_type="string", alternate_sign=False, dtype=np.float32)
    m = fh.transform(record_keys(nm, ad, cfg) for nm, ad in zip(names, addrs))
    m.data[:] = 1.0
    return m.tocsr()


def _hash_chunk(args):
    names, addrs, cfg = args
    return hash_records(names, addrs, cfg)


def hash_records_parallel(names: list[str], addrs: list[str], cfg: KeyConfig = KeyConfig(),
                          n_workers: int = 10, chunk: int = 200_000) -> sparse.csr_matrix:
    from multiprocessing import get_context

    jobs = [(names[i:i + chunk], addrs[i:i + chunk], cfg) for i in range(0, len(names), chunk)]
    if n_workers <= 1 or len(jobs) == 1:
        parts = [_hash_chunk(j) for j in jobs]
    else:
        with get_context("spawn").Pool(n_workers) as pool:
            parts = pool.map(_hash_chunk, jobs, chunksize=1)
    return sparse.vstack(parts, format="csr")


def weight_partition(m_s1: sparse.csr_matrix, m_t: sparse.csr_matrix, max_df: int):
    """Drop keys with total df > max_df or absent from one side; TF-IDF weight
    (idf over the partition's own S1+target records) and L2 normalize rows.
    Returns compact-column (A_s1, A_t, stats)."""
    n_docs = m_s1.shape[0] + m_t.shape[0]
    u1, c1 = np.unique(m_s1.indices, return_counts=True)
    u2, c2 = np.unique(m_t.indices, return_counts=True)
    both = np.intersect1d(u1, u2, assume_unique=True)
    df = c1[np.searchsorted(u1, both)] + c2[np.searchsorted(u2, both)]
    keep = both[df <= max_df]
    df_keep = df[df <= max_df]
    idf = (np.log((n_docs + 1) / (df_keep + 1)) + 1.0).astype(np.float32)
    s1_df = c1[np.searchsorted(u1, keep)].astype(np.int64)
    t_df = c2[np.searchsorted(u2, keep)].astype(np.int64)
    stats = {
        "n_docs": int(n_docs),
        "keys_s1": int(len(u1)),
        "keys_t": int(len(u2)),
        "keys_shared": int(len(both)),
        "keys_kept": int(len(keep)),
        "matmul_cost": int((s1_df * t_df).sum()),
    }

    def compact(m):
        m = m.tocsr()
        pos = np.searchsorted(keep, m.indices)
        pos_c = np.minimum(pos, len(keep) - 1)
        ok = keep[pos_c] == m.indices if len(keep) else np.zeros(len(m.indices), bool)
        rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
        out = sparse.csr_matrix((idf[pos_c[ok]], (rows[ok], pos_c[ok])), shape=(m.shape[0], len(keep)), dtype=np.float32)
        norms = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sparse.csr_matrix(sparse.diags(1.0 / norms).astype(np.float32) @ out)

    return compact(m_s1), compact(m_t), stats


def topk_rows(sims: sparse.csr_matrix, k: int):
    """Per-row top-k of a CSR matrix. Returns (row, col, score, rank) arrays
    (rank 1 = best; ties broken by column index for determinism)."""
    sims = sims.tocsr()
    sims.sum_duplicates()
    counts = np.diff(sims.indptr)
    rows = np.repeat(np.arange(sims.shape[0], dtype=np.int64), counts)
    order = np.lexsort((sims.indices, -sims.data, rows))
    rows_o = rows[order]
    start = np.repeat(sims.indptr[:-1], counts)
    rank = np.arange(len(order), dtype=np.int64) - start + 1
    sel = rank <= k
    return rows_o[sel], sims.indices[order][sel], sims.data[order][sel], rank[sel].astype(np.int16)


def save_csr(prefix: str, m: sparse.csr_matrix) -> None:
    np.save(prefix + "_data.npy", m.data)
    np.save(prefix + "_indices.npy", m.indices)
    np.save(prefix + "_indptr.npy", m.indptr)
    np.save(prefix + "_shape.npy", np.array(m.shape, dtype=np.int64))


def load_csr(prefix: str, mmap: bool = True) -> sparse.csr_matrix:
    mode = "r" if mmap else None
    shape = tuple(np.load(prefix + "_shape.npy"))
    return sparse.csr_matrix(
        (np.load(prefix + "_data.npy", mmap_mode=mode), np.load(prefix + "_indices.npy", mmap_mode=mode),
         np.load(prefix + "_indptr.npy", mmap_mode=mode)), shape=shape, copy=False)


_W = {}


def _init_worker(query_prefix: str, index_t_prefix: str):
    _W["q"] = load_csr(query_prefix)
    _W["it"] = load_csr(index_t_prefix)  # index transposed: (n_keys x n_index) CSR


def _retrieve_chunk(args):
    lo, hi, k = args
    q = _W["q"][lo:hi]
    sims = q @ _W["it"]
    r, c, s, rk = topk_rows(sims, k)
    return lo + r, c, s, rk


def retrieve_topk(query_prefix: str, index_t_prefix: str, n_query: int, k: int,
                  n_workers: int = 3, chunk: int = 5_000):
    from multiprocessing import get_context

    jobs = [(lo, min(lo + chunk, n_query), k) for lo in range(0, n_query, chunk)]
    if not jobs:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32), np.zeros(0, np.int16)
    with get_context("spawn").Pool(n_workers, initializer=_init_worker, initargs=(query_prefix, index_t_prefix)) as pool:
        parts = pool.map(_retrieve_chunk, jobs, chunksize=1)
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def prepare_partition(workdir: str, tag: str, a_s1: sparse.csr_matrix, a_t: sparse.csr_matrix) -> dict:
    """Write query and transposed-index matrices for both directions."""
    os.makedirs(workdir, exist_ok=True)
    p = {k: os.path.join(workdir, f"{tag}_{k}") for k in ("s1", "t", "s1T", "tT")}
    save_csr(p["s1"], a_s1)
    save_csr(p["t"], a_t)
    save_csr(p["s1T"], a_s1.T.tocsr())
    save_csr(p["tT"], a_t.T.tocsr())
    return p
