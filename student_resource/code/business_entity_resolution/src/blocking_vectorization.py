"""TF-IDF fitting with explicit provenance (Phase-2 mission spec §11).

Authoritative fitting policy (frozen for this Phase-2 experiment round; see
docs/PHASE_2_BLOCKING_REPORT.md section 4 for the human-readable version):

1. Vocabulary corpus = normalized joint (name + address) text of:
     - ALL train Source-2 rows,
     - ALL train Source-3 rows,
     - train Source-1 rows RESTRICTED to the `development` split only.
   S2/S3 have no split property (only Source-1 entities are split by
   `validation_v1`); they are the retrieval target universe in both real
   inference and this experiment, so restricting them would not reflect
   production reality and would require peeking at ground truth to decide
   which rows to exclude. What IS restricted to `development` is the
   query-side (Source-1) text: no `validation`-split Source-1 record is
   ever read by any fitting step in Phase 2.
2. IDF corpus = the same corpus as (1) (no separate IDF-only corpus).
3. S1/S2/S3 are fit JOINTLY into one shared vocabulary/IDF (one vectorizer),
   not three separate per-source vectorizers.
4. S2 and S3 share the same fitted statistics (no per-target-source refit).
5. No country partitioning at the fitting stage: one global fit. Country
   routing, if used at all, is applied only at the retrieval/candidate
   stage, never by refitting vocabulary/IDF per country (this also sidesteps
   the open-set France categorical problem entirely at this stage).
6. EXP-B001 (forward) and EXP-B001R (reverse) use the exact same fitted
   `TfidfVectorizer` object and therefore the same vocabulary/IDF/matrix
   representation for a given row, satisfying the "isolate direction, not
   representation" requirement.
7. EXP-B002 (character n-gram) is a SEPARATELY fit vectorizer (different
   representation family), with its own provenance record. It is never
   silently compared against B001 as if they shared statistics.
8. No test-derived, validation-derived, or hidden-label-derived statistic is
   used anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from blocking_normalization import NORMALIZATION_VERSION

WORD_TOKEN_PATTERN = r"(?u)\b\w+\b"


@dataclass(frozen=True)
class VectorizerConfig:
    """Everything needed to reproduce a fitted vectorizer's behavior."""

    representation: Literal["word", "char_ngram"]
    normalization_version: str = NORMALIZATION_VERSION
    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 1)
    token_pattern: str | None = WORD_TOKEN_PATTERN
    min_df: int = 2
    max_df: float = 0.35
    sublinear_tf: bool = True
    norm: str = "l2"
    corpus_scope: str = "s1_development_plus_all_s2_s3"
    fit_id: str = "unset"

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class FittedVectorizer:
    config: VectorizerConfig
    vectorizer: TfidfVectorizer
    n_documents_fit: int
    vocabulary_size: int
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.config_hash = self.config.config_hash()

    def provenance(self) -> dict:
        return {
            "representation": self.config.representation,
            "normalization_version": self.config.normalization_version,
            "analyzer": self.config.analyzer,
            "ngram_range": list(self.config.ngram_range),
            "min_df": self.config.min_df,
            "max_df": self.config.max_df,
            "sublinear_tf": self.config.sublinear_tf,
            "norm": self.config.norm,
            "corpus_scope": self.config.corpus_scope,
            "fit_id": self.config.fit_id,
            "n_documents_fit": self.n_documents_fit,
            "vocabulary_size": self.vocabulary_size,
            "config_hash": self.config_hash,
        }


def build_vectorizer(config: VectorizerConfig) -> TfidfVectorizer:
    if config.representation == "word":
        return TfidfVectorizer(
            analyzer="word",
            token_pattern=config.token_pattern,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            sublinear_tf=config.sublinear_tf,
            norm=config.norm,
            dtype=np.float32,
        )
    if config.representation == "char_ngram":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            sublinear_tf=config.sublinear_tf,
            norm=config.norm,
            dtype=np.float32,
        )
    raise ValueError(f"unknown representation {config.representation!r}")


def fit_vectorizer(corpus_texts: list[str], config: VectorizerConfig) -> FittedVectorizer:
    vec = build_vectorizer(config)
    vec.fit(corpus_texts)
    return FittedVectorizer(
        config=config,
        vectorizer=vec,
        n_documents_fit=len(corpus_texts),
        vocabulary_size=len(vec.vocabulary_),
    )
