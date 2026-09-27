"""Production normalization for Phase-2 candidate generation.

Distinct from `normalization.py` (explicitly audit-only, not frozen for
production per its own docstring). This module is the first production
normalization: NFKC + casefold + whitespace collapse, with no country-
specific rules, no legal-suffix stripping, no transliteration dictionary.
Those remain deferred (EXP-B001I is a conditional, separately-gated future
ablation; see docs/research/RESEARCH_TO_EXPERIMENT_PLAN.md).

Version tag: bump `NORMALIZATION_VERSION` whenever the normalization logic
changes, and record it in every experiment's vectorizer provenance so a
candidate artifact is always traceable to the exact normalization that
produced it.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "prod_v1_nfkc_casefold"

_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_TOKEN_RE = re.compile(r"\d+")
_WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_field(value: str | None) -> str:
    """NFKC-normalize, casefold, and collapse whitespace. No other rules."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.casefold()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def joint_text(business_name: str | None, business_address: str | None) -> str:
    """Concatenate normalized name + address into one retrieval document.

    This is the single joint representation EXP-B001/EXP-B001R vectorize.
    Field order is fixed (name then address) for reproducibility; it has no
    effect on a bag-of-words/TF-IDF representation.
    """
    name = normalize_field(business_name)
    addr = normalize_field(business_address)
    if name and addr:
        return f"{name} {addr}"
    return name or addr


def word_tokens(value: str | None) -> list[str]:
    """Unicode-aware word tokenization on already-normalized text."""
    if not value:
        return []
    return _WORD_TOKEN_RE.findall(value)


def numeric_tokens(value: str | None) -> list[str]:
    """Extract numeric substrings (house numbers, PIN-like codes, etc.)."""
    if not value:
        return []
    return _NUMERIC_TOKEN_RE.findall(value)


def name_token_set(business_name: str | None) -> frozenset[str]:
    return frozenset(word_tokens(normalize_field(business_name)))


def numeric_token_set(business_address: str | None) -> frozenset[str]:
    return frozenset(numeric_tokens(business_address or ""))
