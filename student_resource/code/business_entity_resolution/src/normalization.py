"""AUDIT NORMALIZATION - NOT FROZEN FOR PRODUCTION.

`normalize_for_audit` exists only to characterize the dataset in Phase 1. It
is deliberately conservative and reversible in spirit: Unicode
normalization, casefold, whitespace/punctuation-spacing cleanup. It does not
strip legal suffixes, remove business-type words, transliterate, remove
address numbers, or apply country-specific rules - those are production
normalization decisions Phase 1 explicitly defers.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_SPACING_RE = re.compile(r"\s*([,.;:])\s*")
_NUMERIC_TOKEN_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_for_audit(value: str) -> str:
    """Minimal, conservative normalization for descriptive audit purposes only."""
    if value is None or value == "":
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.casefold()
    text = _PUNCT_SPACING_RE.sub(r"\1 ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(value: str) -> list[str]:
    """Simple Unicode-aware word tokenizer for descriptive Jaccard/set stats."""
    if not value:
        return []
    return _TOKEN_RE.findall(value)


def numeric_tokens(value: str) -> list[str]:
    """Extract numeric substrings (house numbers, PIN-like codes, etc.)."""
    if not value:
        return []
    return _NUMERIC_TOKEN_RE.findall(value)


def token_jaccard(a: str, b: str) -> float | None:
    """Jaccard similarity of the normalized token sets of two strings.

    Returns None when both strings are empty (undefined similarity), rather
    than a misleading 1.0 or 0.0.
    """
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta and not tb:
        return None
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
