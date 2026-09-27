"""Normalization v2: v1 (NFKC + casefold + whitespace collapse) plus
  1. a train-derived script->Latin token alias map for business names
     (alias_map_matcher_train.json, learned only from matcher_train positive
     links by positional token alignment of script-named targets with their
     Latin Source-1 names; support >= 2, confidence >= 0.6), and
  2. removal of combining diacritics on Latin letters ("Àmicale" -> "amicale").
Diacritics are removed only when the base character is Latin, so combining
vowel signs of Indic scripts are preserved. No country-specific rules.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from blocking_normalization import normalize_field

_MAP_PATH = Path(__file__).resolve().parent / "alias_map_matcher_train.json"
_SCRIPT_RE = re.compile(r"[ऀ-෿]")
_STRIP_RE = re.compile(r"[^\wऀ-෿]")


@lru_cache(maxsize=1)
def alias_map() -> dict[str, str]:
    with open(_MAP_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def strip_latin_diacritics(text: str) -> str:
    if text.isascii():
        return text
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and out and ord(out[-1]) < 0x250:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def normalize_name_v2(value: str | None) -> str:
    text = normalize_field(value)
    if not text:
        return ""
    if _SCRIPT_RE.search(text):
        mp = alias_map()
        toks = []
        for tok in text.split():
            key = _STRIP_RE.sub("", tok)
            toks.append(mp.get(key, tok))
        text = " ".join(toks)
    return strip_latin_diacritics(text)


def normalize_address_v2(value: str | None) -> str:
    return strip_latin_diacritics(normalize_field(value))
