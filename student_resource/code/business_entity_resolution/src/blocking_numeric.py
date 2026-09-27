"""Numeric-address-token blocking route (EXP-B003).

Positive-evidence-only index: build a mapping from each individual numeric
token found in a normalized business_address to the set of rows containing
it, then generate candidates for every S1 row that shares at least one
numeric token with a target row. This never rejects a candidate for numeric
*disagreement* -- disagreement is a downstream matcher feature (per the
Phase-2 mission spec §19.1), not a blocking-stage filter. No ubiquitous-
token protection is silently invented: EXP-B000/EXP-B003 measure the
collision-group-size distribution directly and this module exposes the
building blocks (posting lists) needed for that measurement.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from blocking_normalization import numeric_tokens


def build_numeric_postings(entity_ids: list[str], addresses: list[str]) -> dict[str, list[str]]:
    """token -> list of entity_id containing that numeric token."""
    postings: dict[str, list[str]] = defaultdict(list)
    for eid, addr in zip(entity_ids, addresses):
        for tok in set(numeric_tokens(addr)):
            postings[tok].append(eid)
    return postings


def numeric_token_sets(entity_ids: list[str], addresses: list[str]) -> dict[str, frozenset[str]]:
    return {eid: frozenset(numeric_tokens(addr)) for eid, addr in zip(entity_ids, addresses)}


def numeric_candidates(
    s1_ids: list[str],
    s1_addresses: list[str],
    target_ids: list[str],
    target_addresses: list[str],
    max_posting_size: int | None = None,
    min_token_length: int = 1,
) -> pd.DataFrame:
    """Return long-format candidate edges [s1_entity_id, target_entity_id]
    for every (S1, target) pair sharing >=1 numeric address token.

    Uses the target-side posting list (built once) and probes it per S1 row,
    which is cheap because numeric tokens are far sparser than word tokens
    and each S1 row typically has only a handful of numeric tokens.

    `max_posting_size` (measured first via `numeric_collision_group_sizes`,
    never guessed) drops tokens whose posting list exceeds it -- a rarity
    cutoff analogous to TF-IDF's `max_df`, so a catastrophically common
    numeric token (e.g. a generic unit number) cannot blow up candidate load.
    `min_token_length` optionally drops single/double-digit tokens that are
    positive evidence in principle but empirically too generic to be useful.
    """
    target_postings = build_numeric_postings(target_ids, target_addresses)
    if max_posting_size is not None:
        target_postings = {
            tok: ids for tok, ids in target_postings.items() if len(ids) <= max_posting_size
        }
    rows_s1: list[str] = []
    rows_target: list[str] = []
    for s1_id, addr in zip(s1_ids, s1_addresses):
        seen: set[str] = set()
        for tok in set(numeric_tokens(addr)):
            if len(tok) < min_token_length:
                continue
            for tid in target_postings.get(tok, ()):
                if tid not in seen:
                    seen.add(tid)
                    rows_s1.append(s1_id)
                    rows_target.append(tid)
    return pd.DataFrame({"s1_entity_id": rows_s1, "target_entity_id": rows_target})


def numeric_collision_group_sizes(postings: dict) -> pd.Series:
    """Group size (posting-list length) per key, for collision analysis.
    Generic over the key type -- used for both individual numeric tokens
    and the exact-signature route below.
    """
    return pd.Series({key: len(ids) for key, ids in postings.items()}).sort_values(ascending=False)


def numeric_signature(address: str | None) -> tuple[str, ...] | None:
    """Deterministic, non-empty numeric-token signature for an address:
    the sorted tuple of its distinct normalized numeric tokens (order- and
    duplicate-insensitive, so "12 MG Road Building 4" and "Building 4,
    12 M.G. Rd" -- which extract the same numeric tokens {12, 4} in a
    different order -- map to the identical signature ("12", "4")).

    Returns None (never an empty tuple) when the address has no numeric
    tokens at all -- an empty signature must never be usable as positive
    evidence, or every address without a single number would spuriously
    collide with every other numberless address.
    """
    toks = numeric_tokens(address or "")
    if not toks:
        return None
    return tuple(sorted(set(toks)))


def build_signature_postings(
    entity_ids: list[str], addresses: list[str]
) -> dict[tuple[str, ...], list[str]]:
    """signature -> list of entity_id sharing that exact non-empty
    numeric-token signature (EXP-B003 Route A: exact-set evidence)."""
    postings: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for eid, addr in zip(entity_ids, addresses):
        sig = numeric_signature(addr)
        if sig is not None:
            postings[sig].append(eid)
    return postings


def exact_numeric_signature_candidates(
    s1_ids: list[str],
    s1_addresses: list[str],
    target_ids: list[str],
    target_addresses: list[str],
    max_group_size: int | None = None,
) -> pd.DataFrame:
    """Route A: candidates for every (S1, target) pair sharing an IDENTICAL
    non-empty numeric-token signature -- much higher specificity than
    single-token overlap (Route B, `numeric_candidates`), since it requires
    the full set of numeric tokens to match exactly, not just one of them.

    `max_group_size` (measured first via `numeric_collision_group_sizes`
    on the postings this function itself builds, never guessed) drops
    signatures whose group exceeds it.
    """
    target_postings = build_signature_postings(target_ids, target_addresses)
    if max_group_size is not None:
        target_postings = {
            sig: ids for sig, ids in target_postings.items() if len(ids) <= max_group_size
        }
    rows_s1: list[str] = []
    rows_target: list[str] = []
    for s1_id, addr in zip(s1_ids, s1_addresses):
        sig = numeric_signature(addr)
        if sig is None:
            continue
        for tid in target_postings.get(sig, ()):
            rows_s1.append(s1_id)
            rows_target.append(tid)
    return pd.DataFrame({"s1_entity_id": rows_s1, "target_entity_id": rows_target})
