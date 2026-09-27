"""Candidate identity contract for Phase-2 blocking.

Per the Phase-2 mission spec §7: unless dataset evidence proves target IDs
are globally unique across S2 and S3, candidate identity is the composite
key (s1_entity_id, target_source, target_entity_id), never a bare
target_entity_id. `entity_id` values in this dataset are already prefixed
(`S2-...`, `S3-...`), which already makes them source-qualified strings, but
we still carry `target_source` as its own explicit field everywhere so that
union/dedup/metrics code can never accidentally compare or merge an S2 id
against an S3 id that happens to share a numeric suffix, and so provenance
survives even if a future data revision drops the prefix convention.
"""

from __future__ import annotations

SOURCE_S2 = "S2"
SOURCE_S3 = "S3"
VALID_TARGET_SOURCES = (SOURCE_S2, SOURCE_S3)


def target_source_of(entity_id: str) -> str:
    """Derive target_source from a prefixed entity_id (S2-... / S3-...)."""
    if entity_id.startswith("S2-"):
        return SOURCE_S2
    if entity_id.startswith("S3-"):
        return SOURCE_S3
    raise ValueError(f"entity_id {entity_id!r} has no recognized S2/S3 prefix")


def candidate_key(s1_entity_id: str, target_source: str, target_entity_id: str) -> tuple[str, str, str]:
    if target_source not in VALID_TARGET_SOURCES:
        raise ValueError(f"invalid target_source {target_source!r}")
    return (s1_entity_id, target_source, target_entity_id)


def assert_no_cross_source_collision(s2_ids: set[str], s3_ids: set[str]) -> None:
    """Raise if any raw ID string is shared between the S2 and S3 ID sets.

    This is a defensive integrity check, not expected to ever fire given the
    dataset's own S2-/S3- prefixing, but it makes the "S2 and S3 IDs must
    never collide semantically" requirement an enforced invariant rather
    than an assumption.
    """
    overlap = s2_ids & s3_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} entity_id value(s) appear in both S2 and S3 raw ID sets: "
            f"{sorted(overlap)[:5]}..."
        )
