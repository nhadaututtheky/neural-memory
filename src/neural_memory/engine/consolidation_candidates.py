"""Bounded candidate generation for incremental consolidation.

Avoids all-pairs O(N²) scans by bucketing dirty items against local
neighborhoods (SimHash bands, tags, type, time windows).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from neural_memory.engine.consolidation_incremental import DirtySet
from neural_memory.utils.simhash import is_near_duplicate

DEFAULT_CANDIDATE_CAP = 10_000


@dataclass(frozen=True)
class CandidateBuckets:
    """Bounded candidate sets for merge/dedup style strategies."""

    fiber_pairs: tuple[tuple[str, str], ...]
    neuron_pairs: tuple[tuple[str, str], ...]
    truncated: bool
    bucket_sizes: dict[str, int]

    @property
    def pair_count(self) -> int:
        return len(self.fiber_pairs) + len(self.neuron_pairs)


def _band_key(content_hash: int, bands: int = 4) -> str:
    """Coarse SimHash band for locality-sensitive bucketing."""
    # Split 64-bit hash into bands
    band_bits = 64 // max(1, bands)
    parts: list[str] = []
    h = int(content_hash) & ((1 << 64) - 1)
    for i in range(bands):
        shift = i * band_bits
        parts.append(str((h >> shift) & ((1 << band_bits) - 1)))
    return "|".join(parts[:2])  # use first 2 bands for coarser buckets


def build_merge_candidates(
    dirty: DirtySet,
    fibers: list[Any],
    *,
    max_candidates: int = DEFAULT_CANDIDATE_CAP,
    max_bucket: int = 50,
) -> CandidateBuckets:
    """Generate fiber merge candidate pairs involving at least one dirty fiber.

    Compares dirty fibers only against fibers that share a SimHash band,
    overlapping tags, or the same primary type — never full Cartesian product.
    """
    max_candidates = max(1, min(int(max_candidates), 100_000))
    max_bucket = max(2, min(int(max_bucket), 200))

    dirty_f = dirty.fiber_ids
    dirty_n = dirty.neuron_ids

    # Index all fibers by bucket keys
    by_band: dict[str, list[Any]] = defaultdict(list)
    by_tag: dict[str, list[Any]] = defaultdict(list)
    by_type: dict[str, list[Any]] = defaultdict(list)
    fiber_by_id: dict[str, Any] = {}

    for f in fibers:
        fid = getattr(f, "id", None)
        if not isinstance(fid, str):
            continue
        fiber_by_id[fid] = f
        # content hash from metadata or 0
        meta = getattr(f, "metadata", None) or {}
        ch = int(meta.get("content_hash") or meta.get("_content_hash") or 0)
        by_band[_band_key(ch)].append(f)
        tags = getattr(f, "tags", None) or set()
        for tag in list(tags)[:8]:
            by_tag[str(tag).lower()].append(f)
        ftype = str(meta.get("type") or meta.get("memory_type") or "unknown")
        by_type[ftype].append(f)

    # Candidate fiber set: dirty fibers + neighbors in shared buckets
    seed_ids = set(dirty_f)
    if not seed_ids and dirty_n:
        # Fibers that touch dirty neurons
        for f in fibers:
            nids = getattr(f, "neuron_ids", None) or set()
            if nids & dirty_n:
                seed_ids.add(f.id)

    pair_set: set[tuple[str, str]] = set()
    truncated = False
    bucket_sizes: dict[str, int] = {}

    def _add_pairs(seed: Any, bucket: list[Any], label: str) -> None:
        nonlocal truncated
        if truncated:
            return
        size = min(len(bucket), max_bucket)
        bucket_sizes[label] = max(bucket_sizes.get(label, 0), size)
        if len(bucket) > max_bucket:
            truncated = True
        seed_id = seed.id
        for other in bucket[:max_bucket]:
            oid = other.id
            if oid == seed_id:
                continue
            a, b = (seed_id, oid) if seed_id < oid else (oid, seed_id)
            pair_set.add((a, b))
            if len(pair_set) >= max_candidates:
                truncated = True
                return

    for sid in seed_ids:
        seed = fiber_by_id.get(sid)
        if seed is None:
            continue
        meta = getattr(seed, "metadata", None) or {}
        ch = int(meta.get("content_hash") or meta.get("_content_hash") or 0)
        _add_pairs(seed, by_band[_band_key(ch)], "simhash_band")
        if truncated:
            break
        tags = getattr(seed, "tags", None) or set()
        for tag in list(tags)[:5]:
            _add_pairs(seed, by_tag[str(tag).lower()], f"tag:{tag}")
            if truncated:
                break
        ftype = str(meta.get("type") or meta.get("memory_type") or "unknown")
        _add_pairs(seed, by_type[ftype], f"type:{ftype}")
        if truncated:
            break

    # Optional: filter pairs that are near-duplicates by SimHash when available
    refined: list[tuple[str, str]] = []
    for a, b in sorted(pair_set)[:max_candidates]:
        fa, fb = fiber_by_id.get(a), fiber_by_id.get(b)
        if fa is None or fb is None:
            continue
        ma = getattr(fa, "metadata", None) or {}
        mb = getattr(fb, "metadata", None) or {}
        ha = ma.get("content_hash") or ma.get("_content_hash")
        hb = mb.get("content_hash") or mb.get("_content_hash")
        if ha is not None and hb is not None:
            try:
                if not is_near_duplicate(int(ha), int(hb), threshold=10):
                    # Still keep as structural candidate if tag-overlap high
                    ta = set(getattr(fa, "tags", None) or set())
                    tb = set(getattr(fb, "tags", None) or set())
                    if ta and tb:
                        overlap = len(ta & tb) / max(1, len(ta | tb))
                        if overlap < 0.3:
                            continue
            except (TypeError, ValueError):
                pass
        refined.append((a, b))

    return CandidateBuckets(
        fiber_pairs=tuple(refined[:max_candidates]),
        neuron_pairs=(),
        truncated=truncated or len(pair_set) > max_candidates,
        bucket_sizes=dict(bucket_sizes),
    )


def filter_entities_to_dirty(
    entities: list[Any],
    dirty: DirtySet,
    *,
    id_attr: str = "id",
    entity_kind: str = "neuron",
) -> list[Any]:
    """Filter a list of entities down to those present in the dirty set."""
    if entity_kind == "neuron":
        allowed = dirty.neuron_ids
    elif entity_kind == "synapse":
        allowed = dirty.synapse_ids
    elif entity_kind == "fiber":
        allowed = dirty.fiber_ids
    else:
        return entities
    if not allowed:
        return []
    out: list[Any] = []
    for e in entities:
        eid = getattr(e, id_attr, None)
        if isinstance(eid, str) and eid in allowed:
            out.append(e)
    return out
