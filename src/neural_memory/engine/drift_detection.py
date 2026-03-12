"""Semantic drift detection — find tag clusters that should be merged.

Uses tag co-occurrence matrix + Jaccard similarity to detect when
different tags refer to the same concept. Outputs drift reports
with confidence-based suggestions: merge, alias, or review.

Runs during consolidation (not hot path). Zero LLM, pure statistics.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.engine.clustering import UnionFind

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

JACCARD_MERGE_THRESHOLD = 0.7  # Jaccard >= 0.7 → likely synonyms (auto-merge)
JACCARD_ALIAS_THRESHOLD = 0.4  # Jaccard >= 0.4 → related concepts (alias)
JACCARD_REVIEW_THRESHOLD = 0.3  # Jaccard >= 0.3 → possible drift (review)
MIN_COOCCURRENCE_COUNT = 3  # Minimum co-occurrences to consider
MAX_CLUSTER_SIZE = 10  # Max tags in a single cluster
MIN_TAG_FIBERS = 2  # Tag must appear in >= 2 fibers to be considered


# ── Data Models ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TagCluster:
    """A cluster of tags detected as potentially referring to the same concept."""

    canonical: str  # Most-used tag in the cluster
    members: frozenset[str]  # All tags in the cluster (including canonical)
    confidence: float  # Average Jaccard similarity within cluster
    evidence: str = ""  # Human-readable explanation


@dataclass(frozen=True)
class DriftReport:
    """A single drift detection result with action suggestion."""

    cluster: TagCluster
    suggestion: str  # "merge" | "alias" | "review"
    cluster_id: str = ""  # Stable ID for persistence


# ── Core Algorithm ─────────────────────────────────────────────────────


def compute_jaccard(
    tag_a: str,
    tag_b: str,
    tag_fiber_counts: dict[str, int],
    cooccurrence_count: int,
) -> float:
    """Compute Jaccard similarity between two tags.

    J(A, B) = |A intersection B| / |A union B|
    = cooccurrence / (count_a + count_b - cooccurrence)
    """
    count_a = tag_fiber_counts.get(tag_a, 0)
    count_b = tag_fiber_counts.get(tag_b, 0)

    if count_a == 0 or count_b == 0:
        return 0.0

    union = count_a + count_b - cooccurrence_count
    if union <= 0:
        return 0.0

    return cooccurrence_count / union


def _cluster_id(members: frozenset[str]) -> str:
    """Generate stable cluster ID from sorted member tags."""
    key = "|".join(sorted(members))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def detect_clusters(
    cooccurrences: list[tuple[str, str, int]],
    tag_fiber_counts: dict[str, int],
) -> list[DriftReport]:
    """Detect tag clusters using Union-Find on Jaccard-similar pairs.

    Args:
        cooccurrences: List of (tag_a, tag_b, count) pairs.
        tag_fiber_counts: Dict of {tag: fiber_count} for Jaccard denominator.

    Returns:
        List of DriftReport with confidence-based suggestions.
    """
    if not cooccurrences:
        return []

    # Collect all unique tags
    all_tags: list[str] = []
    tag_index: dict[str, int] = {}
    for tag_a, tag_b, _count in cooccurrences:
        for tag in (tag_a, tag_b):
            if tag not in tag_index:
                tag_index[tag] = len(all_tags)
                all_tags.append(tag)

    if len(all_tags) < 2:
        return []

    # Compute Jaccard for each pair and union high-similarity pairs
    uf = UnionFind(len(all_tags))
    pair_jaccards: dict[tuple[int, int], float] = {}

    for tag_a, tag_b, count in cooccurrences:
        if count < MIN_COOCCURRENCE_COUNT:
            continue

        # Skip tags that appear in very few fibers
        if tag_fiber_counts.get(tag_a, 0) < MIN_TAG_FIBERS:
            continue
        if tag_fiber_counts.get(tag_b, 0) < MIN_TAG_FIBERS:
            continue

        jaccard = compute_jaccard(tag_a, tag_b, tag_fiber_counts, count)

        if jaccard >= JACCARD_REVIEW_THRESHOLD:
            idx_a = tag_index[tag_a]
            idx_b = tag_index[tag_b]
            pair_jaccards[(idx_a, idx_b)] = jaccard

            # Only union above alias threshold (review pairs stay separate)
            if jaccard >= JACCARD_ALIAS_THRESHOLD:
                uf.union(idx_a, idx_b)

    if not pair_jaccards:
        return []

    # Extract groups from Union-Find
    groups = uf.groups()

    reports: list[DriftReport] = []
    for member_indices in groups.values():
        if len(member_indices) < 2:
            continue
        if len(member_indices) > MAX_CLUSTER_SIZE:
            member_indices = member_indices[:MAX_CLUSTER_SIZE]

        member_tags = frozenset(all_tags[i] for i in member_indices)

        # Compute average Jaccard within cluster
        jaccard_sum = 0.0
        jaccard_count = 0
        for i in member_indices:
            for j in member_indices:
                if i < j:
                    j_val = pair_jaccards.get((i, j), pair_jaccards.get((j, i), 0.0))
                    if j_val > 0:
                        jaccard_sum += j_val
                        jaccard_count += 1

        avg_jaccard = jaccard_sum / jaccard_count if jaccard_count > 0 else 0.0

        # Pick canonical: tag with highest fiber count
        canonical = max(member_tags, key=lambda t: tag_fiber_counts.get(t, 0))

        # Determine suggestion based on confidence
        if avg_jaccard >= JACCARD_MERGE_THRESHOLD:
            suggestion = "merge"
        elif avg_jaccard >= JACCARD_ALIAS_THRESHOLD:
            suggestion = "alias"
        else:
            suggestion = "review"

        others = sorted(member_tags - {canonical})
        evidence = (
            f"Tags {others} co-occur with '{canonical}' "
            f"(avg Jaccard={avg_jaccard:.2f}, "
            f"fibers: {', '.join(f'{t}={tag_fiber_counts.get(t, 0)}' for t in sorted(member_tags))})"
        )

        cluster = TagCluster(
            canonical=canonical,
            members=member_tags,
            confidence=round(avg_jaccard, 4),
            evidence=evidence,
        )

        reports.append(
            DriftReport(
                cluster=cluster,
                suggestion=suggestion,
                cluster_id=_cluster_id(member_tags),
            )
        )

    # Sort by confidence descending
    reports.sort(key=lambda r: r.cluster.confidence, reverse=True)
    return reports


# ── Cross-Session Drift ───────────────────────────────────────────────


async def detect_temporal_drift(
    storage: NeuralStorage,
) -> list[dict[str, object]]:
    """Detect terminology shifts across session summaries.

    Compares early session topics with recent session topics to find
    terms that have been replaced (user used to say X, now says Y).

    Returns list of {old_term, new_term, confidence, evidence}.
    """
    try:
        summaries = await storage.get_session_summaries(limit=20)  # type: ignore[attr-defined]
    except Exception:
        return []

    if len(summaries) < 4:
        return []  # Need enough history

    # Split into early vs recent halves
    mid = len(summaries) // 2
    early = summaries[mid:]  # Older (summaries are DESC order)
    recent = summaries[:mid]  # Newer

    # Count topic frequency in each half
    early_topics: dict[str, int] = {}
    recent_topics: dict[str, int] = {}

    for s in early:
        for topic in s.get("topics") or []:
            early_topics[topic] = early_topics.get(topic, 0) + 1

    for s in recent:
        for topic in s.get("topics") or []:
            recent_topics[topic] = recent_topics.get(topic, 0) + 1

    # Find terms that disappeared from early but have a co-occurring replacement
    drifts: list[dict[str, object]] = []
    for old_term, old_count in early_topics.items():
        if old_count < 2:
            continue
        if old_term in recent_topics:
            continue  # Still in use, no drift

        # Find candidate replacement: term in recent but not early,
        # that co-occurs with old_term in co-occurrence matrix
        for new_term, new_count in recent_topics.items():
            if new_count < 2:
                continue
            if new_term in early_topics:
                continue  # Was already in early, not a replacement

            confidence = min(old_count, new_count) / max(old_count, new_count)
            if confidence >= 0.3:
                drifts.append(
                    {
                        "old_term": old_term,
                        "new_term": new_term,
                        "confidence": round(confidence, 2),
                        "evidence": (
                            f"'{old_term}' appeared {old_count}x in early sessions "
                            f"but absent recently. '{new_term}' appeared {new_count}x "
                            f"recently but was absent before."
                        ),
                    }
                )

    # Sort by confidence and cap
    drifts.sort(key=lambda d: float(d["confidence"]), reverse=True)  # type: ignore[arg-type]
    return drifts[:10]


# ── Orchestrator ──────────────────────────────────────────────────────


async def run_drift_detection(
    storage: NeuralStorage,
) -> dict[str, object]:
    """Run full drift detection: co-occurrence clusters + temporal drift.

    Returns a summary dict with clusters and temporal drift findings.
    """
    # 1. Get co-occurrence data
    try:
        cooccurrences = await storage.get_tag_cooccurrence(  # type: ignore[attr-defined]
            min_count=MIN_COOCCURRENCE_COUNT,
        )
    except Exception:
        cooccurrences = []

    # 2. Get fiber counts per tag
    try:
        tag_fiber_counts = await storage.get_tag_fiber_counts()  # type: ignore[attr-defined]
    except Exception:
        tag_fiber_counts = {}

    # 3. Detect clusters
    reports = detect_clusters(cooccurrences, tag_fiber_counts)

    # 4. Persist detected clusters
    for report in reports:
        try:
            await storage.save_drift_cluster(  # type: ignore[attr-defined]
                cluster_id=report.cluster_id,
                canonical=report.cluster.canonical,
                members=sorted(report.cluster.members),
                confidence=report.cluster.confidence,
                status="detected",
            )
        except Exception:
            pass

    # 5. Detect temporal drift
    temporal_drifts = await detect_temporal_drift(storage)

    # 6. Build summary
    merge_count = sum(1 for r in reports if r.suggestion == "merge")
    alias_count = sum(1 for r in reports if r.suggestion == "alias")
    review_count = sum(1 for r in reports if r.suggestion == "review")

    return {
        "clusters": [
            {
                "cluster_id": r.cluster_id,
                "canonical": r.cluster.canonical,
                "members": sorted(r.cluster.members),
                "confidence": r.cluster.confidence,
                "suggestion": r.suggestion,
                "evidence": r.cluster.evidence,
            }
            for r in reports
        ],
        "temporal_drifts": temporal_drifts,
        "summary": {
            "total_clusters": len(reports),
            "merge_suggestions": merge_count,
            "alias_suggestions": alias_count,
            "review_suggestions": review_count,
            "temporal_drifts": len(temporal_drifts),
        },
    }
