"""Context Compiler: cross-fiber dedup, information merge, and query-relevance re-scoring.

Pure functions only. No external dependencies beyond stdlib + neural_memory.utils.simhash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime

from neural_memory.utils.simhash import DEFAULT_THRESHOLD, hamming_distance, simhash

# Sentence boundary: split on ". " or ".\n", keeping reasonable granularity.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class CompiledChunk:
    fiber_id: str
    content: str
    activation_score: float
    created_at: datetime | None
    summary: str | None
    final_score: float = field(default=0.0)


def compile_context(
    chunks: list[CompiledChunk],
    query_terms: list[str],
    dedup_threshold: int = DEFAULT_THRESHOLD,
    max_merge_sentences: int = 2,
    relevance_boost_per_term: float = 0.15,
    relevance_boost_cap: float = 0.3,
) -> list[CompiledChunk]:
    """Main compile pipeline: dedup → merge → re-score → sort.

    Args:
        chunks: Input chunks to compile. Not mutated.
        query_terms: Terms used for relevance boosting (case-insensitive).
        dedup_threshold: Max hamming distance to consider chunks near-duplicate.
        max_merge_sentences: Max unique sentences to append from duplicates.
        relevance_boost_per_term: Score boost added per matched query term.
        relevance_boost_cap: Maximum total boost from query terms.

    Returns:
        New list of CompiledChunk sorted by final_score descending.
    """
    if not chunks:
        return []

    groups = _dedup_groups(chunks, dedup_threshold)
    merged = [_merge_group(g, max_merge_sentences) for g in groups]
    rescored = _rescore(merged, query_terms, relevance_boost_per_term, relevance_boost_cap)
    return sorted(rescored, key=lambda c: c.final_score, reverse=True)


def _dedup_groups(
    chunks: list[CompiledChunk],
    threshold: int,
) -> list[list[CompiledChunk]]:
    """Group near-duplicate chunks by SimHash hamming distance.

    Uses a greedy union-find-style grouping: each chunk is assigned to the
    first existing group whose representative fingerprint is within threshold.

    Args:
        chunks: Chunks to group.
        threshold: Max hamming distance to consider near-duplicate.

    Returns:
        List of groups (each group is a non-empty list of chunks).
    """
    # Each entry: (representative_fingerprint, [chunks in group])
    groups: list[tuple[int, list[CompiledChunk]]] = []

    for chunk in chunks:
        fp = simhash(chunk.content)
        placed = False
        for _i, (rep_fp, group_members) in enumerate(groups):
            if hamming_distance(fp, rep_fp) <= threshold:
                group_members.append(chunk)
                placed = True
                break
        if not placed:
            groups.append((fp, [chunk]))

    return [members for _, members in groups]


def _merge_group(
    group: list[CompiledChunk],
    max_extra_sentences: int,
) -> CompiledChunk:
    """Merge a group of near-duplicates into one chunk.

    Keeps the chunk with the highest activation_score as primary.
    Appends up to max_extra_sentences unique sentences from the other chunks.

    Args:
        group: Non-empty list of near-duplicate chunks.
        max_extra_sentences: Max sentences to append from secondary chunks.

    Returns:
        A new CompiledChunk (never mutates inputs).
    """
    primary = max(group, key=lambda c: c.activation_score)

    if len(group) == 1:
        return primary

    primary_sentences = set(_split_sentences(primary.content))
    extra: list[str] = []

    for chunk in group:
        if chunk is primary:
            continue
        for sentence in _split_sentences(chunk.content):
            if sentence not in primary_sentences and sentence not in extra:
                extra.append(sentence)
                if len(extra) >= max_extra_sentences:
                    break
        if len(extra) >= max_extra_sentences:
            break

    if not extra:
        return primary

    merged_content = primary.content.rstrip() + " " + " ".join(extra)
    return replace(primary, content=merged_content)


def _rescore(
    chunks: list[CompiledChunk],
    query_terms: list[str],
    boost_per_term: float,
    boost_cap: float,
) -> list[CompiledChunk]:
    """Apply query-relevance boost to activation scores.

    For each chunk, counts how many query_terms appear (case-insensitive) in
    the content. Adds boost_per_term per match, capped at boost_cap.

    Args:
        chunks: Chunks to rescore.
        query_terms: Search terms for relevance boost.
        boost_per_term: Score added per matched term.
        boost_cap: Maximum total boost.

    Returns:
        New list of CompiledChunk with final_score set.
    """
    lower_terms = [t.lower() for t in query_terms]
    result: list[CompiledChunk] = []

    for chunk in chunks:
        lower_content = chunk.content.lower()
        match_count = sum(1 for term in lower_terms if term and term in lower_content)
        boost = min(match_count * boost_per_term, boost_cap)
        result.append(replace(chunk, final_score=chunk.activation_score + boost))

    return result


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences.

    Args:
        text: Input text.

    Returns:
        List of stripped, non-empty sentence strings.
    """
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
