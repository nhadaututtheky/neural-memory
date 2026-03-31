"""Cross-brain recall — parallel spreading activation across multiple brains.

Resolves brain names to DB paths, opens temporary SQLiteStorage instances,
runs SA in parallel via asyncio.gather, deduplicates results by SimHash,
and merges by confidence.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.engine.retrieval_types import DepthLevel
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from pathlib import Path

    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

# Hard cap on number of brains to query
MAX_CROSS_BRAINS = 5


@dataclass(frozen=True)
class CrossBrainFiber:
    """A fiber result from a cross-brain query."""

    fiber_id: str
    source_brain: str
    summary: str
    confidence: float
    content_hash: int = 0


@dataclass(frozen=True)
class CrossBrainResult:
    """Aggregated result from cross-brain recall."""

    query: str
    brains_queried: list[str]
    fibers: list[CrossBrainFiber]
    total_neurons_activated: int = 0
    merged_context: str = ""


async def _query_single_brain(
    db_path: Path,
    brain_name: str,
    query: str,
    depth: DepthLevel,
    max_tokens: int,
    tags: set[str] | None = None,
    tag_mode: str = "and",
) -> tuple[str, list[CrossBrainFiber], int, str]:
    """Query a single brain and return its results.

    Returns:
        Tuple of (brain_name, fibers, neurons_activated, context)
    """
    from neural_memory.storage.sqlite_store import SQLiteStorage

    storage = SQLiteStorage(db_path)
    try:
        await storage.initialize()

        # Find the brain by name in the DB
        brain = await storage.find_brain_by_name(brain_name)
        if not brain:
            return brain_name, [], 0, ""

        storage.set_brain(brain.id)

        from neural_memory.engine.retrieval import ReflexPipeline

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=utcnow(),
            tags=tags,
            tag_mode=tag_mode,
        )

        fibers: list[CrossBrainFiber] = []
        for fid in result.fibers_matched:
            fiber = await storage.get_fiber(fid)
            if fiber:
                fibers.append(
                    CrossBrainFiber(
                        fiber_id=fid,
                        source_brain=brain_name,
                        summary=fiber.summary or "",
                        confidence=result.confidence,
                        content_hash=getattr(fiber, "content_hash", 0) or 0,
                    )
                )

        return brain_name, fibers, result.neurons_activated, result.context or ""
    except Exception:
        logger.debug("Cross-brain query failed for '%s'", brain_name, exc_info=True)
        return brain_name, [], 0, ""
    finally:
        await storage.close()


def _dedup_fibers(fibers: list[CrossBrainFiber]) -> list[CrossBrainFiber]:
    """Deduplicate fibers by SimHash proximity.

    Keeps the fiber with the highest confidence when duplicates are found.
    """
    from neural_memory.utils.simhash import is_near_duplicate

    result: list[CrossBrainFiber] = []
    seen_hashes: list[tuple[int, int]] = []  # (hash, index in result)

    for fiber in fibers:
        if fiber.content_hash == 0:
            result.append(fiber)
            continue

        is_dup = False
        for existing_hash, idx in seen_hashes:
            if existing_hash != 0 and is_near_duplicate(fiber.content_hash, existing_hash):
                # Keep the one with higher confidence
                if fiber.confidence > result[idx].confidence:
                    result[idx] = fiber
                is_dup = True
                break

        if not is_dup:
            seen_hashes.append((fiber.content_hash, len(result)))
            result.append(fiber)

    return result


async def cross_brain_recall(
    config: UnifiedConfig,
    brain_names: list[str],
    query: str,
    depth: int = 1,
    max_tokens: int = 500,
    tags: set[str] | None = None,
    tag_mode: str = "and",
) -> CrossBrainResult:
    """Run recall across multiple brains in parallel.

    Args:
        config: Unified configuration (for brain DB path resolution)
        brain_names: List of brain names to query (max 5)
        query: The recall query
        depth: Depth level (0-3)
        max_tokens: Max tokens per brain query

    Returns:
        CrossBrainResult with merged, deduplicated results
    """
    # Cap brain count
    brain_names = brain_names[:MAX_CROSS_BRAINS]

    # Resolve brain names to DB paths
    valid_brains: list[tuple[str, Path]] = []
    available = set(config.list_brains())

    for name in brain_names:
        if name not in available:
            logger.debug("Brain '%s' not found, skipping", name)
            continue
        try:
            db_path = config.get_brain_db_path(name)
            if db_path.exists():
                valid_brains.append((name, db_path))
        except ValueError:
            logger.debug("Invalid brain name '%s', skipping", name)

    if not valid_brains:
        return CrossBrainResult(
            query=query,
            brains_queried=[],
            fibers=[],
            merged_context="No valid brains found to query.",
        )

    try:
        depth_level = DepthLevel(depth)
    except ValueError:
        depth_level = DepthLevel.CONTEXT

    # Query all brains in parallel
    tasks = [
        _query_single_brain(
            db_path, name, query, depth_level, max_tokens, tags=tags, tag_mode=tag_mode
        )
        for name, db_path in valid_brains
    ]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    all_fibers: list[CrossBrainFiber] = []
    total_neurons = 0
    context_parts: list[str] = []
    queried: list[str] = []

    for brain_name, fibers, neurons_activated, context in results:
        queried.append(brain_name)
        all_fibers.extend(fibers)
        total_neurons += neurons_activated
        if context:
            context_parts.append(f"[{brain_name}] {context}")

    # Sort by confidence descending
    all_fibers.sort(key=lambda f: f.confidence, reverse=True)

    # Deduplicate by SimHash
    deduped = _dedup_fibers(all_fibers)

    # Merge context
    merged_context = "\n\n".join(context_parts) if context_parts else "No relevant memories found."

    return CrossBrainResult(
        query=query,
        brains_queried=queried,
        fibers=deduped,
        total_neurons_activated=total_neurons,
        merged_context=merged_context,
    )
