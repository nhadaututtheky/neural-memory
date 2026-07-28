"""Regression tests for the summary_fiber tag-repetition pollution loop (PR #194).

_summarize() used to re-ingest its own summary fibers: they carry the same tags
as their source cluster, so each cycle nested the previous cycle's "[tags] …"
prefix into the new concept content — compounding exponentially. These tests pin:

1. summary_fiber entries are excluded from the clustering pool,
2. polluted "[tags] [tags] …" summaries never reach new concept content,
3. duplicate summaries are deduped before joining,
4. _essence_backfill skips anchors whose content matches the pollution pattern.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.consolidation import (
    _TAG_POLLUTION_RE,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from neural_memory.storage.memory_store import InMemoryStorage

POLLUTED_SUMMARY = "[character, novel] [character, novel] [character, novel]"


@pytest_asyncio.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="pollution_test", brain_id="pollution-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


async def _add_fiber(
    store: InMemoryStorage,
    idx: int,
    *,
    summary: str,
    tags: set[str],
    metadata: dict | None = None,
) -> Fiber:
    anchor = Neuron.create(
        type=NeuronType.ENTITY,
        content=f"anchor content {idx}",
        neuron_id=f"n-{idx}",
    )
    await store.add_neuron(anchor)
    fiber = Fiber.create(
        neuron_ids={anchor.id},
        synapse_ids=set(),
        anchor_neuron_id=anchor.id,
        summary=summary,
        tags=tags,
        metadata=metadata or {},
        fiber_id=f"fiber-{idx}",
    )
    await store.add_fiber(fiber)
    return fiber


class TestTagPollutionRegex:
    def test_matches_repeated_tag_groups(self) -> None:
        assert _TAG_POLLUTION_RE.match(POLLUTED_SUMMARY)
        assert _TAG_POLLUTION_RE.match("[a] [b]")
        assert _TAG_POLLUTION_RE.match("[x, y]  [x, y] trailing text")

    def test_does_not_match_single_bracket_prefix(self) -> None:
        # The legitimate "[tags] summary" format has exactly one bracket group.
        assert not _TAG_POLLUTION_RE.match("[api, auth] Fixed the login bug")
        assert not _TAG_POLLUTION_RE.match("plain summary text")
        assert not _TAG_POLLUTION_RE.match("[only one] then text with [brackets] later")


@pytest.mark.asyncio
async def test_summary_fiber_excluded_from_clustering(storage: InMemoryStorage) -> None:
    """Existing summary fibers must not be re-ingested into new clusters."""
    tags = {"character", "novel"}
    for i in range(3):
        await _add_fiber(storage, i, summary=f"source summary {i}", tags=tags)
    old_summary = await _add_fiber(
        storage,
        99,
        summary=POLLUTED_SUMMARY,
        tags=tags,
        metadata={"_consolidation": "summary_fiber", "source_fibers": []},
    )

    engine = ConsolidationEngine(storage)
    await engine.run(strategies=[ConsolidationStrategy.SUMMARIZE])

    new_summary_fibers = [
        f
        for f in await storage.get_fibers(limit=1000)
        if f.metadata.get("_consolidation") == "summary_fiber" and f.id != old_summary.id
    ]
    assert new_summary_fibers, "expected a new summary fiber for the cluster"
    for fiber in new_summary_fibers:
        assert old_summary.id not in fiber.metadata.get("source_fibers", [])
        assert POLLUTED_SUMMARY not in (fiber.summary or "")


@pytest.mark.asyncio
async def test_polluted_summary_filtered_from_concept_content(
    storage: InMemoryStorage,
) -> None:
    """A source fiber with a polluted summary must not leak into concept content."""
    tags = {"alpha", "beta"}
    await _add_fiber(storage, 0, summary=POLLUTED_SUMMARY, tags=tags)
    await _add_fiber(storage, 1, summary="clean summary one", tags=tags)
    await _add_fiber(storage, 2, summary="clean summary two", tags=tags)

    engine = ConsolidationEngine(storage)
    await engine.run(strategies=[ConsolidationStrategy.SUMMARIZE])

    concepts = [
        n
        for n in (await storage.find_neurons(type=NeuronType.CONCEPT))
        if n.metadata.get("_consolidation") == "summary"
    ]
    assert concepts, "expected a concept neuron for the cluster"
    for concept in concepts:
        assert POLLUTED_SUMMARY not in concept.content
        assert "clean summary one" in concept.content


@pytest.mark.asyncio
async def test_duplicate_summaries_deduped(storage: InMemoryStorage) -> None:
    """Identical member summaries appear once in the joined concept content."""
    tags = {"gamma", "delta"}
    for i in range(3):
        await _add_fiber(storage, i, summary="repeated summary", tags=tags)

    engine = ConsolidationEngine(storage)
    await engine.run(strategies=[ConsolidationStrategy.SUMMARIZE])

    concepts = [
        n
        for n in (await storage.find_neurons(type=NeuronType.CONCEPT))
        if n.metadata.get("_consolidation") == "summary"
    ]
    assert concepts
    for concept in concepts:
        assert concept.content.count("repeated summary") == 1


@pytest.mark.asyncio
async def test_essence_backfill_skips_polluted_anchor(storage: InMemoryStorage) -> None:
    """Anchors whose content matches the pollution pattern get no essence."""
    polluted_anchor = Neuron.create(
        type=NeuronType.CONCEPT,
        content=POLLUTED_SUMMARY,
        neuron_id="n-polluted",
    )
    clean_anchor = Neuron.create(
        type=NeuronType.ENTITY,
        content="The deploy failed because the API key expired. Rotating it fixed CI.",
        neuron_id="n-clean",
    )
    await storage.add_neuron(polluted_anchor)
    await storage.add_neuron(clean_anchor)
    polluted_fiber = Fiber.create(
        neuron_ids={polluted_anchor.id},
        synapse_ids=set(),
        anchor_neuron_id=polluted_anchor.id,
        summary="polluted",
        fiber_id="fiber-polluted",
    )
    clean_fiber = Fiber.create(
        neuron_ids={clean_anchor.id},
        synapse_ids=set(),
        anchor_neuron_id=clean_anchor.id,
        summary="clean",
        fiber_id="fiber-clean",
    )
    await storage.add_fiber(polluted_fiber)
    await storage.add_fiber(clean_fiber)

    engine = ConsolidationEngine(storage)
    await engine.run(strategies=[ConsolidationStrategy.ESSENCE_BACKFILL])

    refreshed_polluted = await storage.get_fiber("fiber-polluted")
    refreshed_clean = await storage.get_fiber("fiber-clean")
    assert refreshed_polluted is not None and not refreshed_polluted.essence
    assert refreshed_clean is not None and refreshed_clean.essence
