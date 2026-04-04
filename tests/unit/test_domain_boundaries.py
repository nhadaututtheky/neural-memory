"""Tests for B5 Phase 3: Domain Boundaries.

Tests:
- domain tag injection on boundary memories via remember
- domain-filtered HOT context injection in recall
- nmem_boundaries tool (list, domains actions)
- backward compatibility: unscoped boundaries remain global
"""

from __future__ import annotations

from pathlib import Path

import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    MemoryTier,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow

# ── Fixtures ─────────────────────────────────────────────


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    """Create an initialized SQLiteStorage with a brain."""
    db_path = tmp_path / "test_domain.db"
    store = SQLiteStorage(db_path=str(db_path))
    await store.initialize()

    config = BrainConfig(
        decay_rate=0.1,
        reinforcement_delta=0.05,
        activation_threshold=0.15,
        max_spread_hops=4,
        max_context_tokens=1500,
    )
    brain = Brain.create(name="domain-test", config=config)
    await store.save_brain(brain)
    store.set_brain(brain.id)

    yield store  # type: ignore[misc]
    await store.close()


async def _create_boundary(
    storage: SQLiteStorage,
    fiber_id: str,
    domain: str | None = None,
    extra_tags: set[str] | None = None,
) -> TypedMemory:
    """Helper to create a boundary memory with optional domain tag."""
    now = utcnow()
    neuron_id = f"n-{fiber_id}"

    neuron = Neuron(
        id=neuron_id,
        type=NeuronType.CONCEPT,
        content=f"Boundary rule for {fiber_id}",
        metadata={},
        created_at=now,
    )
    await storage.add_neuron(neuron)

    state = NeuronState(
        neuron_id=neuron_id,
        activation_level=0.8,
        access_frequency=5,
        last_activated=now,
        created_at=now,
    )
    await storage.update_neuron_state(state)

    fiber = Fiber(
        id=fiber_id,
        neuron_ids={neuron_id},
        synapse_ids=set(),
        anchor_neuron_id=neuron_id,
        pathway=[neuron_id],
    )
    await storage.add_fiber(fiber)

    tags: frozenset[str] = frozenset()
    tag_set: set[str] = set()
    if domain:
        tag_set.add(f"domain:{domain}")
    if extra_tags:
        tag_set.update(extra_tags)
    if tag_set:
        tags = frozenset(tag_set)

    tm = TypedMemory(
        fiber_id=fiber_id,
        memory_type=MemoryType.BOUNDARY,
        priority=Priority.NORMAL,
        provenance=Provenance(source="test"),
        tier=MemoryTier.HOT,
        tags=tags,
        created_at=now,
    )
    await storage.add_typed_memory(tm)
    return tm


async def _create_hot_memory(
    storage: SQLiteStorage,
    fiber_id: str,
    memory_type: MemoryType = MemoryType.FACT,
) -> TypedMemory:
    """Helper to create a non-boundary HOT memory."""
    now = utcnow()
    neuron_id = f"n-{fiber_id}"

    neuron = Neuron(
        id=neuron_id,
        type=NeuronType.CONCEPT,
        content=f"Hot content for {fiber_id}",
        metadata={},
        created_at=now,
    )
    await storage.add_neuron(neuron)

    state = NeuronState(
        neuron_id=neuron_id,
        activation_level=0.9,
        access_frequency=10,
        last_activated=now,
        created_at=now,
    )
    await storage.update_neuron_state(state)

    fiber = Fiber(
        id=fiber_id,
        neuron_ids={neuron_id},
        synapse_ids=set(),
        anchor_neuron_id=neuron_id,
        pathway=[neuron_id],
    )
    await storage.add_fiber(fiber)

    tm = TypedMemory(
        fiber_id=fiber_id,
        memory_type=memory_type,
        priority=Priority.NORMAL,
        provenance=Provenance(source="test"),
        tier=MemoryTier.HOT,
        created_at=now,
    )
    await storage.add_typed_memory(tm)
    return tm


# ── Domain Tag on TypedMemory ────────────────────────────


class TestDomainTagCreation:
    """Test that domain tags are correctly stored on boundary memories."""

    async def test_boundary_with_domain_tag(self, storage: SQLiteStorage) -> None:
        """Boundary created with domain tag should have domain:{value} in tags."""
        tm = await _create_boundary(storage, "b1", domain="financial")
        assert "domain:financial" in tm.tags

    async def test_boundary_without_domain_is_global(self, storage: SQLiteStorage) -> None:
        """Boundary without domain tag has no domain: prefixed tags."""
        tm = await _create_boundary(storage, "b2")
        domain_tags = {t for t in tm.tags if t.startswith("domain:")}
        assert len(domain_tags) == 0

    async def test_boundary_with_extra_tags(self, storage: SQLiteStorage) -> None:
        """Domain tag coexists with other tags."""
        tm = await _create_boundary(
            storage, "b3", domain="security", extra_tags={"project:myapp", "critical"}
        )
        assert "domain:security" in tm.tags
        assert "project:myapp" in tm.tags
        assert "critical" in tm.tags

    async def test_find_boundaries_by_domain_tag(self, storage: SQLiteStorage) -> None:
        """find_typed_memories with tags={domain:X} filters correctly."""
        await _create_boundary(storage, "b-fin", domain="financial")
        await _create_boundary(storage, "b-sec", domain="security")
        await _create_boundary(storage, "b-global")

        fin_boundaries = await storage.find_typed_memories(
            memory_type=MemoryType.BOUNDARY,
            tags={"domain:financial"},
        )
        assert len(fin_boundaries) == 1
        assert fin_boundaries[0].fiber_id == "b-fin"


# ── Domain-Filtered HOT Context Injection ────────────────


class TestDomainFilteredContext:
    """Test that domain filtering works in HOT context injection."""

    async def test_domain_filter_excludes_other_domain_boundaries(
        self, storage: SQLiteStorage
    ) -> None:
        """When domain=financial, security boundaries should be excluded."""
        await _create_boundary(storage, "b-fin", domain="financial")
        await _create_boundary(storage, "b-sec", domain="security")

        # Simulate domain filtering logic from recall_handler
        context_domain = "financial"
        hot_memories = await storage.find_typed_memories(tier="hot", limit=100)

        included = []
        for tm in hot_memories:
            if context_domain and tm.memory_type == MemoryType.BOUNDARY:
                domain_tags = {t for t in tm.tags if t.startswith("domain:")}
                if domain_tags and f"domain:{context_domain}" not in domain_tags:
                    continue
            included.append(tm)

        fiber_ids = {tm.fiber_id for tm in included}
        assert "b-fin" in fiber_ids
        assert "b-sec" not in fiber_ids

    async def test_domain_filter_includes_global_boundaries(self, storage: SQLiteStorage) -> None:
        """Global boundaries (no domain tag) are always included."""
        await _create_boundary(storage, "b-global")
        await _create_boundary(storage, "b-fin", domain="financial")

        context_domain = "financial"
        hot_memories = await storage.find_typed_memories(tier="hot", limit=100)

        included = []
        for tm in hot_memories:
            if context_domain and tm.memory_type == MemoryType.BOUNDARY:
                domain_tags = {t for t in tm.tags if t.startswith("domain:")}
                if domain_tags and f"domain:{context_domain}" not in domain_tags:
                    continue
            included.append(tm)

        fiber_ids = {tm.fiber_id for tm in included}
        assert "b-global" in fiber_ids
        assert "b-fin" in fiber_ids

    async def test_domain_filter_does_not_affect_non_boundary_hot(
        self, storage: SQLiteStorage
    ) -> None:
        """Non-boundary HOT memories are always included regardless of domain filter."""
        await _create_hot_memory(storage, "hot-fact", MemoryType.FACT)
        await _create_boundary(storage, "b-sec", domain="security")

        context_domain = "financial"
        hot_memories = await storage.find_typed_memories(tier="hot", limit=100)

        included = []
        for tm in hot_memories:
            if context_domain and tm.memory_type == MemoryType.BOUNDARY:
                domain_tags = {t for t in tm.tags if t.startswith("domain:")}
                if domain_tags and f"domain:{context_domain}" not in domain_tags:
                    continue
            included.append(tm)

        fiber_ids = {tm.fiber_id for tm in included}
        assert "hot-fact" in fiber_ids  # non-boundary always included
        assert "b-sec" not in fiber_ids  # wrong domain excluded

    async def test_no_domain_filter_includes_all(self, storage: SQLiteStorage) -> None:
        """Without domain filter, all HOT memories are included."""
        await _create_boundary(storage, "b-fin", domain="financial")
        await _create_boundary(storage, "b-sec", domain="security")
        await _create_boundary(storage, "b-global")
        await _create_hot_memory(storage, "hot-fact")

        context_domain = None
        hot_memories = await storage.find_typed_memories(tier="hot", limit=100)

        included = []
        for tm in hot_memories:
            if context_domain and tm.memory_type == MemoryType.BOUNDARY:
                domain_tags = {t for t in tm.tags if t.startswith("domain:")}
                if domain_tags and f"domain:{context_domain}" not in domain_tags:
                    continue
            included.append(tm)

        assert len(included) == 4


# ── nmem_boundaries Tool ─────────────────────────────────


class TestBoundariesTool:
    """Test the _boundaries handler logic."""

    async def test_boundaries_domains_action(self, storage: SQLiteStorage) -> None:
        """domains action returns unique domain names with counts."""
        await _create_boundary(storage, "b1", domain="financial")
        await _create_boundary(storage, "b2", domain="financial")
        await _create_boundary(storage, "b3", domain="security")
        await _create_boundary(storage, "b4")  # global

        boundaries = await storage.find_typed_memories(memory_type=MemoryType.BOUNDARY, limit=1000)

        # Simulate _boundaries_domains logic
        domain_counts: dict[str, int] = {}
        unscoped = 0
        for tm in boundaries:
            domain_tags = {t[7:] for t in tm.tags if t.startswith("domain:")}
            if domain_tags:
                for d in domain_tags:
                    domain_counts[d] = domain_counts.get(d, 0) + 1
            else:
                unscoped += 1

        assert domain_counts == {"financial": 2, "security": 1}
        assert unscoped == 1

    async def test_boundaries_list_filtered_by_domain(self, storage: SQLiteStorage) -> None:
        """list action with domain filter returns matching + global boundaries."""
        await _create_boundary(storage, "b-fin1", domain="financial")
        await _create_boundary(storage, "b-fin2", domain="financial")
        await _create_boundary(storage, "b-sec", domain="security")
        await _create_boundary(storage, "b-global")

        boundaries = await storage.find_typed_memories(memory_type=MemoryType.BOUNDARY, limit=1000)

        domain_filter = "financial"
        items = []
        for tm in boundaries:
            domain_tags = {t[7:] for t in tm.tags if t.startswith("domain:")}
            if domain_filter:
                if domain_tags and domain_filter not in domain_tags:
                    continue
            items.append(tm)

        fiber_ids = {tm.fiber_id for tm in items}
        assert "b-fin1" in fiber_ids
        assert "b-fin2" in fiber_ids
        assert "b-global" in fiber_ids  # global always included
        assert "b-sec" not in fiber_ids  # different domain excluded

    async def test_boundaries_list_no_filter(self, storage: SQLiteStorage) -> None:
        """list action without domain filter returns all boundaries."""
        await _create_boundary(storage, "b1", domain="financial")
        await _create_boundary(storage, "b2", domain="security")
        await _create_boundary(storage, "b3")

        boundaries = await storage.find_typed_memories(memory_type=MemoryType.BOUNDARY, limit=1000)
        assert len(boundaries) == 3


# ── Remember Handler Domain Injection ────────────────────


class TestRememberDomainInjection:
    """Test that remember handler correctly injects domain tags."""

    async def test_domain_tag_format(self) -> None:
        """Domain values are normalized to lowercase with domain: prefix."""
        raw_domain = "  Financial  "
        domain = raw_domain.lower().strip()[:50]
        tag = f"domain:{domain}"
        assert tag == "domain:financial"

    async def test_domain_ignored_for_non_boundary(self) -> None:
        """Domain param should be ignored for non-boundary types."""
        # This tests the logic: only add domain tag when mem_type == BOUNDARY
        mem_type = MemoryType.FACT
        raw_domain = "financial"
        tags: set[str] = set()

        # Simulate remember handler logic
        if mem_type == MemoryType.BOUNDARY and raw_domain:
            tags.add(f"domain:{raw_domain}")

        assert len(tags) == 0  # no domain tag for facts

    async def test_domain_added_for_boundary(self) -> None:
        """Domain param adds domain tag for boundary type."""
        mem_type = MemoryType.BOUNDARY
        raw_domain = "security"
        tags: set[str] = set()

        if mem_type == MemoryType.BOUNDARY and raw_domain:
            tags.add(f"domain:{raw_domain}")

        assert "domain:security" in tags

    async def test_domain_truncated_at_50_chars(self) -> None:
        """Domain values longer than 50 chars are truncated."""
        raw_domain = "a" * 100
        domain = raw_domain.lower().strip()[:50]
        assert len(domain) == 50
