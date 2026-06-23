"""Tests for SurfaceGenerator — brain.db → KnowledgeSurface extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.surface.generator import SurfaceGenerator
from neural_memory.surface.models import DepthLevel, SignalLevel
from neural_memory.utils.timeutils import utcnow

# ── Fake Fiber ─────────────────────────────────────


@dataclass
class FakeFiber:
    id: str = "fiber-1"
    neuron_ids: set[str] = field(default_factory=set)
    synapse_ids: set[str] = field(default_factory=set)
    anchor_neuron_id: str = ""
    pathway: list[str] = field(default_factory=list)
    conductivity: float = 1.0
    last_conducted: datetime | None = None
    time_start: datetime | None = None
    time_end: datetime | None = None
    coherence: float = 0.5
    salience: float = 0.5
    frequency: int = 1
    summary: str | None = None
    auto_tags: set[str] = field(default_factory=set)
    agent_tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    compression_tier: int = 0
    pinned: bool = False
    created_at: datetime = field(default_factory=utcnow)

    @property
    def tags(self) -> set[str]:
        return self.auto_tags | self.agent_tags


# ── Fake NeuronState ───────────────────────────────


@dataclass
class FakeNeuronState:
    neuron_id: str
    activation_level: float = 0.5
    access_frequency: int = 3
    last_activated: datetime | None = None
    decay_rate: float = 0.1
    firing_threshold: float = 0.3
    refractory_until: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)


# ── Test Helpers ───────────────────────────────────


def _now() -> datetime:
    return utcnow()


def _make_synapse(
    sid: str = "s1",
    source_id: str = "n1",
    target_id: str = "n2",
    stype: SynapseType = SynapseType.RELATED_TO,
    weight: float = 0.5,
) -> Synapse:
    return Synapse.create(
        source_id=source_id,
        target_id=target_id,
        type=stype,
        weight=weight,
    )


def _make_storage(
    neurons: list[Neuron] | None = None,
    states: dict[str, FakeNeuronState] | None = None,
    synapses_out: dict[str, list[Synapse]] | None = None,
    fibers: list[FakeFiber] | None = None,
    target_neurons: dict[str, Neuron] | None = None,
    stats: dict[str, int] | None = None,
) -> AsyncMock:
    """Build a mock storage with configurable responses."""
    storage = AsyncMock()

    neuron_list = neurons or []
    state_map = states or {}
    synapse_map = synapses_out or {}
    fiber_list = fibers or []
    target_map = target_neurons or {}
    stat_dict = stats or {"neuron_count": len(neuron_list), "synapse_count": 0, "fiber_count": 0}

    storage.find_neurons = AsyncMock(return_value=neuron_list)
    storage.get_neuron_states_batch = AsyncMock(return_value=state_map)
    storage.get_synapses_for_neurons = AsyncMock(return_value=synapse_map)
    storage.get_neurons_batch = AsyncMock(return_value=target_map)
    storage.find_fibers = AsyncMock(return_value=fiber_list)
    storage.get_fibers = AsyncMock(return_value=fiber_list)
    storage.get_synapses = AsyncMock(return_value=[])
    storage.get_stats = AsyncMock(return_value=stat_dict)

    return storage


def _neuron_with_id(
    nid: str,
    ntype: NeuronType = NeuronType.CONCEPT,
    content: str = "Test concept",
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> Neuron:
    """Create a Neuron with a specific ID for testing."""
    from dataclasses import replace as dc_replace

    n = Neuron.create(type=ntype, content=content, metadata=metadata or {}, neuron_id=nid)
    if created_at:
        return dc_replace(n, created_at=created_at)
    return n


# ── Tests: Empty Brain ─────────────────────────────


async def test_empty_brain_produces_valid_surface():
    """Empty brain should produce valid but empty surface."""
    storage = _make_storage()
    gen = SurfaceGenerator(storage, brain_name="empty")
    surface = await gen.generate()

    assert surface.frontmatter.brain == "empty"
    assert surface.graph == ()
    assert surface.clusters == ()
    assert surface.signals == ()
    assert surface.depth_map == ()
    assert surface.meta.coverage == 0.0


# ── Tests: Top Neuron Selection ────────────────────


async def test_top_neurons_selected_by_composite_score():
    """Neurons with higher activation + recency score higher."""
    now = _now()
    high_act = _neuron_with_id("n1", content="High activation concept", created_at=now)
    low_act = _neuron_with_id(
        "n2", content="Low activation concept", created_at=now - timedelta(days=60)
    )

    states = {
        "n1": FakeNeuronState(neuron_id="n1", activation_level=0.9),
        "n2": FakeNeuronState(neuron_id="n2", activation_level=0.1),
    }

    storage = _make_storage(
        neurons=[high_act, low_act],
        states=states,
        stats={"neuron_count": 2, "synapse_count": 0, "fiber_count": 0},
    )

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    assert len(surface.graph) == 2
    # First node should be the high-activation one
    assert surface.graph[0].node.priority >= surface.graph[1].node.priority


# ── Tests: GRAPH Edge Extraction ───────────────────


async def test_graph_edges_from_synapses():
    """Synapses between top neurons become GRAPH edges."""
    now = _now()
    n1 = _neuron_with_id("n1", content="PostgreSQL", created_at=now)
    n2 = _neuron_with_id("n2", content="MongoDB", created_at=now)

    syn = _make_synapse(
        source_id="n1",
        target_id="n2",
        stype=SynapseType.RELATED_TO,
    )

    states = {
        "n1": FakeNeuronState(neuron_id="n1", activation_level=0.8),
        "n2": FakeNeuronState(neuron_id="n2", activation_level=0.7),
    }

    storage = _make_storage(
        neurons=[n1, n2],
        states=states,
        synapses_out={"n1": [syn], "n2": []},
        target_neurons={"n2": n2},
    )

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    # n1 should have an edge to n2
    n1_entry = next((e for e in surface.graph if "PostgreSQL" in e.node.content), None)
    assert n1_entry is not None
    assert len(n1_entry.edges) >= 1
    assert n1_entry.edges[0].edge_type == "related_to"


async def test_skips_temporal_synapse_types():
    """HAPPENED_AT, BEFORE, AFTER, DURING are skipped in GRAPH."""
    now = _now()
    n1 = _neuron_with_id("n1", content="Some event", created_at=now)

    temporal_syn = _make_synapse(
        source_id="n1",
        target_id="t1",
        stype=SynapseType.HAPPENED_AT,
    )

    storage = _make_storage(
        neurons=[n1],
        states={"n1": FakeNeuronState(neuron_id="n1", activation_level=0.8)},
        synapses_out={"n1": [temporal_syn]},
    )

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    # No edges because HAPPENED_AT is skipped
    assert len(surface.graph) == 1
    assert len(surface.graph[0].edges) == 0


async def test_forward_edge_reference_resolves_to_surface_id():
    """Regression (#40): an edge to a lower-scored (later-iterated) target
    must still carry a target_id backref, not degrade to text-only."""
    now = _now()
    # n1 is higher-scored (iterated first) and points forward to n2.
    n1 = _neuron_with_id("n1", content="High score source", created_at=now)
    n2 = _neuron_with_id("n2", content="Low score target", created_at=now)

    syn = _make_synapse(source_id="n1", target_id="n2", stype=SynapseType.RELATED_TO)

    states = {
        "n1": FakeNeuronState(neuron_id="n1", activation_level=0.95),
        "n2": FakeNeuronState(neuron_id="n2", activation_level=0.10),
    }

    storage = _make_storage(
        neurons=[n1, n2],
        states=states,
        synapses_out={"n1": [syn], "n2": []},
        target_neurons={"n2": n2},
    )

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    n1_entry = next((e for e in surface.graph if "source" in e.node.content), None)
    n2_entry = next((e for e in surface.graph if "target" in e.node.content), None)
    assert n1_entry is not None
    assert n2_entry is not None
    assert len(n1_entry.edges) >= 1
    # Forward reference must resolve to the assigned surface node id of n2.
    assert n1_entry.edges[0].target_id == n2_entry.node.id


# ── Tests: CLUSTERS ────────────────────────────────


async def test_clusters_from_fiber_cooccurrence():
    """Neurons co-occurring in fibers get clustered."""
    now = _now()
    n1 = _neuron_with_id("n1", content="PostgreSQL database", created_at=now)
    n2 = _neuron_with_id("n2", content="Payment processing", created_at=now)

    # Both neurons appear in same fiber
    fiber = FakeFiber(
        id="f1",
        neuron_ids={"n1", "n2"},
        created_at=now,
    )

    states = {
        "n1": FakeNeuronState(neuron_id="n1", activation_level=0.7),
        "n2": FakeNeuronState(neuron_id="n2", activation_level=0.6),
    }

    storage = _make_storage(
        neurons=[n1, n2],
        states=states,
        fibers=[fiber],
        target_neurons={"n1": n1, "n2": n2},
    )
    storage.get_neurons_batch = AsyncMock(return_value={"n1": n1, "n2": n2})

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    # Should have at least 1 cluster with both neurons
    assert len(surface.clusters) >= 1

    # Regression (#41): cluster node_ids must be SHORT surface ids (e.g. 'c1'),
    # mapped from the graph pass — never raw neuron UUIDs.
    graph_ids = {e.node.id for e in surface.graph}
    for cluster in surface.clusters:
        assert cluster.node_ids, "cluster must reference at least one surface node"
        for nid in cluster.node_ids:
            assert nid in graph_ids, f"cluster ref {nid!r} is not a surface node id"
            # Raw neuron ids in this test are 'n1'/'n2'; surface ids are
            # type-prefixed counters and must differ from the raw UUIDs.
            assert nid not in {"n1", "n2"}


# ── Tests: SIGNALS ─────────────────────────────────


async def test_high_salience_recent_fiber_becomes_urgent_signal():
    """High-salience fiber from last 7 days → URGENT signal."""
    now = _now()
    fiber = FakeFiber(
        id="f1",
        salience=0.9,
        summary="Critical bug in auth module",
        created_at=now - timedelta(days=2),
        metadata={"memory_type": "error"},
    )

    storage = _make_storage(fibers=[fiber])

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    urgent = [s for s in surface.signals if s.level == SignalLevel.URGENT]
    assert len(urgent) >= 1
    assert "auth" in urgent[0].text.lower() or "bug" in urgent[0].text.lower()


async def test_todo_fiber_becomes_uncertain_signal():
    """Fiber with memory_type=todo → UNCERTAIN signal."""
    now = _now()
    fiber = FakeFiber(
        id="f1",
        salience=0.3,
        summary="Implement caching layer",
        created_at=now - timedelta(days=20),
        metadata={"memory_type": "todo"},
    )

    storage = _make_storage(fibers=[fiber])

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    uncertain = [s for s in surface.signals if s.level == SignalLevel.UNCERTAIN]
    assert len(uncertain) >= 1


# ── Tests: DEPTH MAP ──────────────────────────────


async def test_depth_map_sufficient_when_well_covered():
    """Node with many edges on surface → SUFFICIENT."""
    now = _now()
    n1 = _neuron_with_id("n1", content="Well-documented concept", created_at=now)

    # Many outgoing synapses on surface (non-temporal = counted as edges)
    synapses = [
        _make_synapse(source_id="n1", target_id=f"t{i}", stype=SynapseType.RELATED_TO)
        for i in range(5)
    ]

    target_neurons = {
        f"t{i}": _neuron_with_id(f"t{i}", content=f"Target {i}", created_at=now) for i in range(5)
    }

    storage = _make_storage(
        neurons=[n1],
        states={"n1": FakeNeuronState(neuron_id="n1", activation_level=0.8)},
        synapses_out={"n1": synapses},
        target_neurons=target_neurons,
    )
    # get_synapses for depth check returns same count (all synapses in brain.db)
    storage.get_synapses = AsyncMock(return_value=synapses)

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    assert len(surface.depth_map) >= 1
    # Should be SUFFICIENT since surface edges (5) >= total synapses * 0.5 (5*0.5=2.5)
    hint = surface.depth_map[0]
    assert hint.level == DepthLevel.SUFFICIENT


async def test_depth_map_needs_deep_when_many_hidden():
    """Node with few surface edges but many brain.db synapses → NEEDS_DEEP."""
    now = _now()
    n1 = _neuron_with_id("n1", content="Complex topic", created_at=now)
    t1 = _neuron_with_id("t1", content="Target one", created_at=now)

    # Only 1 synapse on surface, but 20 in brain.db
    surface_syn = _make_synapse(source_id="n1", target_id="t1", stype=SynapseType.RELATED_TO)
    all_synapses = [
        _make_synapse(source_id="n1", target_id=f"t{i}", stype=SynapseType.RELATED_TO)
        for i in range(20)
    ]

    storage = _make_storage(
        neurons=[n1],
        states={"n1": FakeNeuronState(neuron_id="n1", activation_level=0.8)},
        synapses_out={"n1": [surface_syn]},
        target_neurons={"t1": t1},
    )
    storage.get_synapses = AsyncMock(return_value=all_synapses)

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    assert len(surface.depth_map) >= 1
    hint = surface.depth_map[0]
    assert hint.level == DepthLevel.NEEDS_DEEP


# ── Tests: META ────────────────────────────────────


async def test_meta_coverage_and_staleness():
    """META coverage and staleness computed correctly."""
    now = _now()
    n1 = _neuron_with_id("n1", content="Recent fact", created_at=now)

    storage = _make_storage(
        neurons=[n1],
        states={"n1": FakeNeuronState(neuron_id="n1", activation_level=0.8)},
        stats={"neuron_count": 100, "synapse_count": 50, "fiber_count": 10},
    )
    storage.get_neurons_batch = AsyncMock(return_value={"n1": n1})

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    # 1 node out of 100 neurons = 1% coverage
    assert surface.meta.coverage <= 0.1
    # Recent neuron → low staleness
    assert surface.meta.staleness < 0.5
    assert surface.meta.last_consolidation != ""


# ── Tests: Token Budget ───────────────────────────


async def test_surface_respects_token_budget():
    """Generated surface is trimmed to token budget."""
    now = _now()
    # Create many neurons to generate a large surface
    neurons = [
        _neuron_with_id(f"n{i}", content=f"Concept number {i} with some text", created_at=now)
        for i in range(20)
    ]
    states = {
        f"n{i}": FakeNeuronState(neuron_id=f"n{i}", activation_level=0.5 + i * 0.02)
        for i in range(20)
    }

    storage = _make_storage(
        neurons=neurons,
        states=states,
        stats={"neuron_count": 20, "synapse_count": 0, "fiber_count": 0},
    )

    gen = SurfaceGenerator(storage, max_graph_nodes=20, token_budget=200)
    surface = await gen.generate()

    # Surface should be trimmed
    assert surface.token_estimate() <= 250  # Some tolerance for estimation


# ── Tests: Node ID Generation ──────────────────────


async def test_node_ids_have_type_prefix():
    """Generated node IDs use correct type prefix."""
    now = _now()
    concept = _neuron_with_id("n1", ntype=NeuronType.CONCEPT, content="API design", created_at=now)
    entity = _neuron_with_id("n2", ntype=NeuronType.ENTITY, content="PostgreSQL", created_at=now)

    storage = _make_storage(
        neurons=[concept, entity],
        states={
            "n1": FakeNeuronState(neuron_id="n1", activation_level=0.8),
            "n2": FakeNeuronState(neuron_id="n2", activation_level=0.7),
        },
    )

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    ids = [e.node.id for e in surface.graph]
    # Should have concept prefix 'c' and entity prefix 'e'
    prefixes = {nid[0] for nid in ids}
    assert "c" in prefixes or "e" in prefixes


# ── Tests: Single Neuron Brain ─────────────────────


async def test_single_neuron_brain():
    """Brain with 1 neuron produces minimal valid surface."""
    now = _now()
    n1 = _neuron_with_id("n1", content="Only concept", created_at=now)

    storage = _make_storage(
        neurons=[n1],
        states={"n1": FakeNeuronState(neuron_id="n1", activation_level=0.5)},
        stats={"neuron_count": 1, "synapse_count": 0, "fiber_count": 0},
    )

    gen = SurfaceGenerator(storage, max_graph_nodes=10)
    surface = await gen.generate()

    assert len(surface.graph) == 1
    assert surface.graph[0].node.content == "Only concept"
    assert surface.frontmatter.neurons == 1
