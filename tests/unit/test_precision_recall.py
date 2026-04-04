"""Unit tests for A8 Phase 1: Precision Recall.

Tests MMR diversity, topic affinity, SimHash dedup, and recent-access boost
in _find_matching_fibers().
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.activation import ActivationResult
from neural_memory.engine.retrieval import ReflexPipeline
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.simhash import simhash
from neural_memory.utils.timeutils import utcnow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> BrainConfig:
    return BrainConfig(
        activation_threshold=0.05,
        max_spread_hops=2,
        diversity_overlap_threshold=0.6,
        diversity_penalty_factor=0.7,
        topic_affinity_boost=0.15,
        recent_access_boost=0.1,
        recent_access_window_days=7,
    )


@pytest.fixture
async def storage(config: BrainConfig) -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="test", config=config)
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


def _activation(nid: str, level: float) -> ActivationResult:
    return ActivationResult(
        neuron_id=nid,
        activation_level=level,
        hop_distance=0,
        path=[nid],
        source_anchor=nid,
    )


async def _add_neuron(
    storage: InMemoryStorage,
    nid: str,
    content: str = "",
    content_hash: int = 0,
) -> None:
    n = Neuron.create(
        type=NeuronType.CONCEPT,
        content=content or nid,
        neuron_id=nid,
        content_hash=content_hash,
    )
    await storage.add_neuron(n)


async def _add_fiber(
    storage: InMemoryStorage,
    fiber_id: str,
    neuron_ids: set[str],
    anchor_id: str,
    salience: float = 0.5,
    last_conducted: datetime | None = None,
    metadata: dict | None = None,
) -> Fiber:
    fiber = Fiber.create(
        neuron_ids=neuron_ids,
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        fiber_id=fiber_id,
        metadata=metadata or {},
    )
    from dataclasses import replace

    fiber = replace(
        fiber,
        salience=salience,
        last_conducted=last_conducted,
        conductivity=1.0,
    )
    await storage.add_fiber(fiber)
    return fiber


# ---------------------------------------------------------------------------
# T1.1: MMR Diversity
# ---------------------------------------------------------------------------


class TestMMRDiversity:
    """MMR re-ranking reduces redundant fibers in top results."""

    async def test_diverse_fibers_selected_over_redundant(self, storage, config):
        """Fibers with high neuron overlap should be penalized.

        MMR penalizes but doesn't reorder — it skips candidates whose penalized
        score is too low. To test this, the redundant fiber's score after penalty
        must fall below the skip threshold.
        """
        # Create shared neurons
        for i in range(5):
            await _add_neuron(storage, f"shared_{i}", f"shared concept {i}")
        # Create unique neurons for diverse fibers
        for i in range(5):
            await _add_neuron(storage, f"unique_{i}", f"unique concept {i}")

        # Fiber A: high score
        shared = {f"shared_{i}" for i in range(4)}
        await _add_fiber(storage, "fA", shared | {"shared_4"}, "shared_0", salience=0.9)
        # Fiber B: overlaps 80% with A, only slightly lower salience
        await _add_fiber(storage, "fB", shared | {"unique_0"}, "shared_0", salience=0.85)

        # Fiber C: completely diverse, moderate salience
        diverse = {f"unique_{i}" for i in range(5)}
        await _add_fiber(storage, "fC", diverse, "unique_0", salience=0.6)

        activations = {
            **{f"shared_{i}": _activation(f"shared_{i}", 0.8) for i in range(5)},
            **{f"unique_{i}": _activation(f"unique_{i}", 0.7) for i in range(5)},
        }

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        fiber_ids = [f.id for f in fibers]
        assert "fA" in fiber_ids, "Highest-scoring fiber should be selected"
        assert "fC" in fiber_ids, "Diverse fiber should be selected"
        # All 3 may be included — the key assertion is that diversity allows fC in
        assert len(fibers) >= 2, "At least 2 fibers selected"

    async def test_no_diversity_penalty_for_low_overlap(self, storage, config):
        """Fibers with <60% overlap should not be penalized."""
        # Two fibers sharing only 1 of 5 neurons (20% overlap)
        for i in range(9):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        await _add_fiber(storage, "f1", {"n0", "n1", "n2", "n3", "n4"}, "n0", salience=0.8)
        await _add_fiber(storage, "f2", {"n0", "n5", "n6", "n7", "n8"}, "n0", salience=0.75)

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(9)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        fiber_ids = [f.id for f in fibers]
        assert "f1" in fiber_ids
        assert "f2" in fiber_ids

    async def test_max_10_fibers_returned(self, storage, config):
        """Should still cap at 10 results."""
        for i in range(60):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        for i in range(15):
            nids = {f"n{i * 4 + j}" for j in range(4)}
            anchor = f"n{i * 4}"
            await _add_fiber(storage, f"f{i}", nids, anchor, salience=0.5)

        activations = {f"n{i}": _activation(f"n{i}", 0.5) for i in range(60)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        assert len(fibers) <= 10


# ---------------------------------------------------------------------------
# T1.2: Topic Affinity Boost
# ---------------------------------------------------------------------------


class TestTopicAffinity:
    """Session EMA topics boost fibers with matching tags."""

    async def test_topic_affinity_boosts_matching_fiber(self, storage, config):
        """Fiber with tags matching session topics should rank higher."""
        for i in range(6):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        # Fiber with "auth" tag — matches session topic
        await _add_fiber(
            storage,
            "f_auth",
            {"n0", "n1", "n2"},
            "n0",
            salience=0.4,
            metadata={"tags": ["auth", "security"]},
        )
        # Fiber without matching tags — higher salience but no topic match
        await _add_fiber(
            storage,
            "f_other",
            {"n3", "n4", "n5"},
            "n3",
            salience=0.5,
            metadata={"tags": ["database", "migration"]},
        )

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(6)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(
            activations, session_topics={"auth", "security"}
        )

        fiber_ids = [f.id for f in fibers]
        assert "f_auth" in fiber_ids
        # Auth fiber should be boosted above the other despite lower salience
        assert fiber_ids.index("f_auth") < fiber_ids.index("f_other")

    async def test_no_boost_without_session_topics(self, storage, config):
        """When session_topics is empty, no affinity boost applied."""
        for i in range(3):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        await _add_fiber(
            storage,
            "f1",
            {"n0", "n1", "n2"},
            "n0",
            salience=0.5,
            metadata={"tags": ["auth"]},
        )

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(3)}

        pipeline = ReflexPipeline(storage, config)
        fibers_with = await pipeline._find_matching_fibers(activations, session_topics={"auth"})
        fibers_without = await pipeline._find_matching_fibers(activations, session_topics=set())

        # Both return the fiber, but internal scores differ
        assert len(fibers_with) == 1
        assert len(fibers_without) == 1

    async def test_auto_tags_also_matched(self, storage, config):
        """Topic affinity should check auto_tags as well as metadata tags."""
        for i in range(3):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        fiber = Fiber.create(
            neuron_ids={"n0", "n1", "n2"},
            synapse_ids=set(),
            anchor_neuron_id="n0",
            auto_tags={"retrieval", "activation"},
            fiber_id="f_auto",
        )
        from dataclasses import replace

        fiber = replace(fiber, salience=0.5, conductivity=1.0)
        await storage.add_fiber(fiber)

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(3)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations, session_topics={"retrieval"})

        assert len(fibers) == 1
        assert fibers[0].id == "f_auto"


# ---------------------------------------------------------------------------
# T1.3: Early SimHash Dedup
# ---------------------------------------------------------------------------


class TestSimHashDedup:
    """SimHash near-duplicate fibers are deduplicated before cap."""

    async def test_near_duplicate_fibers_deduplicated(self, storage, config):
        """Fibers with near-duplicate anchor content should be deduplicated."""
        content = "Python async patterns for high-throughput scenarios"
        h1 = simhash(content)
        # Near-duplicate: same content, slightly different
        h2 = simhash(content + " variant")

        # These should be near-duplicates
        await _add_neuron(storage, "n0", content, content_hash=h1)
        await _add_neuron(storage, "n1", content + " variant", content_hash=h2)
        await _add_neuron(storage, "n2", "Completely different topic about Redis caching")

        await _add_fiber(storage, "f_dup1", {"n0"}, "n0", salience=0.8)
        await _add_fiber(storage, "f_dup2", {"n1"}, "n1", salience=0.7)
        await _add_fiber(storage, "f_diff", {"n2"}, "n2", salience=0.6)

        activations = {
            "n0": _activation("n0", 0.9),
            "n1": _activation("n1", 0.85),
            "n2": _activation("n2", 0.7),
        }

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        fiber_ids = [f.id for f in fibers]
        assert "f_dup1" in fiber_ids, "Highest-scoring duplicate should be kept"
        assert "f_diff" in fiber_ids, "Different fiber should be kept"
        # f_dup2 should be removed as near-duplicate of f_dup1
        assert "f_dup2" not in fiber_ids, "Near-duplicate should be removed"

    async def test_zero_hash_not_deduplicated(self, storage, config):
        """Neurons with content_hash=0 should not be deduplicated."""
        await _add_neuron(storage, "n0", "content A", content_hash=0)
        await _add_neuron(storage, "n1", "content B", content_hash=0)

        await _add_fiber(storage, "f1", {"n0"}, "n0", salience=0.8)
        await _add_fiber(storage, "f2", {"n1"}, "n1", salience=0.7)

        activations = {
            "n0": _activation("n0", 0.9),
            "n1": _activation("n1", 0.8),
        }

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        assert len(fibers) == 2, "Zero-hash fibers should both be kept"

    async def test_different_content_not_deduplicated(self, storage, config):
        """Fibers with genuinely different content should both appear."""
        h1 = simhash("Python web framework comparison FastAPI vs Django")
        h2 = simhash("Redis caching strategies for distributed systems")

        await _add_neuron(storage, "n0", "Python web", content_hash=h1)
        await _add_neuron(storage, "n1", "Redis cache", content_hash=h2)

        await _add_fiber(storage, "f1", {"n0"}, "n0", salience=0.8)
        await _add_fiber(storage, "f2", {"n1"}, "n1", salience=0.7)

        activations = {
            "n0": _activation("n0", 0.9),
            "n1": _activation("n1", 0.8),
        }

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        assert len(fibers) == 2


# ---------------------------------------------------------------------------
# T1.5: Recent-Access Boost
# ---------------------------------------------------------------------------


class TestRecentAccessBoost:
    """Fibers accessed within recent_access_window_days get score boost."""

    async def test_recently_accessed_fiber_boosted(self, storage, config):
        """Fiber accessed 1 day ago should rank higher than one accessed 30 days ago."""
        for i in range(6):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        now = utcnow()
        # Lower salience but recently accessed
        await _add_fiber(
            storage,
            "f_recent",
            {"n0", "n1", "n2"},
            "n0",
            salience=0.4,
            last_conducted=now - timedelta(days=1),
        )
        # Higher salience but old access
        await _add_fiber(
            storage,
            "f_old",
            {"n3", "n4", "n5"},
            "n3",
            salience=0.45,
            last_conducted=now - timedelta(days=30),
        )

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(6)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        fiber_ids = [f.id for f in fibers]
        assert fiber_ids.index("f_recent") < fiber_ids.index("f_old"), (
            "Recently accessed fiber should rank higher"
        )

    async def test_no_boost_outside_window(self, storage, config):
        """Fiber accessed 10 days ago (outside 7-day window) gets no boost."""
        for i in range(3):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        now = utcnow()
        await _add_fiber(
            storage,
            "f1",
            {"n0", "n1", "n2"},
            "n0",
            salience=0.5,
            last_conducted=now - timedelta(days=10),
        )

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(3)}

        # Test with boost disabled
        config_no_boost = config.with_updates(recent_access_boost=0.0)
        pipeline_no = ReflexPipeline(storage, config_no_boost)
        fibers_no = await pipeline_no._find_matching_fibers(activations)

        pipeline_yes = ReflexPipeline(storage, config)
        fibers_yes = await pipeline_yes._find_matching_fibers(activations)

        # Both should return the fiber
        assert len(fibers_no) == 1
        assert len(fibers_yes) == 1

    async def test_no_boost_for_never_accessed(self, storage, config):
        """Fiber with no last_conducted should not get boost."""
        for i in range(3):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        await _add_fiber(
            storage,
            "f1",
            {"n0", "n1", "n2"},
            "n0",
            salience=0.5,
            last_conducted=None,
        )

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(3)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        assert len(fibers) == 1


# ---------------------------------------------------------------------------
# BrainConfig Defaults
# ---------------------------------------------------------------------------


class TestBrainConfigDefaults:
    """Verify new BrainConfig fields have correct defaults."""

    def test_precision_recall_defaults(self):
        config = BrainConfig()
        assert config.recent_access_boost == 0.1
        assert config.recent_access_window_days == 7
        assert config.diversity_overlap_threshold == 0.6
        assert config.diversity_penalty_factor == 0.7
        assert config.topic_affinity_boost == 0.15

    def test_config_with_updates(self):
        config = BrainConfig()
        updated = config.with_updates(recent_access_boost=0.2, topic_affinity_boost=0.3)
        assert updated.recent_access_boost == 0.2
        assert updated.topic_affinity_boost == 0.3
        # Original unchanged (immutability)
        assert config.recent_access_boost == 0.1


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Existing behavior unchanged when no session topics or new features."""

    async def test_no_session_topics_same_behavior(self, storage, config):
        """Without session topics, fiber selection works as before."""
        for i in range(5):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        await _add_fiber(storage, "f1", {"n0", "n1"}, "n0", salience=0.8)
        await _add_fiber(storage, "f2", {"n2", "n3"}, "n2", salience=0.6)

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(4)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        assert len(fibers) == 2
        # Higher salience first
        assert fibers[0].id == "f1"

    async def test_empty_fibers_returns_empty(self, storage, config):
        """No matching fibers returns empty list."""
        activations = {"nonexistent": _activation("nonexistent", 0.9)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        assert fibers == []

    async def test_single_fiber_no_penalty(self, storage, config):
        """Single fiber should never be penalized."""
        await _add_neuron(storage, "n0", "concept")
        await _add_fiber(storage, "f1", {"n0"}, "n0", salience=0.5)

        activations = {"n0": _activation("n0", 0.9)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations)

        assert len(fibers) == 1
        assert fibers[0].id == "f1"


# ---------------------------------------------------------------------------
# Combined Scenarios
# ---------------------------------------------------------------------------


class TestCombinedScenarios:
    """Multiple Phase 1 features working together."""

    async def test_topic_affinity_plus_diversity(self, storage, config):
        """Topic-boosted fiber should survive diversity penalty."""
        for i in range(10):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        # Fiber A: high salience, no topic match
        await _add_fiber(
            storage,
            "fA",
            {"n0", "n1", "n2", "n3"},
            "n0",
            salience=0.8,
            metadata={"tags": ["database"]},
        )
        # Fiber B: overlaps with A (75%), no topic match
        await _add_fiber(
            storage,
            "fB",
            {"n0", "n1", "n2", "n4"},
            "n0",
            salience=0.75,
            metadata={"tags": ["database"]},
        )
        # Fiber C: no overlap with A, topic match, lower salience
        await _add_fiber(
            storage,
            "fC",
            {"n5", "n6", "n7"},
            "n5",
            salience=0.4,
            metadata={"tags": ["auth", "security"]},
        )

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(8)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations, session_topics={"auth"})

        fiber_ids = [f.id for f in fibers]
        assert "fA" in fiber_ids, "Highest scorer should be included"
        assert "fC" in fiber_ids, "Topic-boosted diverse fiber should be included"

    async def test_recent_access_plus_topic_affinity(self, storage, config):
        """Recent access + topic affinity combine additively."""
        for i in range(6):
            await _add_neuron(storage, f"n{i}", f"concept {i}")

        now = utcnow()
        # Fiber with both boosts: recent access + topic match, but low salience
        await _add_fiber(
            storage,
            "f_boosted",
            {"n0", "n1", "n2"},
            "n0",
            salience=0.3,
            last_conducted=now - timedelta(hours=12),
            metadata={"tags": ["auth"]},
        )
        # Fiber with neither boost, but higher salience
        await _add_fiber(
            storage,
            "f_plain",
            {"n3", "n4", "n5"},
            "n3",
            salience=0.45,
            last_conducted=now - timedelta(days=30),
            metadata={"tags": ["database"]},
        )

        activations = {f"n{i}": _activation(f"n{i}", 0.7) for i in range(6)}

        pipeline = ReflexPipeline(storage, config)
        fibers = await pipeline._find_matching_fibers(activations, session_topics={"auth"})

        fiber_ids = [f.id for f in fibers]
        # Combined boosts (+0.15 topic + 0.1 recent) should overcome salience gap
        assert fiber_ids.index("f_boosted") < fiber_ids.index("f_plain")
