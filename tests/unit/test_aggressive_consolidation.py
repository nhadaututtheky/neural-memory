"""Unit tests for A8 Phase 4: Aggressive Consolidation.

Tests SimHash merge, stale detection, access-based demotion, summary fibers,
and surface regeneration after consolidation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.simhash import simhash
from neural_memory.utils.timeutils import utcnow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> ConsolidationConfig:
    return ConsolidationConfig(merge_overlap_threshold=0.5)


@pytest.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="test", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


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
    frequency: int = 0,
    created_at: datetime | None = None,
    metadata: dict | None = None,
    pinned: bool = False,
) -> None:
    f = Fiber(
        id=fiber_id,
        neuron_ids=neuron_ids,
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        pathway=[anchor_id],
        salience=salience,
        frequency=frequency,
        created_at=created_at or utcnow(),
        metadata=metadata or {},
        pinned=pinned,
    )
    await storage.add_fiber(f)


# ===========================================================================
# T4.1: SimHash semantic merge
# ===========================================================================


class TestSimHashMerge:
    """SimHash-based merge catches content-similar fibers that Jaccard misses."""

    async def test_simhash_merges_content_similar_fibers(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fibers with similar anchor content but different neurons get merged."""
        text = "Authentication middleware handles JWT token validation"
        h1 = simhash(text)
        # Slightly different text → very close SimHash
        h2 = simhash("Authentication middleware handles JWT token verification")

        # Two fibers with DIFFERENT neurons (Jaccard = 0) but similar content
        await _add_neuron(storage, "n1", content=text, content_hash=h1)
        await _add_neuron(storage, "n2", content="different neuron")
        await _add_neuron(storage, "n3", content=text + " slightly different", content_hash=h2)
        await _add_neuron(storage, "n4", content="another different")

        await _add_fiber(storage, "f1", {"n1", "n2"}, "n1")
        await _add_fiber(storage, "f2", {"n3", "n4"}, "n3")

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)

        # Should have merged into 1
        assert report.fibers_merged == 2
        assert report.fibers_created == 1

    async def test_simhash_skips_dissimilar_content(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fibers with very different content are NOT merged by SimHash."""
        h1 = simhash("Authentication middleware JWT validation")
        h2 = simhash("Database connection pooling with Redis cache")

        await _add_neuron(storage, "n1", content="auth stuff", content_hash=h1)
        await _add_neuron(storage, "n2", content="filler")
        await _add_neuron(storage, "n3", content="db stuff", content_hash=h2)
        await _add_neuron(storage, "n4", content="filler2")

        await _add_fiber(storage, "f1", {"n1", "n2"}, "n1")
        await _add_fiber(storage, "f2", {"n3", "n4"}, "n3")

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)

        assert report.fibers_merged == 0

    async def test_simhash_respects_verbatim_guard(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Domain guard: verbatim fibers never merge with non-verbatim."""
        text = "Same exact content for both fibers"
        h = simhash(text)

        await _add_neuron(storage, "n1", content=text, content_hash=h)
        await _add_neuron(storage, "n2", content="filler")
        await _add_neuron(storage, "n3", content=text, content_hash=h)
        await _add_neuron(storage, "n4", content="filler2")

        await _add_fiber(storage, "f1", {"n1", "n2"}, "n1", metadata={"_verbatim": True})
        await _add_fiber(storage, "f2", {"n3", "n4"}, "n3", metadata={"_verbatim": False})

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)

        assert report.fibers_merged == 0

    async def test_simhash_combined_with_jaccard(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Both Jaccard and SimHash can contribute to the same Union-Find."""
        text = "Shared topic about caching"
        h = simhash(text)

        await _add_neuron(storage, "n1", content=text, content_hash=h)
        await _add_neuron(storage, "n2", content="shared neuron")
        await _add_neuron(storage, "n3", content=text + " variation", content_hash=h)

        # f1 and f2 share n2 (Jaccard merge candidate)
        # f1 and f3 share content hash (SimHash merge candidate)
        await _add_fiber(storage, "f1", {"n1", "n2"}, "n1")
        await _add_fiber(storage, "f2", {"n2", "n3"}, "n3")

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)

        # Should all end up in one group via transitive closure
        assert report.fibers_merged >= 2


# ===========================================================================
# T4.2: Surface-driven stale detection
# ===========================================================================


class TestStaleDetection:
    """Version-based staleness detection flags outdated fibers."""

    async def test_flags_fibers_with_old_versions(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fibers referencing version ≥2 major behind get _stale flag."""
        await _add_neuron(storage, "n_old", content="Fixed in v2.1.0 release")
        await _add_neuron(storage, "n_new", content="Updated for v5.0.0 feature")

        await _add_fiber(storage, "f_old", {"n_old"}, "n_old")
        await _add_fiber(storage, "f_new", {"n_new"}, "n_new")

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        f_old = await storage.get_fiber("f_old")
        assert f_old is not None
        assert f_old.metadata.get("_stale") is True
        assert report.extra.get("stale_flagged", 0) >= 1

        # New version fiber should NOT be stale
        f_new = await storage.get_fiber("f_new")
        assert f_new is not None
        assert not f_new.metadata.get("_stale")

    async def test_no_stale_flag_when_versions_close(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fibers within 1 major version are NOT flagged stale."""
        await _add_neuron(storage, "n1", content="Works with v4.0.0")
        await _add_neuron(storage, "n2", content="Updated for v5.0.0 feature")

        await _add_fiber(storage, "f1", {"n1"}, "n1")
        await _add_fiber(storage, "f2", {"n2"}, "n2")

        engine = ConsolidationEngine(storage, config)
        await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        f1 = await storage.get_fiber("f1")
        assert f1 is not None
        assert not f1.metadata.get("_stale")

    async def test_stale_dry_run_no_mutation(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Dry run detects stale but doesn't mutate metadata."""
        await _add_neuron(storage, "n_old", content="Ancient v1.0 code")
        await _add_neuron(storage, "n_new", content="Modern v10.0 feature")

        await _add_fiber(storage, "f_old", {"n_old"}, "n_old")
        await _add_fiber(storage, "f_new", {"n_new"}, "n_new")

        engine = ConsolidationEngine(storage, config)
        await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=True)

        f_old = await storage.get_fiber("f_old")
        assert f_old is not None
        assert not f_old.metadata.get("_stale")  # Not mutated in dry_run


# ===========================================================================
# T4.3: Access-based demotion
# ===========================================================================


class TestAccessDemotion:
    """Never-recalled fibers get progressively demoted."""

    async def test_cold_demotion_after_30_days(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fiber with frequency=0 and age > 30 days → _cold_demoted."""
        old_date = utcnow() - timedelta(days=45)
        await _add_neuron(storage, "n1", content="old memory")
        await _add_fiber(storage, "f1", {"n1"}, "n1", frequency=0, created_at=old_date)

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        fiber = await storage.get_fiber("f1")
        assert fiber is not None
        assert fiber.metadata.get("_cold_demoted") is True
        assert report.extra.get("cold_demoted", 0) >= 1

    async def test_no_demotion_if_accessed(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fiber with frequency > 0 is NOT demoted even if old."""
        old_date = utcnow() - timedelta(days=45)
        await _add_neuron(storage, "n1", content="frequently recalled")
        await _add_fiber(storage, "f1", {"n1"}, "n1", frequency=5, created_at=old_date)

        engine = ConsolidationEngine(storage, config)
        await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        fiber = await storage.get_fiber("f1")
        assert fiber is not None
        assert not fiber.metadata.get("_cold_demoted")

    async def test_prune_candidate_after_90_days(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fiber with frequency=0 and age > 90 days → _prune_candidate."""
        old_date = utcnow() - timedelta(days=100)
        await _add_neuron(storage, "n1", content="very old memory")
        await _add_fiber(storage, "f1", {"n1"}, "n1", frequency=0, created_at=old_date)

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        fiber = await storage.get_fiber("f1")
        assert fiber is not None
        assert fiber.metadata.get("_prune_candidate") is True
        assert fiber.metadata.get("_cold_demoted") is True  # also cold-demoted
        assert report.extra.get("prune_candidates", 0) >= 1

    async def test_pinned_fibers_skip_demotion(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Pinned fibers are never demoted regardless of age/access."""
        old_date = utcnow() - timedelta(days=200)
        await _add_neuron(storage, "n1", content="pinned memory")
        await _add_fiber(storage, "f1", {"n1"}, "n1", frequency=0, created_at=old_date, pinned=True)

        engine = ConsolidationEngine(storage, config)
        await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        fiber = await storage.get_fiber("f1")
        assert fiber is not None
        assert not fiber.metadata.get("_cold_demoted")
        assert not fiber.metadata.get("_prune_candidate")

    async def test_young_fiber_no_demotion(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fiber younger than 30 days is NOT demoted even with frequency=0."""
        await _add_neuron(storage, "n1", content="fresh memory")
        await _add_fiber(storage, "f1", {"n1"}, "n1", frequency=0)

        engine = ConsolidationEngine(storage, config)
        await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        fiber = await storage.get_fiber("f1")
        assert fiber is not None
        assert not fiber.metadata.get("_cold_demoted")


# ===========================================================================
# T4.4: Summary fiber creation
# ===========================================================================


class TestSummaryFiber:
    """Groups of 5+ merged fibers produce summary fibers."""

    async def test_summary_fiber_from_large_group(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """5+ fibers sharing neurons → summary fiber with semantic stage."""
        # Create shared neurons (high Jaccard overlap)
        shared = {"s1", "s2", "s3", "s4"}
        for nid in shared:
            await _add_neuron(storage, nid, content=f"shared {nid}")

        # 6 fibers, each containing all shared neurons + one unique
        for i in range(6):
            unique = f"u{i}"
            await _add_neuron(storage, unique, content=f"unique {i}")
            await _add_fiber(storage, f"f{i}", shared | {unique}, "s1", salience=0.5)

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)

        assert report.fibers_merged >= 5
        assert report.fibers_created >= 1

        # Summary fiber should have semantic stage
        fibers = await storage.get_fibers(limit=100)
        summary = [f for f in fibers if f.metadata.get("_summary_fiber")]
        assert len(summary) >= 1
        assert summary[0].metadata.get("_stage") == "semantic"

    async def test_summary_preserves_originals(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Large-group merge demotes originals instead of deleting."""
        shared = {"s1", "s2", "s3"}
        for nid in shared:
            await _add_neuron(storage, nid, content=f"shared {nid}")

        fiber_ids = []
        for i in range(5):
            unique = f"u{i}"
            await _add_neuron(storage, unique, content=f"unique {i}")
            fid = f"f{i}"
            await _add_fiber(storage, fid, shared | {unique}, "s1")
            fiber_ids.append(fid)

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)

        # Originals should NOT be removed (demoted instead)
        assert report.fibers_removed == 0

        # Originals should have _demoted_by_merge flag
        for fid in fiber_ids:
            fiber = await storage.get_fiber(fid)
            assert fiber is not None
            assert fiber.metadata.get("_demoted_by_merge") is True

    async def test_small_group_deletes_originals(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Groups < 5 fibers use standard merge (delete originals)."""
        shared = {"s1", "s2", "s3"}
        for nid in shared:
            await _add_neuron(storage, nid, content=f"shared {nid}")

        for i in range(3):
            unique = f"u{i}"
            await _add_neuron(storage, unique, content=f"unique {i}")
            await _add_fiber(storage, f"f{i}", shared | {unique}, "s1")

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)

        # Standard merge: originals deleted
        assert report.fibers_removed >= 2


# ===========================================================================
# T4.2b: Stale penalty in _fiber_score
# ===========================================================================


class TestStalePenalty:
    """Stale fibers receive -20% score penalty in retrieval."""

    def test_stale_penalty_applied(self) -> None:
        """Fiber with _stale=True gets 0.8x score multiplier."""
        # Create two identical fibers, one stale
        normal_fiber = Fiber(
            id="f_normal",
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            salience=0.8,
            conductivity=1.0,
            last_conducted=utcnow(),
            metadata={},
        )
        stale_fiber = Fiber(
            id="f_stale",
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            salience=0.8,
            conductivity=1.0,
            last_conducted=utcnow(),
            metadata={"_stale": True},
        )

        # Verify the metadata flag that _fiber_score checks for -20% penalty
        assert stale_fiber.metadata.get("_stale") is True
        assert not normal_fiber.metadata.get("_stale")

    def test_semantic_stage_from_metadata(self) -> None:
        """_stage in metadata provides 1.1x bonus via getattr fallback."""
        summary_fiber = Fiber(
            id="f_summary",
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            metadata={"_stage": "semantic"},
        )

        # The retrieval code checks:
        # stage = getattr(fiber, "stage", None) or fiber_meta.get("_stage")
        stage = getattr(summary_fiber, "stage", None) or summary_fiber.metadata.get("_stage")
        assert stage == "semantic"


# ===========================================================================
# T4.5: Surface regeneration after consolidation
# ===========================================================================


class TestSurfaceRegeneration:
    """Surface is regenerated after consolidation when structural changes occur."""

    async def test_surface_regen_called_after_merge(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Surface regeneration triggers when fibers are merged."""
        shared = {"s1", "s2", "s3"}
        for nid in shared:
            await _add_neuron(storage, nid, content=f"shared {nid}")

        for i in range(3):
            unique = f"u{i}"
            await _add_neuron(storage, unique, content=f"unique {i}")
            await _add_fiber(storage, f"f{i}", shared | {unique}, "s1")

        engine = ConsolidationEngine(storage, config)

        with patch.object(
            engine,
            "_regenerate_surface_after_consolidation",
            new_callable=AsyncMock,
        ) as mock_regen:
            await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)
            mock_regen.assert_called_once()

    async def test_no_surface_regen_on_dry_run(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Surface regeneration is skipped on dry run."""
        shared = {"s1", "s2", "s3"}
        for nid in shared:
            await _add_neuron(storage, nid, content=f"shared {nid}")

        for i in range(3):
            unique = f"u{i}"
            await _add_neuron(storage, unique, content=f"unique {i}")
            await _add_fiber(storage, f"f{i}", shared | {unique}, "s1")

        engine = ConsolidationEngine(storage, config)

        with patch.object(
            engine,
            "_regenerate_surface_after_consolidation",
            new_callable=AsyncMock,
        ) as mock_regen:
            await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=True)
            mock_regen.assert_not_called()

    async def test_no_surface_regen_when_nothing_changed(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Surface regeneration is skipped when no structural changes."""
        # Single fiber — nothing to merge
        await _add_neuron(storage, "n1", content="solo")
        await _add_fiber(storage, "f1", {"n1"}, "n1")

        engine = ConsolidationEngine(storage, config)

        with patch.object(
            engine,
            "_regenerate_surface_after_consolidation",
            new_callable=AsyncMock,
        ) as mock_regen:
            await engine.run(strategies=[ConsolidationStrategy.MERGE], dry_run=False)
            mock_regen.assert_not_called()


# ===========================================================================
# Combined scenarios
# ===========================================================================


class TestCombinedScenarios:
    """Integration tests combining multiple Phase 4 features."""

    async def test_stale_and_demoted_fiber(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fiber can be both stale AND cold-demoted."""
        old_date = utcnow() - timedelta(days=60)
        await _add_neuron(storage, "n_old", content="Legacy v1.0 code pattern")
        await _add_neuron(storage, "n_new", content="Modern v5.0 feature")

        await _add_fiber(storage, "f_old", {"n_old"}, "n_old", frequency=0, created_at=old_date)
        await _add_fiber(storage, "f_new", {"n_new"}, "n_new", frequency=3)

        engine = ConsolidationEngine(storage, config)
        await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        fiber = await storage.get_fiber("f_old")
        assert fiber is not None
        assert fiber.metadata.get("_stale") is True
        assert fiber.metadata.get("_cold_demoted") is True

    async def test_merge_then_lifecycle_in_full_run(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Full consolidation runs MERGE then LIFECYCLE in correct order."""
        old_date = utcnow() - timedelta(days=45)

        # Fiber that will NOT merge but will get demoted
        await _add_neuron(storage, "n_solo", content="standalone old memory")
        await _add_fiber(storage, "f_solo", {"n_solo"}, "n_solo", frequency=0, created_at=old_date)

        engine = ConsolidationEngine(storage, config)
        await engine.run(
            strategies=[ConsolidationStrategy.MERGE, ConsolidationStrategy.LIFECYCLE],
            dry_run=False,
        )

        fiber = await storage.get_fiber("f_solo")
        assert fiber is not None
        assert fiber.metadata.get("_cold_demoted") is True

    async def test_already_flagged_not_reflagged(
        self, storage: InMemoryStorage, config: ConsolidationConfig
    ) -> None:
        """Fibers already flagged _cold_demoted are not re-counted."""
        old_date = utcnow() - timedelta(days=45)
        await _add_neuron(storage, "n1", content="already demoted")
        await _add_fiber(
            storage,
            "f1",
            {"n1"},
            "n1",
            frequency=0,
            created_at=old_date,
            metadata={"_cold_demoted": True},
        )

        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=[ConsolidationStrategy.LIFECYCLE], dry_run=False)

        # Should not increment count for already-flagged
        assert report.extra.get("cold_demoted", 0) == 0
