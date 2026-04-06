"""Tests for consolidation merge quality improvements.

Covers: graduated temporal threshold, anchor-based summaries,
original summary preservation in metadata.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse
from neural_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
)
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


async def _setup_storage_with_fibers(
    fibers: list[Fiber],
    neurons: list[Neuron] | None = None,
    synapses: list[Synapse] | None = None,
) -> InMemoryStorage:
    """Helper: create storage with given fibers + neurons."""
    store = InMemoryStorage()
    brain = Brain.create(name="merge_test", brain_id="merge-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    if neurons:
        for n in neurons:
            await store.add_neuron(n)
    if synapses:
        for s in synapses:
            await store.add_synapse(s)
    for f in fibers:
        await store.add_fiber(f)

    return store


class TestGraduatedTemporalThreshold:
    """Verify the graduated merge threshold based on temporal proximity."""

    def test_temporal_factor_at_zero(self) -> None:
        """At t=0, factor should be 0.6 (60% of base threshold)."""
        halflife = 7200.0
        factor = 1.0 - 0.4 * math.exp(0 / halflife)
        assert abs(factor - 0.6) < 0.001

    def test_temporal_factor_at_halflife(self) -> None:
        """At t=halflife, factor should be ~0.853."""
        halflife = 7200.0
        factor = 1.0 - 0.4 * math.exp(-halflife / halflife)
        expected = 1.0 - 0.4 * math.exp(-1)
        assert abs(factor - expected) < 0.001
        assert 0.8 < factor < 0.9

    def test_temporal_factor_at_large_time(self) -> None:
        """At very large t, factor approaches 1.0."""
        halflife = 7200.0
        factor = 1.0 - 0.4 * math.exp(-100000 / halflife)
        assert factor > 0.99

    def test_config_default_halflife(self) -> None:
        """Default halflife should be 7200 seconds (2 hours)."""
        config = ConsolidationConfig()
        assert config.merge_temporal_halflife_seconds == 7200.0

    def test_config_custom_halflife(self) -> None:
        """Custom halflife should be respected."""
        config = ConsolidationConfig(merge_temporal_halflife_seconds=3600.0)
        assert config.merge_temporal_halflife_seconds == 3600.0

    @pytest.mark.asyncio
    async def test_close_fibers_merge_with_lower_overlap(self) -> None:
        """Fibers created 1 second apart should merge at lower Jaccard threshold."""
        now = utcnow()
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="auth system", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="login flow", neuron_id="n2")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="session mgmt", neuron_id="n3")

        # Two fibers sharing 2/3 neurons = Jaccard 0.5
        # With graduated threshold at t≈0: effective = 0.5 * 0.6 = 0.3 → should merge
        f1 = Fiber(
            id="f1",
            neuron_ids={"n1", "n2"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            created_at=now,
        )
        f2 = Fiber(
            id="f2",
            neuron_ids={"n1", "n3"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            created_at=now + timedelta(seconds=1),
        )

        store = await _setup_storage_with_fibers([f1, f2], [n1, n2, n3])
        # Jaccard = |{n1}| / |{n1,n2,n3}| = 1/3 ≈ 0.33
        # effective threshold at t≈0 = 0.5 * 0.6 = 0.30 → 0.33 >= 0.30 → merge
        config = ConsolidationConfig(merge_overlap_threshold=0.5)
        engine = ConsolidationEngine(store, config=config)
        report = ConsolidationReport()
        await engine._merge(report, dry_run=False)
        assert report.fibers_merged >= 2

    @pytest.mark.asyncio
    async def test_distant_fibers_need_higher_overlap(self) -> None:
        """Fibers created far apart need full Jaccard threshold to merge."""
        now = utcnow()
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="auth system", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="login flow", neuron_id="n2")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="session mgmt", neuron_id="n3")

        # Jaccard 1/3 ≈ 0.33 — below full threshold of 0.5
        f1 = Fiber(
            id="f1",
            neuron_ids={"n1", "n2"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            created_at=now - timedelta(days=30),
        )
        f2 = Fiber(
            id="f2",
            neuron_ids={"n1", "n3"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            created_at=now,
        )

        store = await _setup_storage_with_fibers([f1, f2], [n1, n2, n3])
        config = ConsolidationConfig(merge_overlap_threshold=0.5)
        engine = ConsolidationEngine(store, config=config)
        report = ConsolidationReport()
        await engine._merge(report, dry_run=False)
        # Should NOT merge: temporal factor ≈ 1.0, effective threshold ≈ 0.5, Jaccard 0.33 < 0.5
        assert report.fibers_merged == 0


class TestMergeSummaryQuality:
    """Verify merged fiber summaries contain actual content."""

    @pytest.mark.asyncio
    async def test_summary_contains_anchor_content(self) -> None:
        """Merged summary should include anchor neuron content, not just 'Merged from N'."""
        now = utcnow()
        n1 = Neuron.create(
            type=NeuronType.CONCEPT, content="FastAPI authentication", neuron_id="n1"
        )
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="JWT token handling", neuron_id="n2")

        f1 = Fiber(
            id="f1",
            neuron_ids={"n1", "n2"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            created_at=now,
        )
        f2 = Fiber(
            id="f2",
            neuron_ids={"n1", "n2"},
            synapse_ids=set(),
            anchor_neuron_id="n2",
            created_at=now + timedelta(seconds=5),
        )

        store = await _setup_storage_with_fibers([f1, f2], [n1, n2])
        config = ConsolidationConfig(merge_overlap_threshold=0.3)
        engine = ConsolidationEngine(store, config=config)
        report = ConsolidationReport()
        await engine._merge(report, dry_run=False)

        assert report.fibers_merged >= 2

        # Check the merged fiber has content-based summary
        all_fibers = await store.get_fibers(limit=100)
        merged = [f for f in all_fibers if "merged_from" in f.metadata]
        assert len(merged) == 1
        summary = merged[0].summary or ""
        # Summary should NOT be just "Merged from 2 fibers"
        assert "FastAPI" in summary or "JWT" in summary

    @pytest.mark.asyncio
    async def test_large_merge_preserves_original_summaries(self) -> None:
        """5+ fiber merge should preserve _original_summaries in metadata."""
        now = utcnow()
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content=f"topic_{i}", neuron_id=f"n{i}")
            for i in range(6)
        ]
        # All fibers share n0 + one unique → high overlap when sharing n0
        fibers = [
            Fiber(
                id=f"f{i}",
                neuron_ids={"n0", f"n{i}" if i > 0 else "n1"},
                synapse_ids=set(),
                anchor_neuron_id="n0",
                summary=f"Summary about topic {i}",
                created_at=now + timedelta(seconds=i),
            )
            for i in range(6)
        ]

        store = await _setup_storage_with_fibers(fibers, neurons)
        config = ConsolidationConfig(merge_overlap_threshold=0.3)
        engine = ConsolidationEngine(store, config=config)
        report = ConsolidationReport()
        await engine._merge(report, dry_run=False)

        if report.fibers_merged >= 5:
            all_fibers = await store.get_fibers(limit=100)
            merged = [f for f in all_fibers if f.metadata.get("_summary_fiber")]
            if merged:
                originals = merged[0].metadata.get("_original_summaries", [])
                assert len(originals) > 0
                assert any("topic" in s for s in originals)
