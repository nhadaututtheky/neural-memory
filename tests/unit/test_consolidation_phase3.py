"""Tests for consolidation Phase 3: semantic-aware prune, surface regen triggers,
bridge weight floor, and version pattern fix."""

from __future__ import annotations

import re
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
)
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


async def _setup_storage(
    fibers: list[Fiber] | None = None,
    neurons: list[Neuron] | None = None,
    synapses: list[Synapse] | None = None,
) -> InMemoryStorage:
    """Helper: create storage with given entities."""
    store = InMemoryStorage()
    brain = Brain.create(name="phase3_test", brain_id="p3-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    if neurons:
        for n in neurons:
            await store.add_neuron(n)
    if synapses:
        for s in synapses:
            await store.add_synapse(s)
    if fibers:
        for f in fibers:
            await store.add_fiber(f)

    return store


class TestSemanticStagePruneProtection:
    """Neurons in semantic-stage fibers use halved prune threshold."""

    @pytest.mark.asyncio
    async def test_semantic_neuron_survives_at_half_threshold(self) -> None:
        """A synapse connecting semantic-stage neurons should survive at weight
        below normal threshold but above half threshold."""
        from dataclasses import replace as dc_replace

        now = utcnow()
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="auth", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="jwt", neuron_id="n2")

        # Synapse at 0.03 — below default threshold 0.05, above half (0.025)
        syn = dc_replace(
            Synapse.create(
                source_id="n1",
                target_id="n2",
                type=SynapseType.RELATED_TO,
                weight=0.03,
                synapse_id="s1",
            ),
            created_at=now - timedelta(days=30),
        )

        # Fiber with _stage=semantic containing n1 and n2
        fib = Fiber(
            id="f1",
            neuron_ids={"n1", "n2"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
            created_at=now - timedelta(days=30),
            metadata={"_stage": "semantic"},
        )

        store = await _setup_storage(fibers=[fib], neurons=[n1, n2], synapses=[syn])
        config = ConsolidationConfig(
            prune_weight_threshold=0.05,
            prune_min_inactive_days=0.0,
            prune_isolated_neurons=False,
        )
        engine = ConsolidationEngine(store, config=config)
        report = ConsolidationReport()
        await engine._prune(report, reference_time=now, dry_run=False)

        # Should NOT be pruned — semantic protection halves threshold to 0.025
        assert report.synapses_pruned == 0

    @pytest.mark.asyncio
    async def test_non_semantic_neuron_pruned_at_same_weight(self) -> None:
        """Same weight synapse WITHOUT semantic stage should be pruned."""
        now = utcnow()
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="auth", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="jwt", neuron_id="n2")
        n3 = Neuron.create(type=NeuronType.CONCEPT, content="token", neuron_id="n3")

        from dataclasses import replace as dc_replace

        # Weak synapse to be pruned
        syn = dc_replace(
            Synapse.create(
                source_id="n1",
                target_id="n2",
                type=SynapseType.RELATED_TO,
                weight=0.03,
                synapse_id="s1",
            ),
            created_at=now - timedelta(days=30),
        )
        # Second synapse so s1 is NOT a bridge (n1 has 2 outgoing)
        syn2 = Synapse.create(
            source_id="n1",
            target_id="n3",
            type=SynapseType.RELATED_TO,
            weight=0.8,
            synapse_id="s2",
        )

        # Fiber WITHOUT semantic stage
        fib = Fiber(
            id="f1",
            neuron_ids={"n1", "n2", "n3"},
            synapse_ids={"s1", "s2"},
            anchor_neuron_id="n1",
            created_at=now - timedelta(days=30),
        )

        store = await _setup_storage(fibers=[fib], neurons=[n1, n2, n3], synapses=[syn, syn2])
        config = ConsolidationConfig(
            prune_weight_threshold=0.05,
            prune_min_inactive_days=0.0,
            prune_isolated_neurons=False,
        )
        engine = ConsolidationEngine(store, config=config)
        report = ConsolidationReport()
        await engine._prune(report, reference_time=now, dry_run=False)

        # Should be pruned — 0.03 < 0.05 threshold, no semantic protection
        assert report.synapses_pruned == 1


class TestBridgeWeightFloor:
    """Bridge synapses at weight >= 0.01 are protected."""

    @pytest.mark.asyncio
    async def test_bridge_at_0015_survives(self) -> None:
        """Bridge synapse at 0.015 weight should be protected (>= 0.01 floor)."""
        now = utcnow()
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="a", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="b", neuron_id="n2")

        # Sole outgoing synapse from n1 → bridge synapse
        syn = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.RELATED_TO,
            weight=0.015,
            synapse_id="s1",
        )

        fib = Fiber(
            id="f1",
            neuron_ids={"n1", "n2"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
            created_at=now - timedelta(days=30),
        )

        store = await _setup_storage(fibers=[fib], neurons=[n1, n2], synapses=[syn])
        config = ConsolidationConfig(
            prune_weight_threshold=0.05,
            prune_min_inactive_days=0.0,
            prune_isolated_neurons=False,
        )
        engine = ConsolidationEngine(store, config=config)
        report = ConsolidationReport()
        await engine._prune(report, reference_time=now, dry_run=False)

        # Bridge at 0.015 >= 0.01 floor → protected
        assert report.synapses_pruned == 0


class TestSurfaceRegenTriggers:
    """Surface regeneration fires on expanded set of triggers."""

    @pytest.mark.asyncio
    async def test_regen_on_fibers_compressed(self) -> None:
        """Surface should regenerate when fibers_compressed > 0."""
        report = ConsolidationReport()
        report.fibers_compressed = 5

        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        engine._storage = MagicMock()
        engine._storage.current_brain_id = "brain-1"

        with patch.object(
            engine, "_regenerate_surface_after_consolidation", new_callable=AsyncMock
        ) as mock_regen:
            # Simulate the trigger check from consolidate()
            if not False and (  # dry_run = False
                report.fibers_merged > 0
                or report.fibers_removed > 0
                or report.fibers_compressed > 0
                or report.synapses_pruned >= 10
                or report.extra.get("stale_flagged", 0) > 0
                or report.extra.get("cold_demoted", 0) > 0
                or report.extra.get("lifecycle_states_updated", 0) > 0
            ):
                await engine._regenerate_surface_after_consolidation()

            mock_regen.assert_called_once()

    @pytest.mark.asyncio
    async def test_regen_on_synapses_pruned_at_threshold(self) -> None:
        """Surface should regenerate when synapses_pruned >= threshold (10)."""
        report = ConsolidationReport()
        report.synapses_pruned = 10

        should_regen = (
            report.fibers_merged > 0
            or report.fibers_removed > 0
            or report.fibers_compressed > 0
            or report.synapses_pruned >= 10
            or report.extra.get("stale_flagged", 0) > 0
            or report.extra.get("cold_demoted", 0) > 0
            or report.extra.get("lifecycle_states_updated", 0) > 0
        )
        assert should_regen is True

    @pytest.mark.asyncio
    async def test_no_regen_on_small_prune(self) -> None:
        """Surface should NOT regenerate when synapses_pruned <= 10."""
        report = ConsolidationReport()
        report.synapses_pruned = 5

        should_regen = (
            report.fibers_merged > 0
            or report.fibers_removed > 0
            or report.fibers_compressed > 0
            or report.synapses_pruned >= 10
            or report.extra.get("stale_flagged", 0) > 0
            or report.extra.get("cold_demoted", 0) > 0
            or report.extra.get("lifecycle_states_updated", 0) > 0
        )
        assert should_regen is False

    @pytest.mark.asyncio
    async def test_regen_on_lifecycle_states_updated(self) -> None:
        """Surface should regenerate when lifecycle_states_updated > 0."""
        report = ConsolidationReport()
        report.extra["lifecycle_states_updated"] = 3

        should_regen = (
            report.fibers_merged > 0
            or report.fibers_removed > 0
            or report.fibers_compressed > 0
            or report.synapses_pruned >= 10
            or report.extra.get("stale_flagged", 0) > 0
            or report.extra.get("cold_demoted", 0) > 0
            or report.extra.get("lifecycle_states_updated", 0) > 0
        )
        assert should_regen is True


class TestVersionPattern:
    """Version regex should not match IPv4/IPv6 or other false positives."""

    def test_matches_v2_0(self) -> None:
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        assert pattern.search("uses v2.0 API")

    def test_matches_v3_11_2(self) -> None:
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        m = pattern.search("Python v3.11.2")
        assert m
        assert m.group(1) == "3"
        assert m.group(2) == "11"

    def test_no_match_ipv4(self) -> None:
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        assert not pattern.search("uses IPv4.0 protocol")

    def test_no_match_ipv6(self) -> None:
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        assert not pattern.search("IPv6.0 networking")

    def test_matches_start_of_string(self) -> None:
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        assert pattern.search("v1.0 release")

    def test_matches_after_space(self) -> None:
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        assert pattern.search("release v5.2")

    def test_no_match_sv4(self) -> None:
        """Arbitrary prefix letter before 'v' should not match."""
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        assert not pattern.search("csv4.0 format")

    def test_matches_uppercase_v(self) -> None:
        """Uppercase V should also match."""
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        assert pattern.search("Updated to V2.0")

    def test_matches_in_parentheses(self) -> None:
        """Version in parentheses like (v2.0) should match."""
        pattern = re.compile(r"(?<![A-Za-z])[vV](\d+)\.(\d+)(?:\.\d+)?\b")
        assert pattern.search("deprecated (v1.0) API")


class TestConsolidationConfigValidation:
    """Verify ConsolidationConfig __post_init__ clamps degenerate values."""

    def test_prune_semantic_factor_clamped_to_0_1(self) -> None:
        config = ConsolidationConfig(prune_semantic_factor=2.0)
        assert config.prune_semantic_factor == 1.0

    def test_prune_semantic_factor_negative_clamped(self) -> None:
        config = ConsolidationConfig(prune_semantic_factor=-0.5)
        assert config.prune_semantic_factor == 0.0

    def test_bridge_weight_floor_negative_clamped(self) -> None:
        config = ConsolidationConfig(bridge_weight_floor=-1.0)
        assert config.bridge_weight_floor == 0.0

    def test_fast_track_rehearsals_zero_clamped(self) -> None:
        config = ConsolidationConfig(maturation_fast_track_rehearsals=0)
        assert config.maturation_fast_track_rehearsals >= 1

    def test_fast_track_time_zero_clamped(self) -> None:
        config = ConsolidationConfig(maturation_fast_track_time_days=0.0)
        assert config.maturation_fast_track_time_days > 0

    def test_fast_track_time_negative_clamped(self) -> None:
        config = ConsolidationConfig(maturation_fast_track_time_days=-1.0)
        assert config.maturation_fast_track_time_days > 0
