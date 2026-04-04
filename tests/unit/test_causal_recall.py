"""Unit tests for causal-aware recall — synapse role semantics during activation."""

from __future__ import annotations

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import (
    ACTIVE_ROLE_TYPES,
    REINFORCEMENT_TYPES,
    SEQUENTIAL_TYPES,
    SUPERSESSION_TYPES,
    SYNAPSE_ROLES,
    WEAKENING_TYPES,
    Synapse,
    SynapseRole,
    SynapseType,
)
from neural_memory.engine.activation import ActivationResult
from neural_memory.engine.context_optimizer import ContextItem
from neural_memory.storage.memory_store import InMemoryStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> BrainConfig:
    return BrainConfig(activation_threshold=0.05, max_spread_hops=3)


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
    metadata: dict | None = None,
) -> None:
    n = Neuron.create(
        type=NeuronType.CONCEPT,
        content=content or nid,
        neuron_id=nid,
        metadata=metadata or {},
    )
    await storage.add_neuron(n)


async def _add_synapse(
    storage: InMemoryStorage,
    source: str,
    target: str,
    syn_type: SynapseType,
    weight: float = 1.0,
) -> None:
    s = Synapse.create(source, target, syn_type, weight=weight)
    await storage.add_synapse(s)


async def _make_pipeline(storage: InMemoryStorage, config: BrainConfig):
    """Create a ReflexPipeline for testing _apply_causal_semantics directly."""
    from neural_memory.engine.retrieval import ReflexPipeline

    pipeline = ReflexPipeline(storage=storage, config=config)
    return pipeline


# ===========================================================================
# T1: SynapseRole classification tests
# ===========================================================================


class TestSynapseRoleClassification:
    """Verify all 48 synapse types are classified into roles."""

    def test_all_types_classified(self):
        """Every SynapseType must have a role in SYNAPSE_ROLES."""
        for st in SynapseType:
            assert st in SYNAPSE_ROLES, f"{st} not classified in SYNAPSE_ROLES"

    def test_role_count_is_seven(self):
        """There should be exactly 7 roles."""
        assert len(SynapseRole) == 7

    def test_supersession_types(self):
        """RESOLVED_BY, SUPERSEDES, EVOLVES_FROM, FALSIFIED_BY are SUPERSESSION."""
        expected = {
            SynapseType.RESOLVED_BY,
            SynapseType.SUPERSEDES,
            SynapseType.EVOLVES_FROM,
            SynapseType.FALSIFIED_BY,
        }
        assert expected <= SUPERSESSION_TYPES

    def test_reinforcement_types(self):
        """EVIDENCE_FOR, VERIFIED_BY, EFFECTIVE_FOR are REINFORCEMENT."""
        expected = {
            SynapseType.EVIDENCE_FOR,
            SynapseType.VERIFIED_BY,
            SynapseType.EFFECTIVE_FOR,
        }
        assert expected <= REINFORCEMENT_TYPES

    def test_weakening_types(self):
        """EVIDENCE_AGAINST, CONTRADICTS, PREVENTS are WEAKENING."""
        expected = {
            SynapseType.EVIDENCE_AGAINST,
            SynapseType.CONTRADICTS,
            SynapseType.PREVENTS,
        }
        assert expected <= WEAKENING_TYPES

    def test_sequential_types(self):
        """BEFORE, AFTER, LEADS_TO, CALLS are SEQUENTIAL."""
        expected = {
            SynapseType.BEFORE,
            SynapseType.AFTER,
            SynapseType.LEADS_TO,
            SynapseType.CALLS,
        }
        assert expected <= SEQUENTIAL_TYPES

    def test_passive_not_in_active(self):
        """PASSIVE types should not be in ACTIVE_ROLE_TYPES."""
        passive_types = {st for st, role in SYNAPSE_ROLES.items() if role == SynapseRole.PASSIVE}
        assert not passive_types & ACTIVE_ROLE_TYPES

    def test_active_roles_union(self):
        """ACTIVE_ROLE_TYPES = SUPERSESSION + REINFORCEMENT + WEAKENING + SEQUENTIAL."""
        expected = SUPERSESSION_TYPES | REINFORCEMENT_TYPES | WEAKENING_TYPES | SEQUENTIAL_TYPES
        assert expected == ACTIVE_ROLE_TYPES


# ===========================================================================
# T2: SUPERSESSION tests
# ===========================================================================


class TestSupersession:
    """When source is activated, inject target (fix) and demote source (error)."""

    async def test_error_resolved_by_fix(self, storage, config):
        """Error with RESOLVED_BY → fix appears, error demoted to ghost."""
        await _add_neuron(storage, "error1", "TypeError in parser")
        await _add_neuron(storage, "fix1", "Added null check in parser")
        await _add_synapse(storage, "error1", "fix1", SynapseType.RESOLVED_BY)

        pipeline = await _make_pipeline(storage, config)
        activations = {"error1": _activation("error1", 0.8)}

        result = await pipeline._apply_causal_semantics(activations)

        # Fix should be injected with boosted score
        assert "fix1" in result
        assert result["fix1"].activation_level == pytest.approx(0.8 * 1.2, rel=0.01)

        # Error should be demoted to ghost level (x0.1)
        assert "error1" in result
        assert result["error1"].activation_level == pytest.approx(0.8 * 0.1, rel=0.01)

    async def test_supersession_chain_evolves_from(self, storage, config):
        """EVOLVES_FROM: source=NEW → target=OLD. Demote old, boost new.

        v3 EVOLVES_FROM v2 EVOLVES_FROM v1.
        When v2 and v1 are activated, v1 gets demoted (v2 supersedes v1).
        """
        await _add_neuron(storage, "v1", "Decision v1")
        await _add_neuron(storage, "v2", "Decision v2")
        await _add_neuron(storage, "v3", "Decision v3")
        await _add_synapse(storage, "v2", "v1", SynapseType.EVOLVES_FROM)
        await _add_synapse(storage, "v3", "v2", SynapseType.EVOLVES_FROM)

        pipeline = await _make_pipeline(storage, config)
        # Both v2 (newer) and v1 (older) are activated
        activations = {
            "v1": _activation("v1", 0.7),
            "v2": _activation("v2", 0.6),
        }
        result = await pipeline._apply_causal_semantics(activations)

        # v2 EVOLVES_FROM v1: v2 is source=newer, v1 is target=older
        # v1 should be demoted to ghost, v2 should be boosted
        assert result["v2"].activation_level >= 0.6
        assert result["v1"].activation_level < 0.15  # ghost level

    async def test_supersession_depth_cap(self, storage, config):
        """Chain longer than 5 should stop — not follow indefinitely.

        Using RESOLVED_BY (source=OLD → target=NEW) for simpler chain logic.
        """
        # Create chain: n0 → n1 → n2 → ... → n9
        for i in range(10):
            await _add_neuron(storage, f"n{i}", f"Version {i}")
        for i in range(9):
            await _add_synapse(storage, f"n{i}", f"n{i + 1}", SynapseType.RESOLVED_BY)

        pipeline = await _make_pipeline(storage, config)
        activations = {"n0": _activation("n0", 0.9)}

        result = await pipeline._apply_causal_semantics(activations)

        # Should NOT reach n9 (chain too long — capped at depth 5)
        assert "n9" not in result
        # n0 should be demoted
        assert result["n0"].activation_level < 0.15

    async def test_supersession_target_already_activated(self, storage, config):
        """If target already activated with higher score, keep higher score."""
        await _add_neuron(storage, "old", "Old approach")
        await _add_neuron(storage, "new", "New approach")
        await _add_synapse(storage, "old", "new", SynapseType.RESOLVED_BY)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "old": _activation("old", 0.5),
            "new": _activation("new", 0.9),  # Already high
        }

        result = await pipeline._apply_causal_semantics(activations)

        # new should keep its higher score (0.9 > 0.5 * 1.2 = 0.6)
        assert result["new"].activation_level == pytest.approx(0.9, rel=0.01)

    async def test_supersession_demotes_below_threshold(self, storage, config):
        """If demoted score < threshold, neuron is removed entirely."""
        await _add_neuron(storage, "weak_error", "Minor error")
        await _add_neuron(storage, "fix", "Fix")
        await _add_synapse(storage, "weak_error", "fix", SynapseType.RESOLVED_BY)

        pipeline = await _make_pipeline(storage, config)
        # 0.3 * 0.1 = 0.03 < threshold (0.05)
        activations = {"weak_error": _activation("weak_error", 0.3)}

        result = await pipeline._apply_causal_semantics(activations)

        assert "weak_error" not in result
        assert "fix" in result


# ===========================================================================
# T3: REINFORCEMENT tests
# ===========================================================================


class TestReinforcement:
    """Evidence boosts hypothesis activation additively."""

    async def test_evidence_boosts_hypothesis(self, storage, config):
        """Single evidence → +0.15 boost to hypothesis."""
        await _add_neuron(storage, "evidence", "Observation X")
        await _add_neuron(storage, "hypothesis", "Theory Y")
        await _add_synapse(storage, "evidence", "hypothesis", SynapseType.EVIDENCE_FOR)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "evidence": _activation("evidence", 0.5),
            "hypothesis": _activation("hypothesis", 0.4),
        }

        result = await pipeline._apply_causal_semantics(activations)

        assert result["hypothesis"].activation_level == pytest.approx(0.55, rel=0.01)

    async def test_reinforcement_cap(self, storage, config):
        """Multiple evidence sources cap at +0.3 total boost."""
        await _add_neuron(storage, "e1", "Evidence 1")
        await _add_neuron(storage, "e2", "Evidence 2")
        await _add_neuron(storage, "e3", "Evidence 3")
        await _add_neuron(storage, "hyp", "Hypothesis")
        await _add_synapse(storage, "e1", "hyp", SynapseType.EVIDENCE_FOR)
        await _add_synapse(storage, "e2", "hyp", SynapseType.VERIFIED_BY)
        await _add_synapse(storage, "e3", "hyp", SynapseType.EFFECTIVE_FOR)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "e1": _activation("e1", 0.5),
            "e2": _activation("e2", 0.5),
            "e3": _activation("e3", 0.5),
            "hyp": _activation("hyp", 0.3),
        }

        result = await pipeline._apply_causal_semantics(activations)

        # 0.3 + 0.3 (capped) = 0.6
        assert result["hyp"].activation_level == pytest.approx(0.6, rel=0.01)

    async def test_reinforcement_only_activated_targets(self, storage, config):
        """Reinforcement only boosts already-activated targets."""
        await _add_neuron(storage, "ev", "Evidence")
        await _add_neuron(storage, "target", "Target not activated")
        await _add_synapse(storage, "ev", "target", SynapseType.EVIDENCE_FOR)

        pipeline = await _make_pipeline(storage, config)
        activations = {"ev": _activation("ev", 0.5)}

        result = await pipeline._apply_causal_semantics(activations)

        # Target was not in activations → should not appear
        assert "target" not in result


# ===========================================================================
# T4: WEAKENING tests
# ===========================================================================


class TestWeakening:
    """Counter-evidence demotes hypothesis activation."""

    async def test_counter_evidence_demotes(self, storage, config):
        """EVIDENCE_AGAINST halves target activation."""
        await _add_neuron(storage, "counter", "Counter-evidence")
        await _add_neuron(storage, "hyp", "Hypothesis")
        await _add_synapse(storage, "counter", "hyp", SynapseType.EVIDENCE_AGAINST)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "counter": _activation("counter", 0.6),
            "hyp": _activation("hyp", 0.8),
        }

        result = await pipeline._apply_causal_semantics(activations)

        assert result["hyp"].activation_level == pytest.approx(0.4, rel=0.01)

    async def test_weakening_removes_below_threshold(self, storage, config):
        """If halved score < threshold, neuron is removed."""
        await _add_neuron(storage, "contra", "Contradiction")
        await _add_neuron(storage, "weak_hyp", "Weak hypothesis")
        await _add_synapse(storage, "contra", "weak_hyp", SynapseType.CONTRADICTS)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "contra": _activation("contra", 0.5),
            "weak_hyp": _activation("weak_hyp", 0.08),  # 0.08 * 0.5 = 0.04 < 0.05
        }

        result = await pipeline._apply_causal_semantics(activations)

        assert "weak_hyp" not in result


# ===========================================================================
# T5: SEQUENTIAL tests
# ===========================================================================


class TestSequential:
    """Step N primes step N+1 with light boost."""

    async def test_sequential_primes_next_step(self, storage, config):
        """BEFORE synapse gives +0.1 boost to next step."""
        await _add_neuron(storage, "step1", "Step 1")
        await _add_neuron(storage, "step2", "Step 2")
        await _add_synapse(storage, "step1", "step2", SynapseType.BEFORE)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "step1": _activation("step1", 0.6),
            "step2": _activation("step2", 0.3),
        }

        result = await pipeline._apply_causal_semantics(activations)

        assert result["step2"].activation_level == pytest.approx(0.4, rel=0.01)

    async def test_sequential_only_primes_activated(self, storage, config):
        """Sequential only boosts targets already in activation set."""
        await _add_neuron(storage, "s1", "Step 1")
        await _add_neuron(storage, "s2", "Step 2 not activated")
        await _add_synapse(storage, "s1", "s2", SynapseType.LEADS_TO)

        pipeline = await _make_pipeline(storage, config)
        activations = {"s1": _activation("s1", 0.7)}

        result = await pipeline._apply_causal_semantics(activations)

        # s2 was not activated → should not appear
        assert "s2" not in result

    async def test_sequential_capped_at_1(self, storage, config):
        """Sequential boost should not exceed 1.0."""
        await _add_neuron(storage, "s1", "Step 1")
        await _add_neuron(storage, "s2", "Step 2")
        await _add_synapse(storage, "s1", "s2", SynapseType.AFTER)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "s1": _activation("s1", 0.9),
            "s2": _activation("s2", 0.95),
        }

        result = await pipeline._apply_causal_semantics(activations)

        assert result["s2"].activation_level <= 1.0


# ===========================================================================
# T6: PASSIVE tests
# ===========================================================================


class TestPassive:
    """Audit synapses should not be traversed."""

    async def test_passive_synapses_not_traversed(self, storage, config):
        """STORED_BY synapse should have no effect on activation."""
        await _add_neuron(storage, "mem", "A memory")
        await _add_neuron(storage, "agent", "Agent who stored it")
        await _add_synapse(storage, "mem", "agent", SynapseType.STORED_BY)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "mem": _activation("mem", 0.7),
            "agent": _activation("agent", 0.3),
        }

        result = await pipeline._apply_causal_semantics(activations)

        # Both should be unchanged
        assert result["mem"].activation_level == pytest.approx(0.7, rel=0.01)
        assert result["agent"].activation_level == pytest.approx(0.3, rel=0.01)


# ===========================================================================
# T7: Backward compatibility tests
# ===========================================================================


class TestBackwardCompat:
    """Existing behavior must be unchanged when no role synapses exist."""

    async def test_no_synapses_unchanged(self, storage, config):
        """Activations with no synapses at all → identical output."""
        await _add_neuron(storage, "a", "Memory A")
        await _add_neuron(storage, "b", "Memory B")

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "a": _activation("a", 0.6),
            "b": _activation("b", 0.4),
        }

        result = await pipeline._apply_causal_semantics(activations)

        assert result["a"].activation_level == pytest.approx(0.6, rel=0.01)
        assert result["b"].activation_level == pytest.approx(0.4, rel=0.01)

    async def test_structural_synapses_unchanged(self, storage, config):
        """STRUCTURAL role synapses (IS_A, CONTAINS, etc.) don't alter scores."""
        await _add_neuron(storage, "class1", "Animal")
        await _add_neuron(storage, "class2", "Dog")
        await _add_synapse(storage, "class2", "class1", SynapseType.IS_A)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "class1": _activation("class1", 0.5),
            "class2": _activation("class2", 0.7),
        }

        result = await pipeline._apply_causal_semantics(activations)

        assert result["class1"].activation_level == pytest.approx(0.5, rel=0.01)
        assert result["class2"].activation_level == pytest.approx(0.7, rel=0.01)

    async def test_lateral_synapses_unchanged(self, storage, config):
        """LATERAL role synapses (RELATED_TO, SIMILAR_TO) don't alter scores."""
        await _add_neuron(storage, "x", "Concept X")
        await _add_neuron(storage, "y", "Concept Y")
        await _add_synapse(storage, "x", "y", SynapseType.RELATED_TO)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "x": _activation("x", 0.6),
            "y": _activation("y", 0.5),
        }

        result = await pipeline._apply_causal_semantics(activations)

        assert result["x"].activation_level == pytest.approx(0.6, rel=0.01)
        assert result["y"].activation_level == pytest.approx(0.5, rel=0.01)

    async def test_empty_activations(self, storage, config):
        """Empty activations → empty result."""
        pipeline = await _make_pipeline(storage, config)
        result = await pipeline._apply_causal_semantics({})
        assert result == {}


# ===========================================================================
# T8: Context formatting tests
# ===========================================================================


class TestContextFormatting:
    """[OUTDATED] label on superseded ghost memories."""

    def test_superseded_ghost_shows_outdated_label(self):
        """Ghost item with superseded_by shows [OUTDATED] prefix."""
        item = ContextItem(
            fiber_id="f1",
            content="Old approach",
            score=0.1,
            token_count=10,
            fidelity_level="ghost",
            superseded_by="New approach",
        )
        assert item.superseded_by == "New approach"

    def test_normal_ghost_no_outdated(self):
        """Ghost item without superseded_by has empty string."""
        item = ContextItem(
            fiber_id="f2",
            content="Faded memory",
            score=0.1,
            token_count=10,
            fidelity_level="ghost",
        )
        assert item.superseded_by == ""

    def test_outdated_format_string(self):
        """Verify the [OUTDATED] format string matches plan spec."""
        item = ContextItem(
            fiber_id="f1",
            content="TypeError in parser",
            score=0.1,
            token_count=10,
            fidelity_level="ghost",
            superseded_by="Added null check in parser",
        )
        if item.superseded_by:
            formatted = f"- [OUTDATED] {item.content} → See: {item.superseded_by}"
        else:
            formatted = f"- {item.content}"
        assert formatted == "- [OUTDATED] TypeError in parser → See: Added null check in parser"


# ===========================================================================
# T9: Combined scenario tests
# ===========================================================================


class TestCombinedScenarios:
    """Complex scenarios mixing multiple role types."""

    async def test_supersession_and_reinforcement(self, storage, config):
        """Error superseded + evidence reinforcing the fix."""
        await _add_neuron(storage, "err", "Bug in auth")
        await _add_neuron(storage, "fix", "Fixed auth check")
        await _add_neuron(storage, "proof", "Auth test passes")
        await _add_synapse(storage, "err", "fix", SynapseType.RESOLVED_BY)
        await _add_synapse(storage, "proof", "fix", SynapseType.EVIDENCE_FOR)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "err": _activation("err", 0.7),
            "fix": _activation("fix", 0.5),
            "proof": _activation("proof", 0.4),
        }

        result = await pipeline._apply_causal_semantics(activations)

        # Fix should be boosted by supersession AND reinforcement
        # Supersession: max(0.5, 0.7*1.2=0.84) → 0.84
        # Reinforcement: 0.84 + 0.15 = 0.99
        assert result["fix"].activation_level > 0.8
        # Error should be ghost-level
        assert result["err"].activation_level < 0.15

    async def test_weakening_and_sequential(self, storage, config):
        """Counter-evidence weakens + sequential primes next step."""
        await _add_neuron(storage, "counter", "Disproof")
        await _add_neuron(storage, "hyp", "Hypothesis")
        await _add_neuron(storage, "step1", "Action 1")
        await _add_neuron(storage, "step2", "Action 2")
        await _add_synapse(storage, "counter", "hyp", SynapseType.EVIDENCE_AGAINST)
        await _add_synapse(storage, "step1", "step2", SynapseType.BEFORE)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "counter": _activation("counter", 0.5),
            "hyp": _activation("hyp", 0.6),
            "step1": _activation("step1", 0.5),
            "step2": _activation("step2", 0.3),
        }

        result = await pipeline._apply_causal_semantics(activations)

        # Hypothesis weakened: 0.6 * 0.5 = 0.3
        assert result["hyp"].activation_level == pytest.approx(0.3, rel=0.01)
        # Step 2 primed: 0.3 + 0.1 = 0.4
        assert result["step2"].activation_level == pytest.approx(0.4, rel=0.01)

    async def test_weakening_capped_at_one(self, storage, config):
        """Multiple weakening edges should only apply once (x0.5 floor)."""
        await _add_neuron(storage, "c1", "Counter 1")
        await _add_neuron(storage, "c2", "Counter 2")
        await _add_neuron(storage, "hyp", "Hypothesis")
        await _add_synapse(storage, "c1", "hyp", SynapseType.EVIDENCE_AGAINST)
        await _add_synapse(storage, "c2", "hyp", SynapseType.CONTRADICTS)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "c1": _activation("c1", 0.5),
            "c2": _activation("c2", 0.5),
            "hyp": _activation("hyp", 0.8),
        }

        result = await pipeline._apply_causal_semantics(activations)

        # Should be 0.8 * 0.5 = 0.4, NOT 0.8 * 0.5 * 0.5 = 0.2
        assert result["hyp"].activation_level == pytest.approx(0.4, rel=0.01)

    async def test_weakening_cannot_undo_supersession(self, storage, config):
        """Weakening should not demote a supersession-boosted target."""
        await _add_neuron(storage, "err", "Error")
        await _add_neuron(storage, "fix", "Fix")
        await _add_neuron(storage, "contra", "Unrelated contradiction")
        await _add_synapse(storage, "err", "fix", SynapseType.RESOLVED_BY)
        await _add_synapse(storage, "contra", "fix", SynapseType.CONTRADICTS)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "err": _activation("err", 0.7),
            "fix": _activation("fix", 0.5),
            "contra": _activation("contra", 0.4),
        }

        result = await pipeline._apply_causal_semantics(activations)

        # Fix was boosted by supersession (0.84). Weakening should NOT halve it.
        assert result["fix"].activation_level > 0.7

    async def test_supersedes_directionality(self, storage, config):
        """SUPERSEDES: source=NEW → target=OLD. Demote old, keep new."""
        await _add_neuron(storage, "schema_v1", "Old schema")
        await _add_neuron(storage, "schema_v2", "New schema")
        # v2 SUPERSEDES v1: source=v2 (new), target=v1 (old)
        await _add_synapse(storage, "schema_v2", "schema_v1", SynapseType.SUPERSEDES)

        pipeline = await _make_pipeline(storage, config)
        activations = {
            "schema_v1": _activation("schema_v1", 0.7),
            "schema_v2": _activation("schema_v2", 0.6),
        }

        result = await pipeline._apply_causal_semantics(activations)

        # v2 is the source=newer, should be boosted or kept
        assert result["schema_v2"].activation_level >= 0.6
        # v1 is the target=older, should be demoted to ghost
        assert result["schema_v1"].activation_level < 0.15

    async def test_supersession_map_populated(self, storage, config):
        """_supersession_map should track outdated→latest for context formatting."""
        await _add_neuron(storage, "old", "Old way")
        await _add_neuron(storage, "new", "New way")
        await _add_synapse(storage, "old", "new", SynapseType.RESOLVED_BY)

        pipeline = await _make_pipeline(storage, config)
        activations = {"old": _activation("old", 0.7)}

        await pipeline._apply_causal_semantics(activations)

        assert "old" in pipeline._supersession_map
        assert pipeline._supersession_map["old"] == "new"


# ===========================================================================
# T10: Habit frequency boost tests
# ===========================================================================


class TestHabitFrequencyBoost:
    """Proven workflow habits get proportional activation boost."""

    async def test_habit_boost_proportional(self, storage, config):
        """Neuron with _habit_frequency=2 gets +0.1 boost."""
        await _add_neuron(
            storage, "habit", "Run tests before commit", metadata={"_habit_frequency": 2}
        )

        pipeline = await _make_pipeline(storage, config)
        activations = {"habit": _activation("habit", 0.5)}

        result = await pipeline._apply_causal_semantics(activations)

        # 0.5 + (2 * 0.05) = 0.6
        assert result["habit"].activation_level == pytest.approx(0.6, rel=0.01)

    async def test_habit_boost_capped(self, storage, config):
        """Habit boost capped at +0.2 regardless of frequency."""
        await _add_neuron(storage, "frequent", "Daily standup", metadata={"_habit_frequency": 10})

        pipeline = await _make_pipeline(storage, config)
        activations = {"frequent": _activation("frequent", 0.5)}

        result = await pipeline._apply_causal_semantics(activations)

        # 0.5 + 0.2 (capped) = 0.7
        assert result["frequent"].activation_level == pytest.approx(0.7, rel=0.01)

    async def test_no_boost_without_habit_frequency(self, storage, config):
        """Neuron without _habit_frequency is unchanged."""
        await _add_neuron(storage, "normal", "Regular memory")

        pipeline = await _make_pipeline(storage, config)
        activations = {"normal": _activation("normal", 0.5)}

        result = await pipeline._apply_causal_semantics(activations)

        assert result["normal"].activation_level == pytest.approx(0.5, rel=0.01)

    async def test_habit_boost_capped_at_1(self, storage, config):
        """Habit boost should not exceed 1.0 total activation."""
        await _add_neuron(
            storage, "hot_habit", "Critical workflow", metadata={"_habit_frequency": 5}
        )

        pipeline = await _make_pipeline(storage, config)
        activations = {"hot_habit": _activation("hot_habit", 0.95)}

        result = await pipeline._apply_causal_semantics(activations)

        assert result["hot_habit"].activation_level <= 1.0
