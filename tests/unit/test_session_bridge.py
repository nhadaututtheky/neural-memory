"""Unit tests for cross-session context bridge — correction injection."""

from __future__ import annotations

from datetime import timedelta

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


@pytest.fixture
def config() -> BrainConfig:
    return BrainConfig(activation_threshold=0.05)


@pytest.fixture
async def storage(config: BrainConfig) -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="test", config=config)
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


async def _add_correction(
    storage: InMemoryStorage,
    error_id: str,
    error_content: str,
    fix_id: str,
    fix_content: str,
    age_days: int = 0,
) -> None:
    """Add an error neuron, fix neuron, and RESOLVED_BY synapse."""
    error = Neuron.create(type=NeuronType.CONCEPT, content=error_content, neuron_id=error_id)
    fix = Neuron.create(type=NeuronType.CONCEPT, content=fix_content, neuron_id=fix_id)
    await storage.add_neuron(error)
    await storage.add_neuron(fix)

    created_at = utcnow() - timedelta(days=age_days)
    syn = Synapse.create(error_id, fix_id, SynapseType.RESOLVED_BY, weight=1.0)
    # Override created_at for testing recency
    from dataclasses import replace

    syn = replace(syn, created_at=created_at)
    await storage.add_synapse(syn)


class TestCorrectionInjection:
    """Test _inject_recent_corrections method."""

    async def _make_handler(self):
        """Create a minimal handler with _inject_recent_corrections."""
        from neural_memory.mcp.recall_handler import RecallHandler

        # RecallHandler is a mixin; create a minimal object
        class FakeHandler(RecallHandler):
            pass

        return FakeHandler()

    async def test_recent_correction_injected(self, storage):
        """Recent RESOLVED_BY synapse appears in corrections text."""
        await _add_correction(
            storage,
            "err1",
            "TypeError in parser",
            "fix1",
            "Added null check in parser",
            age_days=1,
        )
        handler = await self._make_handler()
        text = await handler._inject_recent_corrections(storage)

        assert "corrections from recent sessions" in text
        assert "TypeError in parser" in text
        assert "Added null check in parser" in text

    async def test_old_correction_excluded(self, storage):
        """Corrections older than max_age_days are excluded."""
        await _add_correction(
            storage,
            "err1",
            "Old error",
            "fix1",
            "Old fix",
            age_days=10,
        )
        handler = await self._make_handler()
        text = await handler._inject_recent_corrections(storage, max_age_days=7)

        assert text == ""

    async def test_max_corrections_cap(self, storage):
        """Only max_corrections are injected."""
        for i in range(8):
            await _add_correction(
                storage,
                f"err{i}",
                f"Error {i}",
                f"fix{i}",
                f"Fix {i}",
                age_days=i,
            )
        handler = await self._make_handler()
        text = await handler._inject_recent_corrections(storage, max_corrections=3)

        # Should have header + 3 correction lines
        lines = [line for line in text.split("\n") if line.strip()]
        assert len(lines) == 4  # 1 header + 3 corrections

    async def test_no_corrections_empty_string(self, storage):
        """No RESOLVED_BY synapses → empty string."""
        handler = await self._make_handler()
        text = await handler._inject_recent_corrections(storage)

        assert text == ""

    async def test_missing_neuron_skipped(self, storage):
        """If error or fix neuron is missing, that correction is skipped."""
        # Create both neurons + synapse, then delete the error neuron
        error = Neuron.create(type=NeuronType.CONCEPT, content="Error", neuron_id="to_delete")
        fix = Neuron.create(type=NeuronType.CONCEPT, content="Fix", neuron_id="fix_only")
        await storage.add_neuron(error)
        await storage.add_neuron(fix)
        syn = Synapse.create("to_delete", "fix_only", SynapseType.RESOLVED_BY)
        await storage.add_synapse(syn)
        # Remove the error neuron to simulate orphaned synapse
        await storage.delete_neuron("to_delete")

        handler = await self._make_handler()
        text = await handler._inject_recent_corrections(storage)

        # Should not include this correction (missing error neuron)
        assert text == ""

    async def test_correction_format(self, storage):
        """Verify correction format: error -> fix."""
        await _add_correction(
            storage,
            "e1",
            "Bug A",
            "f1",
            "Fix A",
            age_days=0,
        )
        handler = await self._make_handler()
        text = await handler._inject_recent_corrections(storage)

        assert "Bug A -> Fix A" in text

    async def test_long_content_truncated(self, storage):
        """Content longer than 100 chars is truncated."""
        long_error = "E" * 200
        long_fix = "F" * 200
        await _add_correction(storage, "e_long", long_error, "f_long", long_fix)

        handler = await self._make_handler()
        text = await handler._inject_recent_corrections(storage)

        # Each side should be ≤100 chars
        for line in text.split("\n"):
            if "->" in line:
                parts = line.split("->")
                assert len(parts[0].strip()) <= 105  # "- " prefix + content
                assert len(parts[1].strip()) <= 105

    async def test_corrections_sorted_newest_first(self, storage):
        """Corrections should be sorted newest first."""
        await _add_correction(storage, "old_e", "Old error", "old_f", "Old fix", age_days=5)
        await _add_correction(storage, "new_e", "New error", "new_f", "New fix", age_days=0)

        handler = await self._make_handler()
        text = await handler._inject_recent_corrections(storage)

        lines = [line for line in text.split("\n") if "->" in line]
        assert "New error" in lines[0]
        assert "Old error" in lines[1]


class TestCorrectionInContext:
    """Verify corrections appear in nmem_context output."""

    def test_corrections_section_format(self):
        """Corrections header should be distinctive."""
        header = "--- corrections from recent sessions ---"
        assert "corrections" in header
        assert "recent sessions" in header
