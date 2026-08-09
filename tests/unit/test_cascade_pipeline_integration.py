"""Integration tests for cascaded recall inside ReflexPipeline.query()."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.retrieval import ReflexPipeline
from neural_memory.storage.sqlite_store import SQLiteStorage


@pytest.fixture
async def pipeline_storage(
    tmp_path: Path,
) -> tuple[ReflexPipeline, SQLiteStorage]:
    db_path = tmp_path / "cascade.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()
    brain = Brain.create(
        name="cascade_test",
        config=BrainConfig(
            cascade_recall_enabled=True,
            fiber_summary_tier_enabled=False,  # force neuron pipeline
            familiarity_fallback_enabled=False,
        ),
    )
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    # Seed graph: exact hit + neighbors for scope expansion
    n_exact = Neuron.create(
        type=NeuronType.CONCEPT,
        content="Alice email is alice@example.com",
    )
    n_auth = Neuron.create(
        type=NeuronType.CONCEPT,
        content="authentication JWT token flow",
    )
    n_cause = Neuron.create(
        type=NeuronType.CONCEPT,
        content="build failed because missing dependency",
    )
    n_effect = Neuron.create(
        type=NeuronType.CONCEPT,
        content="CI pipeline blocked deploys",
    )
    for n in (n_exact, n_auth, n_cause, n_effect):
        await storage.add_neuron(n)

    # No causal synapse — avoids temporal-reasoning fast path so cascade runs
    pipe = ReflexPipeline(storage, brain.config)
    yield pipe, storage
    await storage.close()


class TestCascadePipeline:
    @pytest.mark.asyncio
    async def test_query_emits_cascade_metadata(
        self, pipeline_storage: tuple[ReflexPipeline, SQLiteStorage]
    ) -> None:
        pipe, _ = pipeline_storage
        result = await pipe.query("Alice email")
        assert "cascade_route" in result.metadata
        assert "cascade_stage" in result.metadata
        assert "cascade_gate_reason" in result.metadata
        assert result.metadata["cascade_route"] in {
            "exact",
            "semantic",
            "temporal",
            "causal",
        }

    @pytest.mark.asyncio
    async def test_causal_query_requires_graph_stage(
        self, pipeline_storage: tuple[ReflexPipeline, SQLiteStorage]
    ) -> None:
        pipe, _ = pipeline_storage
        result = await pipe.query("Why did the build fail?")
        assert result.metadata.get("cascade_route") == "causal"
        # Must not early-exit on lexical alone
        assert result.metadata.get("cascade_stage") != "candidate_exit"
        assert result.metadata.get("cascade_sufficient") is False

    @pytest.mark.asyncio
    async def test_cascade_disabled_skips_metadata(
        self, pipeline_storage: tuple[ReflexPipeline, SQLiteStorage]
    ) -> None:
        pipe, storage = pipeline_storage
        disabled = ReflexPipeline(
            storage,
            BrainConfig(
                cascade_recall_enabled=False,
                fiber_summary_tier_enabled=False,
            ),
        )
        result = await disabled.query("Alice email")
        assert "cascade_route" not in result.metadata

    @pytest.mark.asyncio
    async def test_phase_timings_include_cascade(
        self, pipeline_storage: tuple[ReflexPipeline, SQLiteStorage]
    ) -> None:
        pipe, _ = pipeline_storage
        result = await pipe.query("authentication JWT")
        timings = result.metadata.get("phase_timings_ms", {})
        assert "cascade_pregraph" in timings
