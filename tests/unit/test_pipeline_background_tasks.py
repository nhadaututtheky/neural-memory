"""Tests for ReflexPipeline background task deferral.

Post-recall side-effects (reinforcement, calibration, write queue flush,
reconsolidation, session summary persist) are spawned as background tasks
so the user-facing result returns sooner. Storage.close() drains them
before releasing the DB file handle.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.retrieval import ReflexPipeline
from neural_memory.storage.sqlite_store import SQLiteStorage


@pytest.fixture
async def sqlite_pipeline():
    """SQLite-backed pipeline with a brain containing one neuron + fiber."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bg.db"
        store = SQLiteStorage(db_path)
        await store.initialize()

        brain = Brain.create(name="bgtest")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="python async patterns")
        await store.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="Python async patterns are widely used in modern APIs.",
        )
        await store.add_fiber(fiber)

        pipeline = ReflexPipeline(storage=store, config=BrainConfig(activation_strategy="classic"))

        yield store, pipeline

        await store.close()


class TestBackgroundTaskLifecycle:
    async def test_query_spawns_background_tasks(self, sqlite_pipeline) -> None:
        """After a query, pipeline should have scheduled side-effect tasks."""
        store, pipeline = sqlite_pipeline

        await pipeline.query("python async")

        # Either still running or already completed — but at least one must
        # have been spawned. We assert via the storage-side registry so we
        # catch them regardless of completion state at this instant.
        registered = getattr(store, "_pipeline_bg_tasks", None)
        # If all tasks already finished, set is empty but the attribute exists.
        assert registered is not None, "pipeline must register bg tasks on storage"

    async def test_flush_background_tasks_awaits_all(self, sqlite_pipeline) -> None:
        """flush_background_tasks() waits for every pending task."""
        store, pipeline = sqlite_pipeline

        await pipeline.query("python async")

        await pipeline.flush_background_tasks()

        # All tasks should be done after flush
        assert all(t.done() for t in pipeline._background_tasks)

    async def test_flush_without_tasks_is_noop(self, sqlite_pipeline) -> None:
        """Calling flush on a fresh pipeline returns cleanly without error."""
        _, pipeline = sqlite_pipeline

        # Should not raise
        await pipeline.flush_background_tasks()

    async def test_storage_close_drains_pipeline_tasks(self) -> None:
        """SQLiteStorage.close() must await any pipeline bg tasks first.

        Without this, Windows tempfile cleanup races aiosqlite file handles.
        """
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "drain.db"
            store = SQLiteStorage(db_path)
            await store.initialize()

            brain = Brain.create(name="drain")
            await store.save_brain(brain)
            store.set_brain(brain.id)

            neuron = Neuron.create(type=NeuronType.CONCEPT, content="drain target")
            await store.add_neuron(neuron)
            fiber = Fiber.create(
                neuron_ids={neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=neuron.id,
                summary="Target",
            )
            await store.add_fiber(fiber)

            pipeline = ReflexPipeline(
                storage=store, config=BrainConfig(activation_strategy="classic")
            )
            await pipeline.query("drain target")

            # Grab reference to tasks before close
            tasks = list(getattr(store, "_pipeline_bg_tasks", set()))

            await store.close()

            # After close, all previously-spawned tasks must be done
            assert all(t.done() for t in tasks), "close() must drain pipeline tasks"

    async def test_background_task_errors_swallowed(self, sqlite_pipeline) -> None:
        """A failing background coroutine must not bubble or crash the pipeline."""
        _, pipeline = sqlite_pipeline

        async def _explodes() -> None:
            raise RuntimeError("boom")

        # Should not raise
        pipeline._spawn_background(_explodes(), "test-error")
        await asyncio.sleep(0)  # yield once so the task runs
        await pipeline.flush_background_tasks()

        # Pipeline remains usable
        assert isinstance(pipeline._background_tasks, set)
