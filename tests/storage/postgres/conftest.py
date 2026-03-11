"""Shared fixtures for PostgreSQL storage tests.

All tests in this directory require a running PostgreSQL instance with pgvector.
Set POSTGRES_TEST_HOST / POSTGRES_TEST_PORT / POSTGRES_TEST_DB env vars to override.
Tests are auto-skipped when Postgres is not reachable or asyncpg not installed.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
import pytest_asyncio

POSTGRES_HOST = os.environ.get("POSTGRES_TEST_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_TEST_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_TEST_DB", "neuralmemory_test")
POSTGRES_USER = os.environ.get("POSTGRES_TEST_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_TEST_PASSWORD", "")


def postgres_available() -> bool:
    """Check if asyncpg + Postgres server are both available."""
    try:
        import asyncpg

        async def _check() -> bool:
            try:
                conn = await asyncpg.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    database=POSTGRES_DB,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD or None,
                    timeout=3,
                )
                await conn.execute("SELECT 1")
                await conn.close()
                return True
            except Exception:
                return False

        import asyncio

        return asyncio.run(_check())
    except ImportError:
        return False


_POSTGRES_AVAILABLE = postgres_available()
_SKIP_REASON = f"PostgreSQL not available at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


@pytest.fixture(autouse=True)
def _require_postgres() -> None:
    """Skip all tests when Postgres is not available."""
    if not _POSTGRES_AVAILABLE:
        pytest.skip(_SKIP_REASON)


def _import_deps() -> dict[str, Any]:
    """Lazy-import Postgres storage deps."""
    from neural_memory.core.brain import Brain
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import Neuron, NeuronType
    from neural_memory.core.synapse import Synapse, SynapseType
    from neural_memory.storage.postgres.postgres_store import PostgreSQLStorage

    return {
        "Brain": Brain,
        "Neuron": Neuron,
        "NeuronType": NeuronType,
        "Synapse": Synapse,
        "SynapseType": SynapseType,
        "Fiber": Fiber,
        "PostgreSQLStorage": PostgreSQLStorage,
    }


@pytest.fixture
def brain_id() -> str:
    """Unique brain ID per test to avoid collisions."""
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest_asyncio.fixture
async def storage(brain_id: str) -> Any:
    """PostgreSQL storage with a clean test brain."""
    deps = _import_deps()
    store = deps["PostgreSQLStorage"](
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    await store.initialize()

    brain = deps["Brain"].create(name=brain_id, brain_id=brain_id)
    await store.save_brain(brain)
    store.set_brain(brain_id)

    yield store

    try:
        await store.clear(brain_id)
    except Exception:
        pass
    await store.close()


@pytest.fixture
def make_neuron() -> Any:
    """Factory for creating test neurons."""
    deps = _import_deps()
    neuron_cls = deps["Neuron"]
    neuron_type_cls = deps["NeuronType"]

    def _make(
        type: Any = None,
        content: str = "test neuron",
        metadata: dict | None = None,
    ) -> Any:
        return neuron_cls.create(
            type=type or neuron_type_cls.CONCEPT,
            content=content,
            metadata=metadata or {},
        )

    return _make


@pytest.fixture
def sample_neurons() -> list[Any]:
    """Five neurons of different types."""
    deps = _import_deps()
    neuron_cls = deps["Neuron"]
    nt = deps["NeuronType"]
    return [
        neuron_cls.create(type=nt.TIME, content="Monday morning"),
        neuron_cls.create(type=nt.SPATIAL, content="Office building"),
        neuron_cls.create(type=nt.ENTITY, content="Python language"),
        neuron_cls.create(type=nt.ACTION, content="Write code"),
        neuron_cls.create(type=nt.CONCEPT, content="Neural memory"),
    ]


@pytest.fixture
def sample_synapses(sample_neurons: list[Any]) -> list[Any]:
    """Synapses connecting sample neurons in a chain."""
    deps = _import_deps()
    synapse_cls = deps["Synapse"]
    st = deps["SynapseType"]
    n = sample_neurons
    return [
        synapse_cls.create(n[0].id, n[1].id, st.HAPPENED_AT, weight=0.8),
        synapse_cls.create(n[1].id, n[2].id, st.AT_LOCATION, weight=0.6),
        synapse_cls.create(n[2].id, n[3].id, st.CAUSED_BY, weight=0.7),
        synapse_cls.create(n[3].id, n[4].id, st.RELATED_TO, weight=0.9),
    ]
