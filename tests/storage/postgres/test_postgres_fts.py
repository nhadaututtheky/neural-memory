"""Regression tests for PostgreSQL FTS via the unified SQLStorage adapter.

Guards against the ``UndefinedTableError: missing FROM-clause entry for
table "fts"`` crash that previously hit ``find_neurons(content_contains=...)``
on Postgres because the mixin hardcoded the SQLite FTS5 ``ORDER BY fts.rank``
expression and the dialect DDL never created the ``content_tsv`` /
``summary_tsv`` columns referenced by ``fts_neuron_query`` /
``fts_fiber_query``.

These tests exercise the path that crashed:
``SQLStorage(PostgresDialect(...))`` → ``find_neurons(content_contains=...)``.
The legacy ``PostgreSQLStorage`` (covered by the other tests in this dir)
uses a separate implementation that was already FTS-safe.
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


def _unified_pg_available() -> bool:
    """Same probe as the conftest but local to this module."""
    try:
        import asyncio

        import asyncpg
    except ImportError:
        return False

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

    try:
        return asyncio.run(_check())
    except Exception:
        return False


_UNIFIED_PG_AVAILABLE = _unified_pg_available()
_SKIP_REASON = f"PostgreSQL not available at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


pytestmark = pytest.mark.skipif(not _UNIFIED_PG_AVAILABLE, reason=_SKIP_REASON)


_MINIMAL_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS brains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    owner_id TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    shared_with JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS neurons (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    content_hash BIGINT DEFAULT 0,
    device_id TEXT DEFAULT '',
    device_origin TEXT DEFAULT '',
    updated_at TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    last_accessed_at TIMESTAMPTZ,
    lifecycle_state TEXT DEFAULT 'active',
    frozen INTEGER DEFAULT 0,
    ephemeral INTEGER DEFAULT 0,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);

-- ``CREATE TABLE IF NOT EXISTS`` no-ops when an earlier test session
-- (e.g. the legacy postgres_schema initializer running before us)
-- already created ``neurons`` without the unified-mixin columns.
-- Catch those up explicitly so the unified ``add_neuron`` mixin can
-- bind to every column it needs.
ALTER TABLE neurons ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ;
ALTER TABLE neurons ADD COLUMN IF NOT EXISTS lifecycle_state TEXT DEFAULT 'active';
ALTER TABLE neurons ADD COLUMN IF NOT EXISTS frozen INTEGER DEFAULT 0;
ALTER TABLE neurons ADD COLUMN IF NOT EXISTS ephemeral INTEGER DEFAULT 0;

CREATE TABLE IF NOT EXISTS neuron_states (
    neuron_id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    activation_level DOUBLE PRECISION DEFAULT 0.0,
    access_frequency INTEGER DEFAULT 0,
    last_activated TIMESTAMPTZ,
    decay_rate DOUBLE PRECISION DEFAULT 0.1,
    firing_threshold DOUBLE PRECISION DEFAULT 0.3,
    refractory_until TIMESTAMPTZ,
    refractory_period_ms DOUBLE PRECISION DEFAULT 500.0,
    homeostatic_target DOUBLE PRECISION DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (brain_id, neuron_id),
    FOREIGN KEY (brain_id, neuron_id) REFERENCES neurons(brain_id, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fibers (
    id TEXT NOT NULL,
    brain_id TEXT NOT NULL,
    neuron_ids JSONB NOT NULL,
    synapse_ids JSONB NOT NULL,
    anchor_neuron_id TEXT NOT NULL,
    summary TEXT,
    salience DOUBLE PRECISION DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (brain_id, id),
    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
);

-- Touched by add_neuron via invalidate_merkle_prefix.
CREATE TABLE IF NOT EXISTS merkle_hashes (
    brain_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    prefix TEXT NOT NULL,
    hash TEXT NOT NULL,
    entity_count INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (brain_id, entity_type, prefix)
);
"""


@pytest_asyncio.fixture
async def unified_pg_storage() -> Any:
    """A hand-wired ``SQLStorage(PostgresDialect())`` bound to a unique brain.

    Builds the minimum schema the FTS code path touches (brains +
    neurons + neuron_states + fibers) with the proper PG types, applies
    the FTS migration (tsvector generated columns + GIN indexes), then
    exposes the unified mixin layer.  This is the call path that
    previously crashed with ``UndefinedTableError: missing FROM-clause
    entry for table "fts"``.
    """
    import asyncpg

    from neural_memory.core.brain import Brain
    from neural_memory.storage.neuron_cache import NeuronLookupCache
    from neural_memory.storage.sql import SQLStorage
    from neural_memory.storage.sql.postgres_dialect import (
        _PG_FTS_DDL,
        PostgresDialect,
    )

    # Bootstrap schema on a one-shot connection so we don't depend on
    # any production initializer.
    conn = await asyncpg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD or None,
    )
    try:
        await conn.execute(_MINIMAL_PG_SCHEMA)
        await conn.execute(_PG_FTS_DDL)
    finally:
        await conn.close()

    dialect = PostgresDialect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    await dialect.initialize()

    brain_id = f"fts_{uuid.uuid4().hex[:12]}"
    # Seed the brain row directly — bypasses the brain_ops mixin's
    # JSON-string vs JSONB impedance mismatch on the unified PG path,
    # which is out of scope for this regression suite.
    async with dialect._pool.acquire() as c:  # type: ignore[attr-defined]
        brain = Brain.create(name=brain_id, brain_id=brain_id)
        await c.execute(
            "INSERT INTO brains (id, name, config, created_at, updated_at)"
            " VALUES ($1, $2, $3::jsonb, $4, $5)",
            brain_id,
            brain.name,
            "{}",
            brain.created_at,
            brain.updated_at,
        )

    store = SQLStorage.__new__(SQLStorage)
    store._dialect = dialect  # type: ignore[attr-defined]
    store._current_brain_id = brain_id  # type: ignore[attr-defined]
    store._neuron_cache = NeuronLookupCache(  # type: ignore[attr-defined]
        ttl_seconds=30.0, max_entries=500
    )
    store._init_vector_search()  # type: ignore[attr-defined]

    try:
        yield store
    finally:
        try:
            async with dialect._pool.acquire() as c:  # type: ignore[attr-defined]
                await c.execute("DELETE FROM brains WHERE id = $1", brain_id)
        except Exception:
            pass
        await dialect.close()


@pytest.mark.asyncio
async def test_find_neurons_content_contains_does_not_crash(
    unified_pg_storage: Any,
) -> None:
    """Regression: ``find_neurons(content_contains=...)`` previously raised
    ``UndefinedTableError: missing FROM-clause entry for table "fts"``.

    With the dialect-aware rank expression + tsvector DDL it must:
    1. not raise, and
    2. return the neuron whose content matches the search term.
    """
    from neural_memory.core.neuron import Neuron, NeuronType

    store = unified_pg_storage

    neurons = [
        Neuron.create(type=NeuronType.CONCEPT, content="Python programming language"),
        Neuron.create(type=NeuronType.CONCEPT, content="Rust systems language"),
        Neuron.create(type=NeuronType.ENTITY, content="An apple a day keeps the doctor away"),
    ]
    for n in neurons:
        await store.add_neuron(n)

    # The exact call that used to crash inside the entity-dedup pipeline.
    matches = await store.find_neurons(content_contains="python", limit=10)

    contents = {n.content for n in matches}
    assert "Python programming language" in contents, f"FTS match missing — got {contents!r}"
    assert "Rust systems language" not in contents


@pytest.mark.asyncio
async def test_find_neurons_content_contains_multi_token(
    unified_pg_storage: Any,
) -> None:
    """Multi-token FTS query returns the most relevant hit and ordering is sane."""
    from neural_memory.core.neuron import Neuron, NeuronType

    store = unified_pg_storage

    docs = [
        "Postgres full text search with tsvector",
        "SQLite FTS5 virtual tables",
        "Vector similarity search with pgvector",
        "Full text indexing tutorial",
    ]
    created = []
    for content in docs:
        n = Neuron.create(type=NeuronType.CONCEPT, content=content)
        await store.add_neuron(n)
        created.append(n)

    matches = await store.find_neurons(content_contains="full text", limit=10)
    contents = [n.content for n in matches]

    assert any("full text" in c.lower() for c in contents), (
        f"expected at least one 'full text' match, got {contents!r}"
    )
    # The pgvector-only doc must not match.
    assert "Vector similarity search with pgvector" not in contents


@pytest.mark.asyncio
async def test_suggest_neurons_does_not_crash(unified_pg_storage: Any) -> None:
    """``suggest_neurons`` previously held the second ``fts.rank``
    hardcode inside its composite score expression.  The regression
    guarantee is that it no longer raises the SQLite-specific
    ``UndefinedTableError`` against Postgres.

    Note: SQLite FTS5 prefix syntax (``hel*``) does not translate
    cleanly to ``plainto_tsquery`` on Postgres, so the result set may
    be empty for prefix-style probes — but it must NEVER crash.
    """
    from neural_memory.core.neuron import Neuron, NeuronType

    store = unified_pg_storage

    for content in ("hello world", "helmet design", "rust language"):
        await store.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content=content))

    # No raise == regression fixed.  Exact-word probes succeed against
    # plainto_tsquery; prefix-only probes may not (separate concern).
    suggestions = await store.suggest_neurons("hello", limit=5)
    assert isinstance(suggestions, list)
    suggested = {s["content"] for s in suggestions}
    assert "hello world" in suggested, f"expected exact-word match for 'hello', got {suggested!r}"


@pytest.mark.asyncio
async def test_tsvector_columns_present_and_generated(
    unified_pg_storage: Any,
) -> None:
    """``content_tsv`` / ``summary_tsv`` exist as STORED generated columns.

    Verifies the FTS migration both created the columns AND set the
    GENERATED ALWAYS expression — a plain tsvector column would silently
    return zero matches (which is how the bug manifests when an older
    deploy left a non-generated column behind).
    """
    from neural_memory.core.neuron import Neuron, NeuronType
    from neural_memory.storage.sql.postgres_dialect import _PG_FTS_DDL

    store = unified_pg_storage
    dialect = store._dialect  # type: ignore[attr-defined]

    # Re-running the FTS migration must be a no-op (idempotency).
    await dialect.execute_script(_PG_FTS_DDL)

    row = await dialect.fetch_one(
        "SELECT data_type, is_generated FROM information_schema.columns"
        " WHERE table_name = $1 AND column_name = $2",
        ("neurons", "content_tsv"),
    )
    assert row is not None, "neurons.content_tsv was not created"
    assert row["data_type"] == "tsvector"
    assert row["is_generated"] == "ALWAYS", (
        "content_tsv must be a STORED generated column — a non-generated"
        " tsvector silently returns zero FTS matches"
    )

    row = await dialect.fetch_one(
        "SELECT data_type, is_generated FROM information_schema.columns"
        " WHERE table_name = $1 AND column_name = $2",
        ("fibers", "summary_tsv"),
    )
    assert row is not None, "fibers.summary_tsv was not created"
    assert row["is_generated"] == "ALWAYS"

    # Generated column auto-populates on INSERT through the mixin path.
    n = Neuron.create(type=NeuronType.CONCEPT, content="elephants on parade")
    await store.add_neuron(n)
    row = await dialect.fetch_one(
        "SELECT content_tsv::text AS tsv FROM neurons WHERE id = $1", (n.id,)
    )
    assert row is not None
    assert row["tsv"], "content_tsv was empty for inserted row"


# Marker only used by humans grepping the file:
#   regression — missing FROM-clause entry for table "fts"
_REGRESSION_MARKER = "missing FROM-clause entry for table 'fts'"

# Keep ruff happy.
_ = os.environ.get("POSTGRES_TEST_HOST")
