"""Tests for SQLite -> PostgreSQL migration utility."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.storage.postgres.postgres_migration import migrate_sqlite_to_postgres


@pytest.fixture
def mock_sqlite_storage():
    """Mock SQLite storage with export_brain support."""
    storage = AsyncMock()
    storage.initialize = AsyncMock()
    storage.close = AsyncMock()
    storage.set_brain = MagicMock()

    brain = MagicMock()
    brain.name = "test-brain"
    storage.get_brain = AsyncMock(return_value=brain)

    snapshot = MagicMock()
    snapshot.neurons = [MagicMock(), MagicMock()]
    snapshot.synapses = [MagicMock()]
    snapshot.fibers = [MagicMock(), MagicMock(), MagicMock()]
    storage.export_brain = AsyncMock(return_value=snapshot)

    return storage


@pytest.fixture
def mock_pg_storage():
    """Mock PostgreSQL storage with import_brain support."""
    storage = AsyncMock()
    storage.initialize = AsyncMock()
    storage.close = AsyncMock()
    storage.import_brain = AsyncMock(return_value="test-brain")
    return storage


@patch("neural_memory.storage.postgres.postgres_store.PostgreSQLStorage")
@patch("neural_memory.storage.sqlite_store.SQLiteStorage")
async def test_migrate_single_brain(
    mock_sqlite_cls, mock_pg_cls, mock_sqlite_storage, mock_pg_storage
):
    """Migrate a single named brain from SQLite to PostgreSQL."""
    mock_sqlite_cls.return_value = mock_sqlite_storage
    mock_pg_cls.return_value = mock_pg_storage

    result = await migrate_sqlite_to_postgres(
        sqlite_db_path="/tmp/test.db",
        pg_host="localhost",
        pg_port=5432,
        pg_database="neuralmemory",
        pg_user="postgres",
        pg_password="secret",
        brain_name="test-brain",
    )

    assert result["success"] is True
    assert result["total_brains"] == 1
    assert len(result["brains"]) == 1
    assert result["brains"][0]["name"] == "test-brain"
    assert result["brains"][0]["neurons"] == 2
    assert result["brains"][0]["synapses"] == 1
    assert result["brains"][0]["fibers"] == 3

    mock_sqlite_storage.set_brain.assert_called_once_with("test-brain")
    mock_sqlite_storage.export_brain.assert_called_once_with("test-brain")
    mock_pg_storage.import_brain.assert_called_once()

    # Both storages must be closed
    mock_sqlite_storage.close.assert_awaited_once()
    mock_pg_storage.close.assert_awaited_once()


@patch("neural_memory.storage.postgres.postgres_store.PostgreSQLStorage")
@patch("neural_memory.storage.sqlite_store.SQLiteStorage")
async def test_migrate_skips_missing_brain(
    mock_sqlite_cls, mock_pg_cls, mock_sqlite_storage, mock_pg_storage
):
    """Skip brain that doesn't exist in SQLite."""
    mock_sqlite_cls.return_value = mock_sqlite_storage
    mock_pg_cls.return_value = mock_pg_storage
    mock_sqlite_storage.get_brain = AsyncMock(return_value=None)

    result = await migrate_sqlite_to_postgres(
        sqlite_db_path="/tmp/test.db",
        brain_name="nonexistent",
    )

    assert result["success"] is True
    assert result["total_brains"] == 0
    assert len(result["brains"]) == 0
    mock_sqlite_storage.export_brain.assert_not_awaited()


@patch("neural_memory.storage.postgres.postgres_store.PostgreSQLStorage")
@patch("neural_memory.storage.sqlite_store.SQLiteStorage")
async def test_migrate_handles_error(
    mock_sqlite_cls, mock_pg_cls, mock_sqlite_storage, mock_pg_storage
):
    """Return error stats when migration fails."""
    mock_sqlite_cls.return_value = mock_sqlite_storage
    mock_pg_cls.return_value = mock_pg_storage
    mock_sqlite_storage.export_brain = AsyncMock(side_effect=RuntimeError("DB locked"))

    result = await migrate_sqlite_to_postgres(
        sqlite_db_path="/tmp/test.db",
        brain_name="test-brain",
    )

    assert result["success"] is False
    assert "DB locked" in result["error"]

    # Storages must still be closed on error
    mock_sqlite_storage.close.assert_awaited_once()
    mock_pg_storage.close.assert_awaited_once()


@patch("neural_memory.storage.postgres.postgres_store.PostgreSQLStorage")
@patch("neural_memory.storage.sqlite_store.SQLiteStorage")
async def test_migrate_passes_pg_params(mock_sqlite_cls, mock_pg_cls):
    """Verify PostgreSQL connection params are forwarded."""
    mock_sqlite = AsyncMock()
    mock_sqlite.initialize = AsyncMock()
    mock_sqlite.close = AsyncMock()
    mock_sqlite.set_brain = MagicMock()
    mock_sqlite.get_brain = AsyncMock(return_value=None)
    mock_sqlite_cls.return_value = mock_sqlite

    mock_pg = AsyncMock()
    mock_pg.initialize = AsyncMock()
    mock_pg.close = AsyncMock()
    mock_pg_cls.return_value = mock_pg

    await migrate_sqlite_to_postgres(
        sqlite_db_path="/tmp/test.db",
        pg_host="db.example.com",
        pg_port=5433,
        pg_database="mydb",
        pg_user="admin",
        pg_password="pass123",
        embedding_dim=768,
        brain_name="x",
    )

    mock_pg_cls.assert_called_once_with(
        host="db.example.com",
        port=5433,
        database="mydb",
        user="admin",
        password="pass123",
        embedding_dim=768,
    )


@patch("neural_memory.storage.postgres.postgres_store.PostgreSQLStorage")
@patch("neural_memory.storage.sqlite_store.SQLiteStorage")
async def test_migrate_default_embedding_dim(mock_sqlite_cls, mock_pg_cls):
    """Default embedding_dim is 384."""
    mock_sqlite = AsyncMock()
    mock_sqlite.initialize = AsyncMock()
    mock_sqlite.close = AsyncMock()
    mock_sqlite.set_brain = MagicMock()
    mock_sqlite.get_brain = AsyncMock(return_value=None)
    mock_sqlite_cls.return_value = mock_sqlite

    mock_pg = AsyncMock()
    mock_pg.initialize = AsyncMock()
    mock_pg.close = AsyncMock()
    mock_pg_cls.return_value = mock_pg

    await migrate_sqlite_to_postgres(
        sqlite_db_path="/tmp/test.db",
        brain_name="x",
    )

    call_kwargs = mock_pg_cls.call_args[1]
    assert call_kwargs["embedding_dim"] == 384
