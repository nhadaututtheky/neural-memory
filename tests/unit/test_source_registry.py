"""Tests for source registry (Phase 2: Source-Aware Memory)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.source import Source, SourceStatus, SourceType

# ──────────────────── Source dataclass ────────────────────


class TestSourceDataclass:
    """Verify Source frozen dataclass behavior."""

    def test_create_defaults(self) -> None:
        src = Source.create(brain_id="brain-1", name="BLDS 2015")
        assert src.name == "BLDS 2015"
        assert src.brain_id == "brain-1"
        assert src.source_type == SourceType.DOCUMENT
        assert src.status == SourceStatus.ACTIVE
        assert src.version == ""
        assert src.id  # auto-generated UUID

    def test_create_with_all_fields(self) -> None:
        src = Source.create(
            brain_id="b",
            name="Contract A",
            source_type="contract",
            version="2024-01",
            file_hash="abc123",
            metadata={"pages": 42},
            source_id="src-fixed",
        )
        assert src.id == "src-fixed"
        assert src.source_type == SourceType.CONTRACT
        assert src.file_hash == "abc123"
        assert src.metadata == {"pages": 42}

    def test_with_status_immutable(self) -> None:
        src = Source.create(brain_id="b", name="X")
        updated = src.with_status("superseded")
        assert updated.status == SourceStatus.SUPERSEDED
        assert src.status == SourceStatus.ACTIVE  # original unchanged

    def test_with_version_immutable(self) -> None:
        src = Source.create(brain_id="b", name="X")
        updated = src.with_version("v2.0")
        assert updated.version == "v2.0"
        assert src.version == ""  # original unchanged

    def test_is_active_property(self) -> None:
        src = Source.create(brain_id="b", name="X")
        assert src.is_active is True
        superseded = src.with_status("superseded")
        assert superseded.is_active is False

    def test_frozen_cannot_mutate(self) -> None:
        src = Source.create(brain_id="b", name="X")
        with pytest.raises(AttributeError):
            src.name = "Y"  # type: ignore[misc]

    def test_source_type_enum(self) -> None:
        for t in (
            "law",
            "contract",
            "ledger",
            "document",
            "api",
            "manual",
            "website",
            "book",
            "research",
        ):
            assert SourceType(t).value == t

    def test_source_status_enum(self) -> None:
        for s in ("active", "superseded", "repealed", "draft"):
            assert SourceStatus(s).value == s


# ──────────────────── SynapseType.SOURCE_OF ────────────────────


class TestSourceOfSynapseType:
    """Verify SOURCE_OF was added to SynapseType."""

    def test_source_of_exists(self) -> None:
        from neural_memory.core.synapse import SynapseType

        assert hasattr(SynapseType, "SOURCE_OF")
        assert SynapseType.SOURCE_OF.value == "source_of"


# ──────────────────── Schema migration v23 ────────────────────


class TestSchemaMigrationV23:
    """Verify schema version and migration SQL."""

    def test_schema_version_is_25(self) -> None:
        from neural_memory.storage.sqlite_schema import SCHEMA_VERSION

        assert SCHEMA_VERSION == 36

    def test_migration_22_23_exists(self) -> None:
        from neural_memory.storage.sqlite_schema import MIGRATIONS

        assert (22, 23) in MIGRATIONS
        stmts = MIGRATIONS[(22, 23)]
        assert len(stmts) >= 1
        assert "sources" in stmts[0].lower()

    def test_full_schema_has_sources_table(self) -> None:
        from neural_memory.storage.sqlite_schema import SCHEMA

        assert "CREATE TABLE IF NOT EXISTS sources" in SCHEMA


# ──────────────────── SQLite storage mixin ────────────────────


class TestSQLiteSourcesMixin:
    """Test source CRUD via SQLiteStorage."""

    @pytest.fixture
    async def storage(self, tmp_path):
        from neural_memory.storage.sqlite_store import SQLiteStorage

        db_path = tmp_path / "test.db"
        s = SQLiteStorage(db_path)
        await s.initialize()
        # Create a brain
        from neural_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await s.save_brain(brain)
        s.set_brain(brain.id)
        yield s
        await s.close()

    async def test_add_and_get_source(self, storage) -> None:
        src = Source.create(brain_id=storage.brain_id, name="Test Doc", source_id="src-1")
        await storage.add_source(src)

        fetched = await storage.get_source("src-1")
        assert fetched is not None
        assert fetched.name == "Test Doc"
        assert fetched.source_type == SourceType.DOCUMENT
        assert fetched.status == SourceStatus.ACTIVE

    async def test_get_source_not_found(self, storage) -> None:
        result = await storage.get_source("nonexistent")
        assert result is None

    async def test_list_sources(self, storage) -> None:
        for i in range(3):
            src = Source.create(brain_id=storage.brain_id, name=f"Doc {i}", source_id=f"src-{i}")
            await storage.add_source(src)

        sources = await storage.list_sources()
        assert len(sources) == 3

    async def test_list_sources_filter_type(self, storage) -> None:
        s1 = Source.create(brain_id=storage.brain_id, name="A", source_type="law", source_id="s1")
        s2 = Source.create(brain_id=storage.brain_id, name="B", source_type="api", source_id="s2")
        await storage.add_source(s1)
        await storage.add_source(s2)

        laws = await storage.list_sources(source_type="law")
        assert len(laws) == 1
        assert laws[0].name == "A"

    async def test_list_sources_filter_status(self, storage) -> None:
        s1 = Source.create(brain_id=storage.brain_id, name="A", source_id="s1")
        s2 = Source.create(brain_id=storage.brain_id, name="B", status="draft", source_id="s2")
        await storage.add_source(s1)
        await storage.add_source(s2)

        drafts = await storage.list_sources(status="draft")
        assert len(drafts) == 1
        assert drafts[0].name == "B"

    async def test_update_source(self, storage) -> None:
        src = Source.create(brain_id=storage.brain_id, name="X", source_id="s1")
        await storage.add_source(src)

        updated = await storage.update_source("s1", status="superseded", version="v2")
        assert updated is True

        fetched = await storage.get_source("s1")
        assert fetched is not None
        assert fetched.status == SourceStatus.SUPERSEDED
        assert fetched.version == "v2"

    async def test_update_source_not_found(self, storage) -> None:
        result = await storage.update_source("nonexistent", status="draft")
        assert result is False

    async def test_delete_source(self, storage) -> None:
        src = Source.create(brain_id=storage.brain_id, name="X", source_id="s1")
        await storage.add_source(src)

        deleted = await storage.delete_source("s1")
        assert deleted is True
        assert await storage.get_source("s1") is None

    async def test_delete_source_not_found(self, storage) -> None:
        result = await storage.delete_source("nonexistent")
        assert result is False

    async def test_find_source_by_name(self, storage) -> None:
        src = Source.create(brain_id=storage.brain_id, name="BLDS 2015", source_id="s1")
        await storage.add_source(src)

        found = await storage.find_source_by_name("BLDS 2015")
        assert found is not None
        assert found.id == "s1"

        not_found = await storage.find_source_by_name("nonexistent")
        assert not_found is None

    async def test_count_neurons_for_source(self, storage) -> None:
        # No synapses yet
        count = await storage.count_neurons_for_source("s1")
        assert count == 0


# ──────────────────── MCP handler ────────────────────


class TestSourceHandler:
    """Test nmem_source MCP handler."""

    def _make_handler(self, brain_id: str = "test-brain"):
        from neural_memory.mcp.tool_handlers import ToolHandler

        config = MagicMock()
        config.auto = MagicMock(enabled=False)
        config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
        config.safety = MagicMock(auto_redact_min_severity=3)

        handler = ToolHandler.__new__(ToolHandler)
        handler.config = config
        handler.hooks = MagicMock()
        handler.hooks.emit = AsyncMock()

        storage = AsyncMock()
        storage.brain_id = brain_id
        storage._current_brain_id = brain_id
        storage.current_brain_id = brain_id

        handler.storage = storage
        handler.get_storage = AsyncMock(return_value=storage)
        handler._check_maintenance = AsyncMock(return_value=None)
        handler._get_maintenance_hint = MagicMock(return_value=None)
        handler.get_update_hint = MagicMock(return_value=None)
        handler._record_tool_action = AsyncMock()
        handler._check_onboarding = AsyncMock(return_value=None)
        handler._surface_pending_alerts = AsyncMock(return_value=None)
        handler._check_cross_language_hint = AsyncMock(return_value=None)

        return handler, storage

    @pytest.mark.asyncio
    async def test_register_source(self) -> None:
        handler, storage = self._make_handler()
        storage.add_source = AsyncMock(return_value="src-123")

        result = await handler._source(
            {
                "action": "register",
                "name": "BLDS 2015",
                "source_type": "law",
            }
        )

        assert result["name"] == "BLDS 2015"
        assert result["source_type"] == "law"
        storage.add_source.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_missing_name(self) -> None:
        handler, _ = self._make_handler()
        result = await handler._source({"action": "register"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_sources(self) -> None:
        handler, storage = self._make_handler()
        mock_source = Source.create(brain_id="b", name="Test", source_id="s1")
        storage.list_sources = AsyncMock(return_value=[mock_source])

        result = await handler._source({"action": "list"})
        assert result["count"] == 1
        assert result["sources"][0]["name"] == "Test"

    @pytest.mark.asyncio
    async def test_get_source(self) -> None:
        handler, storage = self._make_handler()
        mock_source = Source.create(brain_id="b", name="Doc", source_id="s1")
        storage.get_source = AsyncMock(return_value=mock_source)
        storage.count_neurons_for_source = AsyncMock(return_value=5)

        result = await handler._source({"action": "get", "source_id": "s1"})
        assert result["name"] == "Doc"
        assert result["linked_neuron_count"] == 5

    @pytest.mark.asyncio
    async def test_get_source_not_found(self) -> None:
        handler, storage = self._make_handler()
        storage.get_source = AsyncMock(return_value=None)

        result = await handler._source({"action": "get", "source_id": "nope"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_source(self) -> None:
        handler, storage = self._make_handler()
        storage.update_source = AsyncMock(return_value=True)

        result = await handler._source(
            {
                "action": "update",
                "source_id": "s1",
                "status": "superseded",
            }
        )
        assert result["updated"] is True

    @pytest.mark.asyncio
    async def test_delete_with_linked_neurons(self) -> None:
        handler, storage = self._make_handler()
        storage.count_neurons_for_source = AsyncMock(return_value=10)
        storage.update_source = AsyncMock(return_value=True)

        result = await handler._source({"action": "delete", "source_id": "s1"})
        assert result["superseded"] is True
        assert result["deleted"] is False
        assert "warning" in result

    @pytest.mark.asyncio
    async def test_delete_no_linked_neurons(self) -> None:
        handler, storage = self._make_handler()
        storage.count_neurons_for_source = AsyncMock(return_value=0)
        storage.delete_source = AsyncMock(return_value=True)

        result = await handler._source({"action": "delete", "source_id": "s1"})
        assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_missing_action(self) -> None:
        handler, _ = self._make_handler()
        result = await handler._source({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:
        handler, _ = self._make_handler()
        result = await handler._source({"action": "explode"})
        assert "error" in result


# ──────────────────── Core export ────────────────────


class TestCoreExport:
    """Verify Source is exported from neural_memory.core."""

    def test_source_in_core(self) -> None:
        from neural_memory.core import Source, SourceStatus, SourceType

        assert Source is not None
        assert SourceType is not None
        assert SourceStatus is not None
