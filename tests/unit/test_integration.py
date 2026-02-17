"""Tests for the external source integration layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.memory_types import MemoryType
from neural_memory.core.synapse import SynapseType
from neural_memory.integration.adapter import SourceAdapter
from neural_memory.integration.mapper import (
    _RELATION_TYPE_MAP,
    _SOURCE_TYPE_MAP,
    MappingResult,
    RecordMapper,
)
from neural_memory.integration.models import (
    ExternalRecord,
    ExternalRelationship,
    ImportResult,
    SourceCapability,
    SourceSystemType,
    SyncState,
)
from neural_memory.integration.sync_engine import SyncEngine
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow

# ---------------------------------------------------------------------------
# Mock adapter for testing
# ---------------------------------------------------------------------------


class MockAdapter:
    """Mock adapter implementing the SourceAdapter protocol."""

    def __init__(self, records: list[ExternalRecord] | None = None) -> None:
        self._records = records or []

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.MEMORY_LAYER

    @property
    def system_name(self) -> str:
        return "mock"

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset(
            {
                SourceCapability.FETCH_ALL,
                SourceCapability.HEALTH_CHECK,
            }
        )

    async def fetch_all(
        self,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        records = self._records
        if limit:
            records = records[:limit]
        return records

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        raise NotImplementedError

    async def health_check(self) -> dict[str, Any]:
        return {"healthy": True, "message": "Mock adapter OK"}


class MockAdapterWithIncremental(MockAdapter):
    """Mock adapter that also supports incremental fetch."""

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset(
            {
                SourceCapability.FETCH_ALL,
                SourceCapability.FETCH_SINCE,
                SourceCapability.HEALTH_CHECK,
            }
        )

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        filtered = [r for r in self._records if r.created_at >= since]
        if limit:
            filtered = filtered[:limit]
        return filtered


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample_records() -> list[ExternalRecord]:
    return [
        ExternalRecord.create(
            id="rec-1",
            source_system="mock",
            content="We decided to use PostgreSQL for the main database",
            source_type="decision",
            tags={"database", "architecture"},
            created_at=datetime(2024, 6, 1, 10, 0),
        ),
        ExternalRecord.create(
            id="rec-2",
            source_system="mock",
            content="The API returns 429 when rate limit is exceeded",
            source_type="error",
            tags={"api"},
            created_at=datetime(2024, 6, 2, 14, 30),
        ),
    ]


@pytest.fixture
def sample_records() -> list[ExternalRecord]:
    return _make_sample_records()


@pytest.fixture
def brain_config() -> BrainConfig:
    return BrainConfig()


@pytest.fixture
def brain(brain_config: BrainConfig) -> Brain:
    return Brain.create(name="test_integration", config=brain_config)


@pytest.fixture
async def storage(brain: Brain) -> InMemoryStorage:
    store = InMemoryStorage()
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


# ---------------------------------------------------------------------------
# TestExternalRecord
# ---------------------------------------------------------------------------


class TestExternalRecord:
    """Tests for ExternalRecord data model."""

    def test_create_minimal(self) -> None:
        record = ExternalRecord.create(
            id="r1",
            source_system="test",
            content="Hello world",
        )
        assert record.id == "r1"
        assert record.source_system == "test"
        assert record.content == "Hello world"
        assert record.source_collection == "default"
        assert record.source_type is None
        assert record.embedding is None
        assert record.tags == frozenset()
        assert record.relationships == ()

    def test_create_with_all_fields(self) -> None:
        rel = ExternalRelationship(
            source_record_id="r1",
            target_record_id="r2",
            relation_type="related_to",
        )
        record = ExternalRecord.create(
            id="r1",
            source_system="chromadb",
            content="Test content",
            source_collection="my_collection",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 6, 1),
            source_type="document",
            metadata={"key": "value"},
            embedding=[0.1, 0.2, 0.3],
            tags={"tag1", "tag2"},
            relationships=[rel],
        )
        assert record.source_collection == "my_collection"
        assert record.source_type == "document"
        assert record.embedding == [0.1, 0.2, 0.3]
        assert record.tags == frozenset({"tag1", "tag2"})
        assert len(record.relationships) == 1

    def test_frozen_immutability(self) -> None:
        record = ExternalRecord.create(id="r1", source_system="t", content="x")
        with pytest.raises(AttributeError):
            record.content = "modified"  # type: ignore[misc]

    def test_factory_defaults(self) -> None:
        record = ExternalRecord.create(id="r1", source_system="t", content="x")
        assert isinstance(record.created_at, datetime)
        assert record.metadata == {}


# ---------------------------------------------------------------------------
# TestExternalRelationship
# ---------------------------------------------------------------------------


class TestExternalRelationship:
    def test_create_minimal(self) -> None:
        rel = ExternalRelationship(
            source_record_id="a",
            target_record_id="b",
            relation_type="caused_by",
        )
        assert rel.weight == 0.5
        assert rel.metadata == {}

    def test_create_with_metadata(self) -> None:
        rel = ExternalRelationship(
            source_record_id="a",
            target_record_id="b",
            relation_type="leads_to",
            weight=0.8,
            metadata={"context": "temporal"},
        )
        assert rel.weight == 0.8
        assert rel.metadata["context"] == "temporal"


# ---------------------------------------------------------------------------
# TestSyncState
# ---------------------------------------------------------------------------


class TestSyncState:
    def test_create_default(self) -> None:
        state = SyncState(source_system="test", source_collection="default")
        assert state.last_sync_at is None
        assert state.records_imported == 0
        assert state.last_record_id is None

    def test_with_update_immutability(self) -> None:
        state = SyncState(source_system="test", source_collection="default")
        now = utcnow()
        updated = state.with_update(last_sync_at=now, records_imported=10)
        # Original unchanged
        assert state.records_imported == 0
        assert state.last_sync_at is None
        # Updated has new values
        assert updated.records_imported == 10
        assert updated.last_sync_at == now

    def test_with_update_preserves_fields(self) -> None:
        state = SyncState(
            source_system="chromadb",
            source_collection="docs",
            records_imported=5,
        )
        updated = state.with_update(records_imported=10)
        assert updated.source_system == "chromadb"
        assert updated.source_collection == "docs"


# ---------------------------------------------------------------------------
# TestImportResult
# ---------------------------------------------------------------------------


class TestImportResult:
    def test_create_default(self) -> None:
        result = ImportResult(source_system="test", source_collection="default")
        assert result.records_fetched == 0
        assert result.records_imported == 0
        assert result.errors == ()
        assert result.fibers_created == ()

    def test_tuple_immutability(self) -> None:
        result = ImportResult(
            source_system="test",
            source_collection="default",
            errors=("error1", "error2"),
        )
        assert len(result.errors) == 2
        with pytest.raises(AttributeError):
            result.errors = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRecordMapper
# ---------------------------------------------------------------------------


class TestRecordMapper:
    @pytest.mark.asyncio
    async def test_map_simple_record(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        mapper = RecordMapper(storage, brain_config)
        record = ExternalRecord.create(
            id="test-1",
            source_system="mock",
            content="We decided to use Redis for caching",
            source_type="decision",
        )
        result = await mapper.map_record(record)

        assert isinstance(result, MappingResult)
        assert result.external_record_id == "test-1"
        assert result.source_system == "mock"
        assert result.encoding_result.fiber is not None
        assert len(result.encoding_result.neurons_created) > 0

    @pytest.mark.asyncio
    async def test_map_record_preserves_provenance(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        mapper = RecordMapper(storage, brain_config)
        record = ExternalRecord.create(
            id="prov-1",
            source_system="chromadb",
            content="Python 3.11 was released in October 2022",
        )
        result = await mapper.map_record(record)

        assert result.typed_memory.provenance.source == "import:chromadb"
        assert result.typed_memory.provenance.confidence.value == "medium"

    @pytest.mark.asyncio
    async def test_map_record_with_embedding(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        mapper = RecordMapper(storage, brain_config)
        embedding = [0.1] * 1536
        record = ExternalRecord.create(
            id="emb-1",
            source_system="chromadb",
            content="Document with embedding vector",
            embedding=embedding,
        )
        result = await mapper.map_record(record)

        # Check embedding stored on anchor neuron
        anchor = await storage.get_neuron(result.encoding_result.fiber.anchor_neuron_id)
        assert anchor is not None
        assert "embedding" in anchor.metadata
        assert len(anchor.metadata["embedding"]) == 1536

    def test_resolve_memory_type_explicit(self) -> None:
        mapper = RecordMapper.__new__(RecordMapper)
        record = ExternalRecord.create(
            id="t1", source_system="x", content="x", source_type="decision"
        )
        assert mapper._resolve_memory_type(record) == MemoryType.DECISION

    def test_resolve_memory_type_heuristic(self) -> None:
        mapper = RecordMapper.__new__(RecordMapper)
        record = ExternalRecord.create(
            id="t2",
            source_system="x",
            content="We decided to use PostgreSQL",
            source_type="unknown_type",
        )
        # Falls through to heuristic since "unknown_type" not in map
        result = mapper._resolve_memory_type(record)
        assert result == MemoryType.DECISION

    def test_resolve_synapse_type_known(self) -> None:
        mapper = RecordMapper.__new__(RecordMapper)
        assert mapper._resolve_synapse_type("caused_by") == SynapseType.CAUSED_BY
        assert mapper._resolve_synapse_type("leads_to") == SynapseType.LEADS_TO

    def test_resolve_synapse_type_unknown_defaults_related(self) -> None:
        mapper = RecordMapper.__new__(RecordMapper)
        assert mapper._resolve_synapse_type("some_unknown") == SynapseType.RELATED_TO

    @pytest.mark.asyncio
    async def test_map_record_creates_typed_memory(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        mapper = RecordMapper(storage, brain_config)
        record = ExternalRecord.create(
            id="tm-1",
            source_system="mem0",
            content="User prefers dark mode",
            source_type="preference",
            tags={"ui"},
        )
        result = await mapper.map_record(record)

        assert result.typed_memory.memory_type == MemoryType.PREFERENCE
        assert "import:mem0" in result.typed_memory.tags
        assert result.typed_memory.metadata["import_source"] == "mem0"

    @pytest.mark.asyncio
    async def test_create_relationship_synapses(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        mapper = RecordMapper(storage, brain_config)

        # Import two records
        rec1 = ExternalRecord.create(
            id="rel-1", source_system="mock", content="PostgreSQL database setup"
        )
        rec2 = ExternalRecord.create(
            id="rel-2", source_system="mock", content="Redis caching layer"
        )
        result1 = await mapper.map_record(rec1)
        result2 = await mapper.map_record(rec2)

        record_to_fiber = {
            "rel-1": result1.encoding_result.fiber.id,
            "rel-2": result2.encoding_result.fiber.id,
        }

        relationships = [
            ExternalRelationship(
                source_record_id="rel-1",
                target_record_id="rel-2",
                relation_type="leads_to",
                weight=0.7,
            ),
        ]

        synapses = await mapper.create_relationship_synapses(record_to_fiber, relationships)
        assert len(synapses) == 1
        assert synapses[0].type == SynapseType.LEADS_TO
        assert synapses[0].weight == 0.7


# ---------------------------------------------------------------------------
# TestSyncEngine
# ---------------------------------------------------------------------------


class TestSyncEngine:
    @pytest.mark.asyncio
    async def test_sync_full_import(
        self,
        storage: InMemoryStorage,
        brain_config: BrainConfig,
        sample_records: list[ExternalRecord],
    ) -> None:
        adapter = MockAdapter(records=sample_records)
        engine = SyncEngine(storage, brain_config, batch_size=10)

        result, state = await engine.sync(adapter)

        assert result.records_fetched == 2
        assert result.records_imported == 2
        assert result.records_skipped == 0
        assert result.records_failed == 0
        assert len(result.fibers_created) == 2
        assert state.records_imported == 2
        assert state.last_sync_at is not None

    @pytest.mark.asyncio
    async def test_sync_incremental(
        self,
        storage: InMemoryStorage,
        brain_config: BrainConfig,
    ) -> None:
        old_record = ExternalRecord.create(
            id="old-1",
            source_system="mock",
            content="Old data from January",
            created_at=datetime(2024, 1, 1),
        )
        new_record = ExternalRecord.create(
            id="new-1",
            source_system="mock",
            content="New data from June",
            created_at=datetime(2024, 6, 15),
        )
        adapter = MockAdapterWithIncremental(records=[old_record, new_record])

        # First sync with a "since" state
        sync_state = SyncState(
            source_system="mock",
            source_collection="default",
            last_sync_at=datetime(2024, 6, 1),
        )
        result, state = await engine_sync(storage, brain_config, adapter, sync_state)

        # Only the new record should be fetched
        assert result.records_fetched == 1
        assert result.records_imported == 1

    @pytest.mark.asyncio
    async def test_sync_skips_empty_content(
        self,
        storage: InMemoryStorage,
        brain_config: BrainConfig,
    ) -> None:
        records = [
            ExternalRecord.create(id="empty-1", source_system="mock", content=""),
            ExternalRecord.create(id="empty-2", source_system="mock", content="   "),
            ExternalRecord.create(
                id="valid-1",
                source_system="mock",
                content="Valid content here",
            ),
        ]
        adapter = MockAdapter(records=records)
        engine = SyncEngine(storage, brain_config)

        result, _ = await engine.sync(adapter)

        assert result.records_imported == 1
        assert result.records_skipped == 2

    @pytest.mark.asyncio
    async def test_sync_skips_duplicates(
        self,
        storage: InMemoryStorage,
        brain_config: BrainConfig,
        sample_records: list[ExternalRecord],
    ) -> None:
        adapter = MockAdapter(records=sample_records)
        engine = SyncEngine(storage, brain_config)

        # First sync
        result1, _ = await engine.sync(adapter)
        assert result1.records_imported == 2

        # Second sync — same records should be skipped
        result2, _ = await engine.sync(adapter)
        assert result2.records_imported == 0
        assert result2.records_skipped == 2

    @pytest.mark.asyncio
    async def test_sync_progress_callback(
        self,
        storage: InMemoryStorage,
        brain_config: BrainConfig,
        sample_records: list[ExternalRecord],
    ) -> None:
        adapter = MockAdapter(records=sample_records)
        engine = SyncEngine(storage, brain_config)

        progress_calls: list[tuple[int, int, str]] = []

        def on_progress(processed: int, total: int, record_id: str) -> None:
            progress_calls.append((processed, total, record_id))

        await engine.sync(adapter, progress_callback=on_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0][0] == 1  # First record
        assert progress_calls[1][0] == 2  # Second record
        assert progress_calls[0][1] == 2  # Total is 2

    @pytest.mark.asyncio
    async def test_sync_updates_sync_state(
        self,
        storage: InMemoryStorage,
        brain_config: BrainConfig,
        sample_records: list[ExternalRecord],
    ) -> None:
        adapter = MockAdapter(records=sample_records)
        engine = SyncEngine(storage, brain_config)

        _, state = await engine.sync(adapter)

        assert state.source_system == "mock"
        assert state.records_imported == 2
        assert state.last_record_id == "rec-2"
        assert state.last_sync_at is not None

    @pytest.mark.asyncio
    async def test_health_check(
        self,
        storage: InMemoryStorage,
        brain_config: BrainConfig,
    ) -> None:
        adapter = MockAdapter()
        engine = SyncEngine(storage, brain_config)

        result = await engine.health_check(adapter)

        assert result["healthy"] is True
        assert "Mock" in result["message"]

    @pytest.mark.asyncio
    async def test_sync_with_limit(
        self,
        storage: InMemoryStorage,
        brain_config: BrainConfig,
        sample_records: list[ExternalRecord],
    ) -> None:
        adapter = MockAdapter(records=sample_records)
        engine = SyncEngine(storage, brain_config)

        result, _ = await engine.sync(adapter, limit=1)

        assert result.records_fetched == 1
        assert result.records_imported == 1


# Helper for incremental test
async def engine_sync(
    storage: InMemoryStorage,
    config: BrainConfig,
    adapter: MockAdapterWithIncremental,
    sync_state: SyncState,
) -> tuple[ImportResult, SyncState]:
    engine = SyncEngine(storage, config)
    return await engine.sync(adapter, sync_state=sync_state)


# ---------------------------------------------------------------------------
# TestAdapterRegistry
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    def test_list_adapters(self) -> None:
        from neural_memory.integration.adapters import list_adapters

        adapters = list_adapters()
        assert "chromadb" in adapters
        assert "mem0" in adapters
        assert "awf" in adapters
        assert "cognee" in adapters
        assert "graphiti" in adapters
        assert "llamaindex" in adapters

    def test_get_unknown_adapter_raises(self) -> None:
        from neural_memory.integration.adapters import get_adapter

        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("nonexistent_system")

    def test_register_and_get_adapter(self) -> None:
        from neural_memory.integration.adapters import (
            get_adapter,
            register_adapter,
        )

        register_adapter("mock_test", MockAdapter)
        adapter = get_adapter("mock_test")
        assert adapter.system_name == "mock"


# ---------------------------------------------------------------------------
# TestSourceAdapterProtocol
# ---------------------------------------------------------------------------


class TestSourceAdapterProtocol:
    def test_mock_adapter_is_source_adapter(self) -> None:
        adapter = MockAdapter()
        assert isinstance(adapter, SourceAdapter)

    def test_mock_adapter_properties(self) -> None:
        adapter = MockAdapter()
        assert adapter.system_type == SourceSystemType.MEMORY_LAYER
        assert adapter.system_name == "mock"
        assert SourceCapability.FETCH_ALL in adapter.capabilities


# ---------------------------------------------------------------------------
# TestLookupTables
# ---------------------------------------------------------------------------


class TestLookupTables:
    def test_source_type_map_coverage(self) -> None:
        """Verify key external types are mapped."""
        assert "fact" in _SOURCE_TYPE_MAP
        assert "decision" in _SOURCE_TYPE_MAP
        assert "document" in _SOURCE_TYPE_MAP
        assert "knowledge" in _SOURCE_TYPE_MAP

    def test_relation_type_map_coverage(self) -> None:
        """Verify key relation types are mapped."""
        assert "related_to" in _RELATION_TYPE_MAP
        assert "caused_by" in _RELATION_TYPE_MAP
        assert "leads_to" in _RELATION_TYPE_MAP
        assert "enables" in _RELATION_TYPE_MAP


# ── AWF Adapter Tests ──────────────────────────────────────────────────────


class TestAWFAdapter:
    """Tests for the AWF (.brain/ directory) adapter."""

    @pytest.fixture()
    def brain_dir(self, tmp_path: Any) -> Any:
        """Create a temporary .brain/ directory with test data."""
        brain_dir = tmp_path / ".brain"
        brain_dir.mkdir()

        brain_json = {
            "project": {"name": "MyApp", "tech_stack": ["Next.js", "Prisma"]},
            "key_decisions": [
                {"decision": "Use NextAuth", "reason": "Simple, team familiar"},
                {"decision": "PostgreSQL over MySQL", "reason": "Better JSON support"},
            ],
        }
        (brain_dir / "brain.json").write_text(
            __import__("json").dumps(brain_json), encoding="utf-8"
        )

        session_json = {
            "working_on": {"feature": "Authentication", "task": "Login form", "progress": 65},
            "errors_history": [
                {"error": "CORS", "fixed": True},
                {"error": "TypeScript strict mode", "fixed": False},
            ],
            "conversation_summary": [
                "Discussed auth options, picked NextAuth",
            ],
            "recent_files": [
                "src/app/login/page.tsx",
            ],
        }
        (brain_dir / "session.json").write_text(
            __import__("json").dumps(session_json), encoding="utf-8"
        )

        snapshots_dir = brain_dir / "snapshots"
        snapshots_dir.mkdir()
        snapshot = {"summary": "Sprint 1 complete", "notes": "All tests passing"}
        (snapshots_dir / "2024-01-15_1430.json").write_text(
            __import__("json").dumps(snapshot), encoding="utf-8"
        )

        return brain_dir

    @pytest.mark.asyncio()
    async def test_fetch_all(self, brain_dir: Any) -> None:
        """Test fetching all tiers."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        records = await adapter.fetch_all()

        # brain.json: 1 project + 2 decisions = 3
        # session.json: 1 working_on + 2 errors + 1 summary + 1 file = 5
        # snapshots: 2 entries from snapshot = 2
        assert len(records) == 10
        assert all(r.source_system == "awf" for r in records)

    @pytest.mark.asyncio()
    async def test_fetch_tier1_only(self, brain_dir: Any) -> None:
        """Test filtering to tier1 (brain.json) only."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        records = await adapter.fetch_all(collection="tier1")

        assert len(records) == 3  # 1 project + 2 decisions
        assert all("tier:1" in r.tags for r in records)

    @pytest.mark.asyncio()
    async def test_fetch_tier2_only(self, brain_dir: Any) -> None:
        """Test filtering to tier2 (session.json) only."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        records = await adapter.fetch_all(collection="tier2")

        assert len(records) == 5  # 1 working_on + 2 errors + 1 summary + 1 file
        assert all("tier:2" in r.tags for r in records)

    @pytest.mark.asyncio()
    async def test_brain_json_parsing(self, brain_dir: Any) -> None:
        """Test brain.json content is correctly parsed."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        records = await adapter.fetch_all(collection="tier1")

        project_record = next(r for r in records if "project" in r.tags)
        assert "MyApp" in project_record.content
        assert "Next.js" in project_record.content
        assert project_record.source_type == "fact"

        decisions = [r for r in records if "decision" in r.tags]
        assert len(decisions) == 2
        assert "NextAuth" in decisions[0].content
        assert decisions[0].source_type == "decision"

    @pytest.mark.asyncio()
    async def test_session_json_parsing(self, brain_dir: Any) -> None:
        """Test session.json content is correctly parsed."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        records = await adapter.fetch_all(collection="tier2")

        working = next(r for r in records if "working_on" in r.tags)
        assert "Authentication" in working.content
        assert "65%" in working.content
        assert working.source_type == "context"

        errors = [r for r in records if "error" in r.tags]
        assert len(errors) == 2
        assert "(fixed)" in errors[0].content
        assert "(fixed)" not in errors[1].content

    @pytest.mark.asyncio()
    async def test_limit(self, brain_dir: Any) -> None:
        """Test limit parameter."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        records = await adapter.fetch_all(limit=2)
        assert len(records) == 2

    @pytest.mark.asyncio()
    async def test_health_check_healthy(self, brain_dir: Any) -> None:
        """Test health check on valid .brain/ dir."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        result = await adapter.health_check()

        assert result["healthy"] is True
        assert "MyApp" in result["message"]

    @pytest.mark.asyncio()
    async def test_health_check_missing_dir(self, tmp_path: Any) -> None:
        """Test that constructing AWFAdapter with nonexistent dir raises ValueError."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        with pytest.raises(ValueError, match="does not exist"):
            AWFAdapter(brain_dir=tmp_path / "nonexistent")

    @pytest.mark.asyncio()
    async def test_empty_brain_dir(self, tmp_path: Any) -> None:
        """Test fetching from empty .brain/ directory."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        brain_dir = tmp_path / ".brain"
        brain_dir.mkdir()

        adapter = AWFAdapter(brain_dir=brain_dir)
        records = await adapter.fetch_all()
        assert records == []

    @pytest.mark.asyncio()
    async def test_fetch_since_not_supported(self, brain_dir: Any) -> None:
        """Test that fetch_since raises NotImplementedError."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        with pytest.raises(NotImplementedError):
            await adapter.fetch_since(since=utcnow())

    def test_properties(self, brain_dir: Any) -> None:
        """Test adapter properties."""
        from neural_memory.integration.adapters.awf_adapter import AWFAdapter

        adapter = AWFAdapter(brain_dir=brain_dir)
        assert adapter.system_name == "awf"
        assert adapter.system_type == SourceSystemType.FILE_STORE
        assert SourceCapability.FETCH_ALL in adapter.capabilities
        assert SourceCapability.HEALTH_CHECK in adapter.capabilities
        assert SourceCapability.FETCH_SINCE not in adapter.capabilities


# ── Cognee Adapter Tests ─────────────────────────────────────────────────


class TestCogneeAdapter:
    """Tests for the Cognee knowledge graph adapter."""

    @pytest.fixture(autouse=True)
    def _mock_cognee(self, monkeypatch: Any) -> None:
        """Mock cognee module so no real connection is needed."""
        import sys
        import types

        mock_cognee = types.ModuleType("cognee")
        mock_search_module = types.ModuleType("cognee.api.v1.search")

        class MockSearchType:
            CHUNKS = "chunks"

        mock_search_module.SearchType = MockSearchType  # type: ignore[attr-defined]

        class MockConfig:
            @staticmethod
            def set_llm_api_key(key: str) -> None:
                pass

        mock_cognee.config = MockConfig()  # type: ignore[attr-defined]

        # Default: return sample chunks
        self._search_results: list[dict[str, Any]] = [
            {
                "id": "chunk-1",
                "text": "Python is a programming language",
                "edges": [
                    {
                        "source_id": "chunk-1",
                        "target_id": "chunk-2",
                        "relationship_type": "related_to",
                        "weight": 0.7,
                    }
                ],
            },
            {
                "id": "chunk-2",
                "text": "Django is a Python web framework",
                "edges": [],
            },
        ]

        async def mock_search(**kwargs: Any) -> list[dict[str, Any]]:
            return self._search_results

        mock_cognee.search = mock_search  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "cognee", mock_cognee)
        monkeypatch.setitem(sys.modules, "cognee.api", types.ModuleType("cognee.api"))
        monkeypatch.setitem(sys.modules, "cognee.api.v1", types.ModuleType("cognee.api.v1"))
        monkeypatch.setitem(sys.modules, "cognee.api.v1.search", mock_search_module)

    @pytest.mark.asyncio()
    async def test_fetch_all(self) -> None:
        """Test fetching all chunks from Cognee."""
        from neural_memory.integration.adapters.cognee_adapter import CogneeAdapter

        adapter = CogneeAdapter(api_key="test-key")
        records = await adapter.fetch_all()

        assert len(records) == 2
        assert records[0].source_system == "cognee"
        assert records[0].content == "Python is a programming language"
        assert records[0].source_type == "knowledge"
        assert "import:cognee" in records[0].tags

    @pytest.mark.asyncio()
    async def test_relationships_extracted(self) -> None:
        """Test that graph edges become ExternalRelationship tuples."""
        from neural_memory.integration.adapters.cognee_adapter import CogneeAdapter

        adapter = CogneeAdapter(api_key="test-key")
        records = await adapter.fetch_all()

        # First chunk has edges
        assert len(records[0].relationships) == 1
        rel = records[0].relationships[0]
        assert rel.source_record_id == "chunk-1"
        assert rel.target_record_id == "chunk-2"
        assert rel.relation_type == "related_to"
        assert rel.weight == 0.7

    @pytest.mark.asyncio()
    async def test_fetch_all_with_limit(self) -> None:
        """Test limit parameter."""
        from neural_memory.integration.adapters.cognee_adapter import CogneeAdapter

        adapter = CogneeAdapter(api_key="test-key")
        records = await adapter.fetch_all(limit=1)

        assert len(records) == 1

    @pytest.mark.asyncio()
    async def test_fetch_since_not_supported(self) -> None:
        """Test that fetch_since raises NotImplementedError."""
        from neural_memory.integration.adapters.cognee_adapter import CogneeAdapter

        adapter = CogneeAdapter(api_key="test-key")
        with pytest.raises(NotImplementedError):
            await adapter.fetch_since(since=utcnow())

    @pytest.mark.asyncio()
    async def test_health_check(self) -> None:
        """Test health check succeeds with mock."""
        from neural_memory.integration.adapters.cognee_adapter import CogneeAdapter

        adapter = CogneeAdapter(api_key="test-key")
        result = await adapter.health_check()

        assert result["healthy"] is True
        assert result["system"] == "cognee"

    def test_properties(self) -> None:
        """Test adapter properties."""
        from neural_memory.integration.adapters.cognee_adapter import CogneeAdapter

        adapter = CogneeAdapter(api_key="test-key")
        assert adapter.system_name == "cognee"
        assert adapter.system_type == SourceSystemType.GRAPH_STORE
        assert SourceCapability.FETCH_ALL in adapter.capabilities
        assert SourceCapability.FETCH_RELATIONSHIPS in adapter.capabilities
        assert SourceCapability.HEALTH_CHECK in adapter.capabilities
        assert SourceCapability.FETCH_SINCE not in adapter.capabilities


# ── Graphiti Adapter Tests ────────────────────────────────────────────────


class TestGraphitiAdapter:
    """Tests for the Graphiti bi-temporal graph adapter."""

    @pytest.fixture(autouse=True)
    def _mock_graphiti(self, monkeypatch: Any) -> None:
        """Mock graphiti_core module."""
        import sys
        import types

        mock_graphiti_core = types.ModuleType("graphiti_core")

        self._mock_nodes = [
            types.SimpleNamespace(
                id="node-1",
                name="Authentication",
                summary="User authentication system",
                created_at=datetime(2024, 6, 1, 10, 0),
                group_id="g1",
            ),
            types.SimpleNamespace(
                id="node-2",
                name="Authorization",
                summary="Role-based access control",
                created_at=datetime(2024, 6, 5, 14, 0),
                group_id="g1",
            ),
        ]

        self._mock_episodes = [
            types.SimpleNamespace(
                id="ep-1",
                content="Auth module handles JWT tokens",
                name="uses",
                created_at=datetime(2024, 6, 2, 12, 0),
                source_node_id="node-1",
                target_node_id="node-2",
                valid_at=datetime(2024, 6, 1),
                invalid_at=None,
            ),
            types.SimpleNamespace(
                id="ep-2",
                content="Old password hashing was MD5",
                name="replaced_by",
                created_at=datetime(2024, 5, 1, 8, 0),
                source_node_id="node-1",
                target_node_id="node-2",
                valid_at=datetime(2024, 3, 1),
                invalid_at=datetime(2024, 5, 15),
            ),
        ]

        class MockGraphiti:
            def __init__(self, uri: str = "") -> None:
                self.uri = uri

            async def retrieve_nodes(
                self, query: str = "", num_results: int = 10, **kw: Any
            ) -> list[Any]:
                return self._mock_nodes  # type: ignore[attr-defined]

            async def retrieve_episodes(
                self, query: str = "", num_results: int = 10, **kw: Any
            ) -> list[Any]:
                return self._mock_episodes  # type: ignore[attr-defined]

        # Attach mock data to the class so tests can access it
        MockGraphiti._mock_nodes = self._mock_nodes  # type: ignore[attr-defined]
        MockGraphiti._mock_episodes = self._mock_episodes  # type: ignore[attr-defined]

        mock_graphiti_core.Graphiti = MockGraphiti  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "graphiti_core", mock_graphiti_core)

    @pytest.mark.asyncio()
    async def test_fetch_all(self) -> None:
        """Test fetching all nodes and episodes."""
        from neural_memory.integration.adapters.graphiti_adapter import GraphitiAdapter

        adapter = GraphitiAdapter(uri="bolt://test:7687")
        records = await adapter.fetch_all()

        # 2 nodes + 2 episodes = 4
        assert len(records) == 4
        assert all(r.source_system == "graphiti" for r in records)

        # Check node records
        entities = [r for r in records if r.source_type == "entity"]
        assert len(entities) == 2
        assert entities[0].content == "User authentication system"
        assert "import:graphiti" in entities[0].tags
        assert "entity" in entities[0].tags

        # Check episode records
        episodes = [r for r in records if r.source_type == "episode"]
        assert len(episodes) == 2

    @pytest.mark.asyncio()
    async def test_relationships_with_temporal_weight(self) -> None:
        """Test that valid_at/invalid_at affects relationship weight."""
        from neural_memory.integration.adapters.graphiti_adapter import GraphitiAdapter

        adapter = GraphitiAdapter(uri="bolt://test:7687")
        records = await adapter.fetch_all()

        episodes = [r for r in records if r.source_type == "episode"]

        # ep-1: valid_at set, invalid_at=None → weight 0.8
        assert len(episodes[0].relationships) == 1
        assert episodes[0].relationships[0].weight == 0.8

        # ep-2: has invalid_at → weight 0.3
        assert len(episodes[1].relationships) == 1
        assert episodes[1].relationships[0].weight == 0.3

    @pytest.mark.asyncio()
    async def test_fetch_since(self) -> None:
        """Test temporal filtering with fetch_since."""
        from neural_memory.integration.adapters.graphiti_adapter import GraphitiAdapter

        adapter = GraphitiAdapter(uri="bolt://test:7687")
        # Only nodes created after June 3 should appear
        records = await adapter.fetch_since(since=datetime(2024, 6, 3))

        # node-2 (June 5) passes filter, node-1 (June 1) does not
        # Both episodes are returned by retrieve_episodes
        entities = [r for r in records if r.source_type == "entity"]
        assert len(entities) == 1
        assert entities[0].content == "Role-based access control"

    @pytest.mark.asyncio()
    async def test_fetch_all_with_limit(self) -> None:
        """Test limit parameter."""
        from neural_memory.integration.adapters.graphiti_adapter import GraphitiAdapter

        adapter = GraphitiAdapter(uri="bolt://test:7687")
        records = await adapter.fetch_all(limit=2)

        assert len(records) == 2

    @pytest.mark.asyncio()
    async def test_health_check(self) -> None:
        """Test health check with mock."""
        from neural_memory.integration.adapters.graphiti_adapter import GraphitiAdapter

        adapter = GraphitiAdapter(uri="bolt://test:7687")
        result = await adapter.health_check()

        assert result["healthy"] is True
        assert result["system"] == "graphiti"

    def test_properties(self) -> None:
        """Test adapter properties."""
        from neural_memory.integration.adapters.graphiti_adapter import GraphitiAdapter

        adapter = GraphitiAdapter(uri="bolt://test:7687")
        assert adapter.system_name == "graphiti"
        assert adapter.system_type == SourceSystemType.GRAPH_STORE
        assert SourceCapability.FETCH_ALL in adapter.capabilities
        assert SourceCapability.FETCH_SINCE in adapter.capabilities
        assert SourceCapability.FETCH_RELATIONSHIPS in adapter.capabilities
        assert SourceCapability.HEALTH_CHECK in adapter.capabilities

    def test_temporal_weight_computation(self) -> None:
        """Test the static temporal weight helper."""
        from neural_memory.integration.adapters.graphiti_adapter import GraphitiAdapter

        # Currently valid (no invalid_at)
        assert GraphitiAdapter._compute_temporal_weight(datetime(2024, 1, 1), None) == 0.8
        # Expired
        assert (
            GraphitiAdapter._compute_temporal_weight(datetime(2024, 1, 1), datetime(2024, 6, 1))
            == 0.3
        )
        # Unknown
        assert GraphitiAdapter._compute_temporal_weight(None, None) == 0.5


# ── LlamaIndex Adapter Tests ─────────────────────────────────────────────


class TestLlamaIndexAdapter:
    """Tests for the LlamaIndex index adapter."""

    @pytest.fixture(autouse=True)
    def _mock_llamaindex(self, monkeypatch: Any) -> None:
        """Mock llama_index modules."""
        import sys
        import types

        # Create mock node objects
        class MockTextNode:
            def __init__(
                self,
                node_id: str,
                text: str,
                metadata: dict[str, Any] | None = None,
                embedding: list[float] | None = None,
                relationships: dict[str, Any] | None = None,
            ) -> None:
                self.node_id = node_id
                self.text = text
                self.metadata = metadata or {}
                self.embedding = embedding
                self.relationships = relationships or {}

        self._mock_nodes = {
            "node-a": MockTextNode(
                node_id="node-a",
                text="Introduction to neural networks",
                metadata={"source": "textbook", "chapter": "1"},
                embedding=[0.1, 0.2, 0.3],
                relationships={
                    "CHILD": types.SimpleNamespace(node_id="node-b"),
                },
            ),
            "node-b": MockTextNode(
                node_id="node-b",
                text="Backpropagation algorithm details",
                metadata={"source": "textbook", "chapter": "2"},
                embedding=[0.4, 0.5, 0.6],
                relationships={
                    "PARENT": types.SimpleNamespace(node_id="node-a"),
                },
            ),
            "node-c": MockTextNode(
                node_id="node-c",
                text="Gradient descent optimization",
                metadata={"source": "textbook", "chapter": "3"},
            ),
        }

        class MockDocstore:
            def __init__(self, docs: dict[str, Any]) -> None:
                self.docs = docs

        class MockIndex:
            def __init__(self, docs: dict[str, Any]) -> None:
                self.docstore = MockDocstore(docs)

        self._MockIndex = MockIndex
        self._MockTextNode = MockTextNode

        # Mock the llama_index.core module
        mock_core = types.ModuleType("llama_index.core")
        mock_llama_index = types.ModuleType("llama_index")

        def mock_storage_context_from_defaults(persist_dir: str = "") -> Any:
            return types.SimpleNamespace(persist_dir=persist_dir)

        def mock_load_index_from_storage(storage_context: Any) -> MockIndex:
            return MockIndex(self._mock_nodes)

        mock_core.StorageContext = types.SimpleNamespace(  # type: ignore[attr-defined]
            from_defaults=staticmethod(mock_storage_context_from_defaults)
        )
        mock_core.load_index_from_storage = mock_load_index_from_storage  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "llama_index", mock_llama_index)
        monkeypatch.setitem(sys.modules, "llama_index.core", mock_core)

    @pytest.mark.asyncio()
    async def test_fetch_all(self) -> None:
        """Test fetching all nodes from LlamaIndex."""
        from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(persist_dir="/fake/path")
        records = await adapter.fetch_all()

        assert len(records) == 3
        assert all(r.source_system == "llamaindex" for r in records)
        assert "import:llamaindex" in records[0].tags

    @pytest.mark.asyncio()
    async def test_embeddings_preserved(self) -> None:
        """Test that node embeddings are preserved in records."""
        from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(persist_dir="/fake/path")
        records = await adapter.fetch_all()

        record_a = next(r for r in records if r.id == "node-a")
        assert record_a.embedding == [0.1, 0.2, 0.3]

        record_c = next(r for r in records if r.id == "node-c")
        assert record_c.embedding is None

    @pytest.mark.asyncio()
    async def test_node_relationships(self) -> None:
        """Test that parent/child relationships are extracted."""
        from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(persist_dir="/fake/path")
        records = await adapter.fetch_all()

        record_a = next(r for r in records if r.id == "node-a")
        assert len(record_a.relationships) == 1
        assert record_a.relationships[0].relation_type == "child_of"
        assert record_a.relationships[0].target_record_id == "node-b"

        record_b = next(r for r in records if r.id == "node-b")
        assert len(record_b.relationships) == 1
        assert record_b.relationships[0].relation_type == "parent_of"

    @pytest.mark.asyncio()
    async def test_fetch_since_not_supported(self) -> None:
        """Test that fetch_since raises NotImplementedError."""
        from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(persist_dir="/fake/path")
        with pytest.raises(NotImplementedError):
            await adapter.fetch_since(since=utcnow())

    @pytest.mark.asyncio()
    async def test_health_check(self) -> None:
        """Test health check with mock index."""
        from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(persist_dir="/fake/path")
        result = await adapter.health_check()

        assert result["healthy"] is True
        assert "3 documents" in result["message"]
        assert result["system"] == "llamaindex"

    def test_properties(self) -> None:
        """Test adapter properties."""
        from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter(persist_dir="/fake/path")
        assert adapter.system_name == "llamaindex"
        assert adapter.system_type == SourceSystemType.INDEX_STORE
        assert SourceCapability.FETCH_ALL in adapter.capabilities
        assert SourceCapability.FETCH_EMBEDDINGS in adapter.capabilities
        assert SourceCapability.HEALTH_CHECK in adapter.capabilities
        assert SourceCapability.FETCH_SINCE not in adapter.capabilities

    @pytest.mark.asyncio()
    async def test_with_live_index(self) -> None:
        """Test adapter with a live index object instead of persist_dir."""
        from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

        mock_index = self._MockIndex(self._mock_nodes)
        adapter = LlamaIndexAdapter(index=mock_index)
        records = await adapter.fetch_all()

        assert len(records) == 3

    def test_missing_config_raises(self) -> None:
        """Test that no persist_dir and no index raises ValueError."""
        from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter()
        with pytest.raises(ValueError, match="requires either"):
            adapter._get_index()
