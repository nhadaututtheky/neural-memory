"""Data models for the external source integration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from neural_memory.utils.timeutils import utcnow


class SourceSystemType(StrEnum):
    """Categories of external memory systems."""

    VECTOR_DB = "vector_db"
    MEMORY_LAYER = "memory_layer"
    GRAPH_STORE = "graph_store"
    INDEX_STORE = "index_store"
    FILE_STORE = "file_store"


class SourceCapability(StrEnum):
    """Capabilities a source adapter may support."""

    FETCH_ALL = "fetch_all"
    FETCH_SINCE = "fetch_since"
    FETCH_EMBEDDINGS = "fetch_embeddings"
    FETCH_RELATIONSHIPS = "fetch_relationships"
    FETCH_METADATA = "fetch_metadata"
    HEALTH_CHECK = "health_check"


@dataclass(frozen=True)
class ExternalRelationship:
    """A relationship from an external graph system.

    Attributes:
        source_record_id: ID of the source record in the external system
        target_record_id: ID of the target record in the external system
        relation_type: The relationship type string from the external system
        weight: Relationship weight/confidence (0.0 - 1.0)
        metadata: Additional relationship-specific data
    """

    source_record_id: str
    target_record_id: str
    relation_type: str
    weight: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalRecord:
    """Normalized record from any external memory system.

    This is the universal intermediate representation that all SourceAdapters
    produce. The RecordMapper converts these into NeuralMemory structures.

    Attributes:
        id: Unique identifier within the source system
        source_system: Name of the source system (e.g., "chromadb", "mem0")
        source_collection: Collection/namespace within the source system
        content: The text content of this memory
        created_at: When this record was created in the source system
        updated_at: When this record was last modified
        source_type: Category hint from the source (e.g., "fact", "document")
        metadata: Original metadata from the source system
        embedding: Optional embedding vector (if available from vector DBs)
        tags: Tags from the source system
        relationships: Relationships to other records (from graph systems)
    """

    id: str
    source_system: str
    source_collection: str
    content: str
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime | None = None
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    relationships: tuple[ExternalRelationship, ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        id: str,
        source_system: str,
        content: str,
        source_collection: str = "default",
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        tags: set[str] | None = None,
        relationships: list[ExternalRelationship] | None = None,
    ) -> ExternalRecord:
        """Factory method to create an ExternalRecord."""
        return cls(
            id=id,
            source_system=source_system,
            source_collection=source_collection,
            content=content,
            created_at=created_at or utcnow(),
            updated_at=updated_at,
            source_type=source_type,
            metadata=metadata or {},
            embedding=embedding,
            tags=frozenset(tags) if tags else frozenset(),
            relationships=tuple(relationships) if relationships else (),
        )


@dataclass(frozen=True)
class SyncState:
    """Tracks the state of a sync operation for a given source.

    Attributes:
        source_system: Name of the source system
        source_collection: Collection within the source
        last_sync_at: When the last sync completed
        records_imported: Total records imported so far
        last_record_id: ID of the last imported record
        metadata: Additional sync state data
    """

    source_system: str
    source_collection: str
    last_sync_at: datetime | None = None
    records_imported: int = 0
    last_record_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_update(
        self,
        last_sync_at: datetime | None = None,
        records_imported: int | None = None,
        last_record_id: str | None = None,
    ) -> SyncState:
        """Create updated SyncState (immutable pattern)."""
        return SyncState(
            source_system=self.source_system,
            source_collection=self.source_collection,
            last_sync_at=last_sync_at or self.last_sync_at,
            records_imported=(
                records_imported if records_imported is not None else self.records_imported
            ),
            last_record_id=last_record_id if last_record_id is not None else self.last_record_id,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ImportResult:
    """Result of importing records from an external source.

    Attributes:
        source_system: Name of the source system
        source_collection: Collection within the source
        records_fetched: Number of records fetched from source
        records_imported: Number of records successfully imported
        records_skipped: Number of records skipped (duplicates, errors)
        records_failed: Number of records that failed to import
        errors: List of error messages for failed records
        duration_seconds: Time taken for the import operation
        fibers_created: IDs of fibers created during import
    """

    source_system: str
    source_collection: str
    records_fetched: int = 0
    records_imported: int = 0
    records_skipped: int = 0
    records_failed: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)
    duration_seconds: float = 0.0
    fibers_created: tuple[str, ...] = field(default_factory=tuple)
