"""RecordMapper — converts ExternalRecord to NeuralMemory neural structures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    TypedMemory,
    suggest_memory_type,
)
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.encoder import EncodingResult, MemoryEncoder
from neural_memory.integration.models import ExternalRecord, ExternalRelationship

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Mapping from common external type strings to NeuralMemory MemoryType
_SOURCE_TYPE_MAP: dict[str, MemoryType] = {
    # Mem0 types
    "fact": MemoryType.FACT,
    "preference": MemoryType.PREFERENCE,
    "memory": MemoryType.FACT,
    "instruction": MemoryType.INSTRUCTION,
    # ChromaDB / generic
    "document": MemoryType.REFERENCE,
    "note": MemoryType.FACT,
    "code": MemoryType.REFERENCE,
    # Graphiti / graph types
    "entity": MemoryType.FACT,
    "relationship": MemoryType.FACT,
    "episode": MemoryType.CONTEXT,
    # LlamaIndex
    "text_node": MemoryType.REFERENCE,
    "index_node": MemoryType.REFERENCE,
    # Cognee
    "knowledge": MemoryType.INSIGHT,
    "concept": MemoryType.INSIGHT,
    # Generic fallbacks
    "todo": MemoryType.TODO,
    "decision": MemoryType.DECISION,
    "error": MemoryType.ERROR,
    "workflow": MemoryType.WORKFLOW,
    "insight": MemoryType.INSIGHT,
    "context": MemoryType.CONTEXT,
    "reference": MemoryType.REFERENCE,
}

# Mapping from external relationship type strings to SynapseType
_RELATION_TYPE_MAP: dict[str, SynapseType] = {
    "related_to": SynapseType.RELATED_TO,
    "similar_to": SynapseType.SIMILAR_TO,
    "caused_by": SynapseType.CAUSED_BY,
    "leads_to": SynapseType.LEADS_TO,
    "is_a": SynapseType.IS_A,
    "has_property": SynapseType.HAS_PROPERTY,
    "involves": SynapseType.INVOLVES,
    "before": SynapseType.BEFORE,
    "after": SynapseType.AFTER,
    "co_occurs": SynapseType.CO_OCCURS,
    "at_location": SynapseType.AT_LOCATION,
    "contains": SynapseType.CONTAINS,
    "enables": SynapseType.ENABLES,
    "prevents": SynapseType.PREVENTS,
}


@dataclass(frozen=True)
class MappingResult:
    """Result of mapping an ExternalRecord to neural structures.

    Attributes:
        encoding_result: The result from MemoryEncoder.encode()
        typed_memory: The TypedMemory wrapper created
        external_record_id: Original ID from the source system
        source_system: Name of the source system
    """

    encoding_result: EncodingResult
    typed_memory: TypedMemory
    external_record_id: str
    source_system: str


class RecordMapper:
    """Maps ExternalRecord instances to NeuralMemory structures.

    Uses the existing MemoryEncoder pipeline to create neurons, synapses,
    and fibers, then wraps the result with TypedMemory and provenance.
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
        encoder: MemoryEncoder | None = None,
    ) -> None:
        self._storage = storage
        self._config = config
        self._encoder = encoder or MemoryEncoder(storage, config)

    def _resolve_memory_type(self, record: ExternalRecord) -> MemoryType:
        """Determine the MemoryType for an external record.

        Priority: explicit source_type mapping, then content heuristic.
        """
        if record.source_type:
            mapped = _SOURCE_TYPE_MAP.get(record.source_type.lower())
            if mapped is not None:
                return mapped
        return suggest_memory_type(record.content)

    def _resolve_synapse_type(self, relation_type: str) -> SynapseType:
        """Map an external relationship type string to SynapseType."""
        return _RELATION_TYPE_MAP.get(
            relation_type.lower().replace(" ", "_"),
            SynapseType.RELATED_TO,
        )

    def _build_metadata(self, record: ExternalRecord) -> dict[str, Any]:
        """Build metadata dict preserving source info."""
        meta: dict[str, Any] = {
            "import_source": record.source_system,
            "import_collection": record.source_collection,
            "import_record_id": record.id,
        }
        if record.embedding is not None:
            meta["embedding_dim"] = len(record.embedding)
            # Store fingerprint (first/last 4 values) for lightweight identification
            if len(record.embedding) > 8:
                meta["embedding_fingerprint"] = record.embedding[:4] + record.embedding[-4:]
        if record.metadata:
            for key, value in record.metadata.items():
                meta[f"src_{key}"] = value
        return meta

    async def map_record(self, record: ExternalRecord) -> MappingResult:
        """Map a single ExternalRecord to NeuralMemory structures.

        This:
        1. Runs the record through MemoryEncoder to create neurons/synapses/fiber
        2. Stores embedding as metadata on the anchor neuron (if present)
        3. Creates TypedMemory with provenance

        Args:
            record: The external record to map

        Returns:
            MappingResult with all created structures
        """
        # Check for sensitive content before importing
        from neural_memory.safety.sensitive import check_sensitive_content

        sensitive_matches = check_sensitive_content(record.content, min_severity=2)
        if sensitive_matches:
            types_found = sorted({m.type.value for m in sensitive_matches})
            logger.warning(
                "Skipping import record %s: sensitive content detected (%s)",
                record.id,
                ", ".join(types_found),
            )
            raise ValueError("Record rejected by content policy")

        tags = set(record.tags) | {
            f"import:{record.source_system}",
            f"collection:{record.source_collection}",
        }

        metadata = self._build_metadata(record)

        encoding_result = await self._encoder.encode(
            content=record.content,
            timestamp=record.created_at,
            metadata=metadata,
            tags=tags,
        )

        # Store full embedding vector on anchor neuron if present
        if record.embedding is not None:
            anchor = await self._storage.get_neuron(encoding_result.fiber.anchor_neuron_id)
            if anchor is not None:
                updated_anchor = anchor.with_metadata(embedding=record.embedding)
                await self._storage.update_neuron(updated_anchor)

        memory_type = self._resolve_memory_type(record)

        typed_memory = TypedMemory.create(
            fiber_id=encoding_result.fiber.id,
            memory_type=memory_type,
            priority=Priority.NORMAL,
            source=f"import:{record.source_system}",
            confidence=Confidence.MEDIUM,
            tags=tags,
            metadata={
                "import_source": record.source_system,
                "import_record_id": record.id,
                "import_collection": record.source_collection,
            },
        )
        await self._storage.add_typed_memory(typed_memory)

        return MappingResult(
            encoding_result=encoding_result,
            typed_memory=typed_memory,
            external_record_id=record.id,
            source_system=record.source_system,
        )

    async def create_relationship_synapses(
        self,
        record_to_fiber: dict[str, str],
        relationships: list[ExternalRelationship],
    ) -> list[Synapse]:
        """Create synapses from external relationships after all records are imported.

        Called as a second pass after all records have been mapped,
        because relationships reference other records that must already exist.

        Args:
            record_to_fiber: Mapping of external record ID -> fiber ID
            relationships: List of external relationships to convert

        Returns:
            List of created Synapse instances
        """
        created_synapses: list[Synapse] = []

        for rel in relationships:
            source_fiber_id = record_to_fiber.get(rel.source_record_id)
            target_fiber_id = record_to_fiber.get(rel.target_record_id)

            if source_fiber_id is None or target_fiber_id is None:
                continue

            source_fiber = await self._storage.get_fiber(source_fiber_id)
            target_fiber = await self._storage.get_fiber(target_fiber_id)

            if source_fiber is None or target_fiber is None:
                continue

            synapse_type = self._resolve_synapse_type(rel.relation_type)

            synapse = Synapse.create(
                source_id=source_fiber.anchor_neuron_id,
                target_id=target_fiber.anchor_neuron_id,
                type=synapse_type,
                weight=rel.weight,
                metadata={
                    "import_source": "external_relationship",
                    "original_type": rel.relation_type,
                    **rel.metadata,
                },
            )

            try:
                await self._storage.add_synapse(synapse)
                created_synapses.append(synapse)
            except ValueError:
                logger.debug(
                    "Synapse already exists between %s and %s",
                    source_fiber.anchor_neuron_id,
                    target_fiber.anchor_neuron_id,
                )

        return created_synapses
