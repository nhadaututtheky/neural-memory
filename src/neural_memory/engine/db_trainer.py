"""DB-to-Brain training pipeline — train brains from database schemas.

Extracts SCHEMA KNOWLEDGE (not raw data) and encodes it into a brain:
- Table descriptions as CONCEPT neurons
- FK relationships as typed synapses (IS_A, INVOLVES, etc.)
- Schema patterns as insight neurons (audit trail, soft delete, etc.)

Mirrors DocTrainer architecture: batch save, error isolation per table,
shared domain neuron, optional ENRICH consolidation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.db_introspector import SchemaIntrospector
from neural_memory.engine.db_knowledge import KnowledgeExtractor
from neural_memory.engine.encoder import MemoryEncoder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class DBTrainingConfig:
    """Configuration for DB-to-Brain training.

    Attributes:
        connection_string: Database connection (e.g., "sqlite:///path/to/db").
        domain_tag: Tag applied to all schema knowledge (e.g., "ecommerce").
        brain_name: Target brain name (empty = current brain).
        consolidate: Run ENRICH consolidation after encoding.
        salience_ceiling: Cap initial fiber salience (imported knowledge
            starts weaker than organic memories).
        initial_stage: Maturation stage ("episodic" = skip STM/WORKING).
        include_patterns: Detect and encode schema patterns.
        include_relationships: Create FK-based relationship synapses.
        max_tables: Maximum tables to process (guard against huge schemas).
    """

    connection_string: str
    domain_tag: str = ""
    brain_name: str = ""
    consolidate: bool = True
    salience_ceiling: float = 0.5
    initial_stage: str = "episodic"
    include_patterns: bool = True
    include_relationships: bool = True
    max_tables: int = 100


@dataclass(frozen=True)
class DBTrainingResult:
    """Result of a DB-to-Brain training run.

    Attributes:
        tables_processed: Tables successfully encoded as entities.
        tables_skipped: Tables skipped due to max_tables limit.
        columns_processed: Total columns across all processed tables.
        relationships_mapped: FK relationships encoded as synapses.
        patterns_detected: Schema patterns detected and encoded.
        neurons_created: Total neurons created.
        synapses_created: Total synapses created (encode + relationships).
        enrichment_synapses: Synapses from ENRICH consolidation.
        schema_fingerprint: SHA256 fingerprint for re-training detection.
        brain_name: Name of the trained brain.
    """

    tables_processed: int
    tables_skipped: int = 0
    columns_processed: int = 0
    relationships_mapped: int = 0
    patterns_detected: int = 0
    neurons_created: int = 0
    synapses_created: int = 0
    enrichment_synapses: int = 0
    schema_fingerprint: str = ""
    brain_name: str = "current"


class DBTrainer:
    """Trains a neural memory brain from database schema knowledge.

    NOT a data import pipeline. The differences:
    - Data import: copies rows into the brain (expensive, noisy).
    - Schema training: teaches the brain to UNDERSTAND the database
      structure (tables, relationships, patterns). Cheap, semantic.

    Mirrors DocTrainer architecture:
    - Batch mode (disable_auto_save -> process -> batch_save)
    - Error isolation per table (one failure doesn't abort batch)
    - Shared domain neuron (like session TIME neuron)
    - Optional ENRICH consolidation for cross-linking
    """

    def __init__(self, storage: NeuralStorage, config: BrainConfig) -> None:
        self._storage = storage
        self._config = config
        self._encoder = MemoryEncoder(storage, config)
        self._introspector = SchemaIntrospector()
        self._extractor = KnowledgeExtractor()

    async def train(self, training_config: DBTrainingConfig) -> DBTrainingResult:
        """Train brain from database schema.

        Pipeline:
        1. Introspect schema -> SchemaSnapshot
        2. Extract knowledge -> SchemaKnowledge
        3. Batch encode entities, patterns, relationships
        4. Optional ENRICH consolidation
        """
        tc = training_config

        # Step 1: Introspect
        try:
            snapshot = await self._introspector.introspect(tc.connection_string)
        except Exception as exc:
            logger.error("Schema introspection failed: %s", exc, exc_info=True)
            msg = f"Failed to introspect schema: {exc}"
            raise ValueError(msg) from exc

        # Step 2: Extract knowledge
        knowledge = self._extractor.extract(snapshot)

        # Step 3: Batch encode
        neurons_created = 0
        synapses_created = 0
        tables_processed = 0
        tables_skipped = 0
        columns_processed = 0

        self._storage.disable_auto_save()

        try:
            # Create domain CONCEPT neuron (shared anchor, like session TIME)
            domain_neuron = None
            if tc.domain_tag:
                domain_neuron = Neuron.create(
                    type=NeuronType.CONCEPT,
                    content=f"database schema: {tc.domain_tag}",
                    metadata={
                        "db_schema_domain": True,
                        "domain_tag": tc.domain_tag,
                        "schema_fingerprint": snapshot.schema_fingerprint,
                    },
                )
                await self._storage.add_neuron(domain_neuron)
                neurons_created += 1

            # Encode table entities (up to max_tables)
            table_neurons: dict[str, str] = {}  # table_name -> anchor_neuron_id
            for entity in knowledge.entities:
                if tables_processed >= tc.max_tables:
                    tables_skipped += 1
                    continue

                try:
                    tags: set[str] = {"db_schema", "table"}
                    if tc.domain_tag:
                        tags.add(tc.domain_tag)

                    metadata: dict[str, object] = {
                        "type": "reference",
                        "table_name": entity.table_name,
                        "db_schema": True,
                        "business_purpose": entity.business_purpose,
                        "knowledge_confidence": entity.confidence,
                    }

                    result = await self._encoder.encode(
                        content=entity.description,
                        tags=tags,
                        metadata=metadata,
                        skip_conflicts=True,
                        skip_time_neurons=True,
                        initial_stage=tc.initial_stage,
                        salience_ceiling=tc.salience_ceiling,
                    )

                    neurons_created += len(result.neurons_created)
                    synapses_created += len(result.synapses_created)
                    table_neurons[entity.table_name] = result.fiber.anchor_neuron_id
                    tables_processed += 1
                    columns_processed += len(
                        [
                            p
                            for p in knowledge.properties
                            if p.table_name == entity.table_name
                        ]
                    )

                    # Link to domain neuron
                    if domain_neuron:
                        synapse = Synapse.create(
                            source_id=domain_neuron.id,
                            target_id=result.fiber.anchor_neuron_id,
                            type=SynapseType.CONTAINS,
                            weight=0.8,
                            metadata={"db_domain_link": True},
                        )
                        await self._storage.add_synapse(synapse)
                        synapses_created += 1

                except Exception:
                    logger.warning(
                        "Failed to encode table %s",
                        entity.table_name,
                        exc_info=True,
                    )

            # Encode patterns
            patterns_encoded = 0
            if tc.include_patterns:
                for pattern in knowledge.patterns:
                    # Skip patterns for tables that weren't encoded
                    if pattern.table_name not in table_neurons:
                        continue

                    try:
                        tags = {
                            "db_schema",
                            "pattern",
                            pattern.pattern_type.value,
                        }
                        if tc.domain_tag:
                            tags.add(tc.domain_tag)

                        metadata = {
                            "type": "insight",
                            "pattern_type": pattern.pattern_type.value,
                            "table_name": pattern.table_name,
                            "db_schema_pattern": True,
                            "knowledge_confidence": pattern.confidence,
                        }

                        result = await self._encoder.encode(
                            content=pattern.description,
                            tags=tags,
                            metadata=metadata,
                            skip_conflicts=True,
                            skip_time_neurons=True,
                            initial_stage=tc.initial_stage,
                            salience_ceiling=tc.salience_ceiling,
                        )

                        neurons_created += len(result.neurons_created)
                        synapses_created += len(result.synapses_created)
                        patterns_encoded += 1

                        # Link pattern to its table
                        table_anchor = table_neurons.get(pattern.table_name)
                        if table_anchor:
                            synapse = Synapse.create(
                                source_id=result.fiber.anchor_neuron_id,
                                target_id=table_anchor,
                                type=SynapseType.HAS_PROPERTY,
                                weight=pattern.confidence,
                                metadata={"db_pattern_link": True},
                            )
                            await self._storage.add_synapse(synapse)
                            synapses_created += 1

                    except Exception:
                        logger.warning(
                            "Failed to encode pattern %s for %s",
                            pattern.pattern_type,
                            pattern.table_name,
                            exc_info=True,
                        )

            # Create relationship synapses (direct, not via encoder)
            relationships_mapped = 0
            if tc.include_relationships:
                for rel in knowledge.relationships:
                    source_anchor = table_neurons.get(rel.source_table)
                    target_anchor = table_neurons.get(rel.target_table)

                    if source_anchor and target_anchor:
                        try:
                            synapse = Synapse.create(
                                source_id=source_anchor,
                                target_id=target_anchor,
                                type=rel.synapse_type,
                                weight=rel.confidence,
                                metadata={
                                    "fk_column": rel.source_column,
                                    "fk_target": rel.target_column,
                                    "db_relationship": True,
                                    "knowledge_confidence": rel.confidence,
                                },
                            )
                            await self._storage.add_synapse(synapse)
                            synapses_created += 1
                            relationships_mapped += 1
                        except Exception:
                            logger.debug(
                                "Failed to create relationship synapse %s -> %s",
                                rel.source_table,
                                rel.target_table,
                                exc_info=True,
                            )

            # Batch save
            await self._storage.batch_save()

        finally:
            self._storage.enable_auto_save()

        # Run ENRICH consolidation if requested
        enrichment_synapses = 0
        if tc.consolidate and tables_processed > 0:
            enrichment_synapses = await self._run_enrichment()

        return DBTrainingResult(
            tables_processed=tables_processed,
            tables_skipped=tables_skipped,
            columns_processed=columns_processed,
            relationships_mapped=relationships_mapped,
            patterns_detected=patterns_encoded,
            neurons_created=neurons_created,
            synapses_created=synapses_created,
            enrichment_synapses=enrichment_synapses,
            schema_fingerprint=snapshot.schema_fingerprint,
            brain_name=tc.brain_name or "current",
        )

    async def _run_enrichment(self) -> int:
        """Run ENRICH consolidation to create cross-cluster links."""
        from neural_memory.engine.consolidation import (
            ConsolidationEngine,
            ConsolidationStrategy,
        )

        engine = ConsolidationEngine(self._storage)
        report = await engine.run(strategies=[ConsolidationStrategy.ENRICH])
        return report.synapses_enriched
