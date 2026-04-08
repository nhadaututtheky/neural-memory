"""Schema assimilation — bottom-up knowledge organization.

When enough memories cluster around a topic, a SCHEMA neuron emerges
that represents the shared pattern. New memories either assimilate
(fit the schema) or accommodate (force schema evolution).

Neuroscience basis: Piaget's assimilation/accommodation — new information
is either absorbed into existing mental models or forces their restructuring.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


class AssimilationAction(StrEnum):
    NO_SCHEMA = "no_schema"
    SCHEMA_CREATED = "schema_created"
    ASSIMILATED = "assimilated"
    ACCOMMODATED = "accommodated"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class AssimilationResult:
    """Result of schema assimilation for a memory."""

    action: AssimilationAction
    schema_id: str | None = None
    version: int = 0


async def assimilate_or_accommodate(
    new_neuron: Neuron,
    storage: NeuralStorage,
    config: BrainConfig,
) -> AssimilationResult:
    """Check if a new memory fits, creates, or contradicts a schema.

    Args:
        new_neuron: The newly encoded anchor neuron
        storage: Storage backend
        config: Brain configuration

    Returns:
        AssimilationResult describing the action taken
    """
    enabled = getattr(config, "schema_assimilation_enabled", False)
    if not isinstance(enabled, bool):
        enabled = False
    if not enabled:
        return AssimilationResult(action=AssimilationAction.SKIPPED)

    min_cluster = getattr(config, "schema_min_cluster_size", 10)
    min_cluster = int(min_cluster) if isinstance(min_cluster, (int, float)) else 10

    # Extract tags from the new neuron
    new_tags = set(new_neuron.metadata.get("tags", [])) if new_neuron.metadata else set()
    if not new_tags:
        return AssimilationResult(action=AssimilationAction.NO_SCHEMA)

    # Find existing SCHEMA neurons with overlapping tags
    schema_neurons = await storage.find_neurons(
        type=NeuronType.SCHEMA,
        limit=50,
    )

    matching_schema: Neuron | None = None
    best_overlap = 0.0

    for schema in schema_neurons:
        schema_tags = set(schema.metadata.get("tags", [])) if schema.metadata else set()
        if not schema_tags:
            continue
        overlap = len(new_tags & schema_tags) / len(new_tags | schema_tags)
        if overlap > best_overlap and overlap > 0.3:
            best_overlap = overlap
            matching_schema = schema

    if matching_schema is not None:
        # Check for contradiction
        from neural_memory.engine.prediction_error import _detects_reversal

        if _detects_reversal(new_neuron.content, matching_schema.content):
            # ACCOMMODATE: create new schema version
            return await _accommodate(new_neuron, matching_schema, new_tags, storage)
        # ASSIMILATE: link to existing schema
        return await _assimilate(new_neuron, matching_schema, storage)

    # No matching schema — check if cluster is large enough to create one
    domain_neurons = await _find_neurons_by_tags(
        storage,
        new_tags,
        min_matches=min_cluster,
        max_results=min_cluster + 5,
    )

    if len(domain_neurons) < min_cluster:
        return AssimilationResult(action=AssimilationAction.NO_SCHEMA)

    # CREATE new schema
    return await _create_schema(domain_neurons, new_tags, storage)


async def _assimilate(
    neuron: Neuron,
    schema: Neuron,
    storage: NeuralStorage,
) -> AssimilationResult:
    """Link a neuron to an existing schema via IS_A synapse."""
    synapse = Synapse.create(
        source_id=neuron.id,
        target_id=schema.id,
        type=SynapseType.IS_A,
        weight=0.6,
    )
    await storage.add_synapse(synapse)

    # Store schema ID on neuron for stratum MMR diversity tracking
    meta = dict(neuron.metadata) if neuron.metadata else {}
    if meta.get("_schema_id") != schema.id:
        meta["_schema_id"] = schema.id
        updated = neuron.with_metadata(_schema_id=schema.id)
        try:
            await storage.update_neuron(updated)
        except Exception:
            logger.debug("Schema ID metadata update skipped for %s", neuron.id[:12])

    logger.debug("Assimilated neuron %s into schema %s", neuron.id[:12], schema.id[:12])

    version = (schema.metadata or {}).get("schema_version", 1)
    return AssimilationResult(
        action=AssimilationAction.ASSIMILATED,
        schema_id=schema.id,
        version=int(version) if isinstance(version, (int, float)) else 1,
    )


async def _accommodate(
    neuron: Neuron,
    old_schema: Neuron,
    tags: set[str],
    storage: NeuralStorage,
) -> AssimilationResult:
    """Create a new schema version when contradiction detected."""
    old_version = (old_schema.metadata or {}).get("schema_version", 1)
    old_version = int(old_version) if isinstance(old_version, (int, float)) else 1
    new_version = old_version + 1

    # Create new schema neuron
    shared = _extract_shared_entities([neuron.content, old_schema.content])
    summary = f"Schema v{new_version}: {', '.join(shared[:5]) or 'updated model'} (accommodated)"

    new_schema = Neuron.create(
        content=summary,
        type=NeuronType.SCHEMA,
        metadata={
            "tags": sorted(tags),
            "schema_version": new_version,
            "accommodation_trigger": neuron.id,
            "previous_schema": old_schema.id,
        },
    )
    await storage.add_neuron(new_schema)

    # Link new schema SUPERSEDES old
    supersedes = Synapse.create(
        source_id=new_schema.id,
        target_id=old_schema.id,
        type=SynapseType.SUPERSEDES,
        weight=0.9,
    )
    await storage.add_synapse(supersedes)

    # Link triggering neuron to new schema
    is_a = Synapse.create(
        source_id=neuron.id,
        target_id=new_schema.id,
        type=SynapseType.IS_A,
        weight=0.6,
    )
    await storage.add_synapse(is_a)

    logger.debug(
        "Accommodated: schema %s v%d → v%d (trigger: %s)",
        old_schema.id[:12],
        old_version,
        new_version,
        neuron.id[:12],
    )

    return AssimilationResult(
        action=AssimilationAction.ACCOMMODATED,
        schema_id=new_schema.id,
        version=new_version,
    )


async def _create_schema(
    neurons: list[Neuron],
    tags: set[str],
    storage: NeuralStorage,
) -> AssimilationResult:
    """Create a new SCHEMA neuron from a cluster of related memories."""
    shared = _extract_shared_entities([n.content for n in neurons])
    tag_str = ", ".join(sorted(tags)[:5])
    entity_str = ", ".join(shared[:5]) if shared else "common patterns"

    summary = f"Schema: {tag_str} — entities: {entity_str} ({len(neurons)} memories)"

    schema = Neuron.create(
        content=summary,
        type=NeuronType.SCHEMA,
        metadata={
            "tags": sorted(tags),
            "schema_version": 1,
            "cluster_size": len(neurons),
        },
    )
    await storage.add_neuron(schema)

    # Link cluster members to schema
    for neuron in neurons[:20]:  # Cap to avoid excessive synapses
        syn = Synapse.create(
            source_id=neuron.id,
            target_id=schema.id,
            type=SynapseType.IS_A,
            weight=0.4,
        )
        await storage.add_synapse(syn)

    logger.debug("Created schema %s for %d neurons (%s)", schema.id[:12], len(neurons), tag_str)

    return AssimilationResult(
        action=AssimilationAction.SCHEMA_CREATED,
        schema_id=schema.id,
        version=1,
    )


async def _find_neurons_by_tags(
    storage: NeuralStorage,
    tags: set[str],
    *,
    min_matches: int = 0,
    max_results: int = 500,
) -> list[Neuron]:
    """Paginated fetch of neurons matching any of the given tags.

    Fetches in pages of 1000 to avoid loading entire brain into memory,
    stops when we have enough matches or exhaust the brain.
    """
    page_size = 1000
    offset = 0
    matched: list[Neuron] = []

    while True:
        batch = await storage.find_neurons(limit=page_size, offset=offset)
        if not batch:
            break

        for n in batch:
            ntags = n.metadata.get("tags", []) if n.metadata else []
            if set(ntags) & tags:
                matched.append(n)
                if len(matched) >= max_results:
                    return matched

        # Early stop: we have enough matches and no minimum target
        if min_matches > 0 and len(matched) >= min_matches:
            return matched

        # If batch was smaller than page_size, we've exhausted the brain
        if len(batch) < page_size:
            break

        offset += page_size

    return matched


def _extract_shared_entities(contents: list[str]) -> list[str]:
    """Extract frequently occurring capitalized terms across contents."""
    word_counts: Counter[str] = Counter()
    for text in contents:
        # Find capitalized words (likely entities/proper nouns)
        words = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        word_counts.update(set(words))  # Count unique per document

    # Return words appearing in 2+ documents
    threshold = max(2, len(contents) // 3)
    return [w for w, c in word_counts.most_common(10) if c >= threshold]


async def batch_schema_assimilation(
    storage: NeuralStorage,
    config: BrainConfig,
    dry_run: bool = False,
) -> int:
    """Scan tag clusters and create schemas where appropriate.

    Used by consolidation strategy. Returns count of schemas created/updated.
    """
    enabled = getattr(config, "schema_assimilation_enabled", False)
    if not isinstance(enabled, bool):
        enabled = False
    if not enabled:
        return 0

    min_cluster = getattr(config, "schema_min_cluster_size", 10)
    min_cluster = int(min_cluster) if isinstance(min_cluster, (int, float)) else 10

    # Get existing schemas to avoid duplicates
    existing_schemas = await storage.find_neurons(
        type=NeuronType.SCHEMA,
        limit=200,
    )
    covered_tags: set[str] = set()
    for s in existing_schemas:
        covered_tags.update(s.metadata.get("tags", []) if s.metadata else [])

    # Find popular tags that don't have schemas yet (paginated to handle large brains)
    tag_counts: Counter[str] = Counter()
    tag_neurons: dict[str, list[Neuron]] = {}
    page_size = 1000
    offset = 0
    while True:
        batch = await storage.find_neurons(limit=page_size, offset=offset)
        if not batch:
            break
        for n in batch:
            ntags = n.metadata.get("tags", []) if n.metadata else []
            for t in ntags:
                if t not in covered_tags:
                    tag_counts[t] += 1
                    tag_neurons.setdefault(t, []).append(n)
        if len(batch) < page_size:
            break
        offset += page_size

    schemas_created = 0
    for tag, count in tag_counts.most_common(20):
        if count < min_cluster:
            break
        if dry_run:
            schemas_created += 1
            continue

        neurons = tag_neurons[tag][: min_cluster + 10]
        result = await _create_schema(neurons, {tag}, storage)
        if result.action == AssimilationAction.SCHEMA_CREATED:
            schemas_created += 1

    return schemas_created
