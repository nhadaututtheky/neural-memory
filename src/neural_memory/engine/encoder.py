"""Memory encoder for converting experiences into neural structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.extraction.entities import EntityExtractor, EntityType
from neural_memory.extraction.keywords import extract_keywords, extract_weighted_keywords
from neural_memory.extraction.temporal import TemporalExtractor

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage


@dataclass
class EncodingResult:
    """
    Result of encoding a memory.

    Attributes:
        fiber: The created memory fiber
        neurons_created: List of newly created neurons
        neurons_linked: List of existing neuron IDs that were linked
        synapses_created: List of newly created synapses
    """

    fiber: Fiber
    neurons_created: list[Neuron]
    neurons_linked: list[str]
    synapses_created: list[Synapse]


class MemoryEncoder:
    """
    Encoder for converting experiences into neural structures.

    The encoder:
    1. Extracts neurons from content (time, entities, actions, concepts)
    2. Finds existing similar neurons for de-duplication
    3. Creates synapses based on relationships
    4. Bundles everything into a Fiber
    5. Auto-links with nearby temporal neurons
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
        temporal_extractor: TemporalExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        """
        Initialize the encoder.

        Args:
            storage: Storage backend
            config: Brain configuration
            temporal_extractor: Custom temporal extractor
            entity_extractor: Custom entity extractor
        """
        self._storage = storage
        self._config = config
        self._temporal = temporal_extractor or TemporalExtractor()
        self._entity = entity_extractor or EntityExtractor()

    async def encode(
        self,
        content: str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        tags: set[str] | None = None,
        language: str = "auto",
    ) -> EncodingResult:
        """
        Encode content into neural structures.

        Args:
            content: The text content to encode
            timestamp: When this memory occurred (default: now)
            metadata: Additional metadata to attach
            tags: Optional tags for the fiber
            language: Language hint ("vi", "en", or "auto")

        Returns:
            EncodingResult with created structures
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        neurons_created: list[Neuron] = []
        neurons_linked: list[str] = []
        synapses_created: list[Synapse] = []

        # 1. Extract time neurons
        time_neurons = await self._extract_time_neurons(content, timestamp)
        neurons_created.extend(time_neurons)

        # 2. Extract entity neurons
        entity_neurons = await self._extract_entity_neurons(content, language)
        neurons_created.extend(entity_neurons)

        # 3. Extract concept/keyword neurons
        concept_neurons = await self._extract_concept_neurons(content, language)
        neurons_created.extend(concept_neurons)

        # 4. Create the anchor neuron (main content)
        anchor_neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content=content,
            metadata={
                "is_anchor": True,
                "timestamp": timestamp.isoformat(),
                **(metadata or {}),
            },
        )
        await self._storage.add_neuron(anchor_neuron)
        neurons_created.append(anchor_neuron)

        # 5. Create synapses between neurons
        all_neurons = neurons_created

        # Connect anchor to time neurons
        for time_neuron in time_neurons:
            synapse = Synapse.create(
                source_id=anchor_neuron.id,
                target_id=time_neuron.id,
                type=SynapseType.HAPPENED_AT,
                weight=0.9,
            )
            await self._storage.add_synapse(synapse)
            synapses_created.append(synapse)

        # Connect anchor to entity neurons (weight by mention frequency)
        content_lower = content.lower()
        for entity_neuron in entity_neurons:
            mention_count = content_lower.count(entity_neuron.content.lower())
            entity_weight = min(0.95, 0.7 + 0.05 * mention_count)
            synapse = Synapse.create(
                source_id=anchor_neuron.id,
                target_id=entity_neuron.id,
                type=SynapseType.INVOLVES,
                weight=entity_weight,
            )
            await self._storage.add_synapse(synapse)
            synapses_created.append(synapse)

        # Connect anchor to concept neurons (weight by keyword importance)
        kw_weight_map = {
            kw.text: kw.weight for kw in extract_weighted_keywords(content, language=language)
        }
        for concept_neuron in concept_neurons:
            kw_weight = kw_weight_map.get(concept_neuron.content.lower(), 0.5)
            concept_weight = min(0.8, 0.4 + 0.3 * kw_weight)
            synapse = Synapse.create(
                source_id=anchor_neuron.id,
                target_id=concept_neuron.id,
                type=SynapseType.RELATED_TO,
                weight=concept_weight,
            )
            await self._storage.add_synapse(synapse)
            synapses_created.append(synapse)

        # Connect entities that co-occur
        for i, neuron_a in enumerate(entity_neurons):
            for neuron_b in entity_neurons[i + 1 :]:
                synapse = Synapse.create(
                    source_id=neuron_a.id,
                    target_id=neuron_b.id,
                    type=SynapseType.CO_OCCURS,
                    weight=0.5,
                )
                await self._storage.add_synapse(synapse)
                synapses_created.append(synapse)

        # 6. Link to nearby temporal memories
        linked = await self._link_temporal_neighbors(anchor_neuron, timestamp)
        neurons_linked.extend(linked)

        # 7. Create fiber
        neuron_ids = {n.id for n in all_neurons}
        synapse_ids = {s.id for s in synapses_created}

        fiber = Fiber.create(
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=anchor_neuron.id,
            time_start=timestamp,
            time_end=timestamp,
            tags=tags,
            metadata=metadata,
        )

        # Calculate coherence (simple: edges / possible edges)
        possible_edges = len(neuron_ids) * (len(neuron_ids) - 1) / 2
        coherence = len(synapse_ids) / max(1, possible_edges)
        fiber = fiber.with_salience(min(1.0, coherence + 0.3))

        await self._storage.add_fiber(fiber)

        return EncodingResult(
            fiber=fiber,
            neurons_created=neurons_created,
            neurons_linked=neurons_linked,
            synapses_created=synapses_created,
        )

    async def _extract_time_neurons(
        self,
        content: str,
        reference_time: datetime,
    ) -> list[Neuron]:
        """Extract and create time neurons from content."""
        neurons: list[Neuron] = []

        # Extract time hints from content
        time_hints = self._temporal.extract(content, reference_time)

        for hint in time_hints:
            # Check for existing similar time neuron
            existing = await self._find_similar_time_neuron(hint.midpoint)
            if existing:
                # Use existing neuron
                continue

            neuron = Neuron.create(
                type=NeuronType.TIME,
                content=hint.original,
                metadata={
                    "absolute_start": hint.absolute_start.isoformat(),
                    "absolute_end": hint.absolute_end.isoformat(),
                    "granularity": hint.granularity.value,
                },
            )
            await self._storage.add_neuron(neuron)
            neurons.append(neuron)

        # Always create a time neuron for the reference timestamp
        timestamp_neuron = Neuron.create(
            type=NeuronType.TIME,
            content=reference_time.strftime("%Y-%m-%d %H:%M"),
            metadata={
                "absolute_start": reference_time.isoformat(),
                "absolute_end": reference_time.isoformat(),
                "granularity": "minute",
            },
        )
        await self._storage.add_neuron(timestamp_neuron)
        neurons.append(timestamp_neuron)

        return neurons

    async def _find_similar_time_neuron(
        self,
        timestamp: datetime,
    ) -> Neuron | None:
        """Find existing time neuron close to given timestamp."""
        from datetime import timedelta

        # Look for time neurons within 1 hour
        start = timestamp - timedelta(hours=1)
        end = timestamp + timedelta(hours=1)

        existing = await self._storage.find_neurons(
            type=NeuronType.TIME,
            time_range=(start, end),
            limit=1,
        )

        return existing[0] if existing else None

    async def _extract_entity_neurons(
        self,
        content: str,
        language: str = "auto",
    ) -> list[Neuron]:
        """Extract and create entity neurons from content."""
        neurons: list[Neuron] = []

        entities = self._entity.extract(content, language=language)

        for entity in entities:
            # Map entity type to neuron type
            neuron_type = self._entity_type_to_neuron_type(entity.type)

            # Check for existing similar entity
            existing = await self._find_similar_entity(entity.text)
            if existing:
                continue

            neuron = Neuron.create(
                type=neuron_type,
                content=entity.text,
                metadata={
                    "entity_type": entity.type.value,
                    "confidence": entity.confidence,
                },
            )
            await self._storage.add_neuron(neuron)
            neurons.append(neuron)

        return neurons

    def _entity_type_to_neuron_type(self, entity_type: EntityType) -> NeuronType:
        """Map entity type to neuron type."""
        mapping = {
            EntityType.PERSON: NeuronType.ENTITY,
            EntityType.LOCATION: NeuronType.SPATIAL,
            EntityType.ORGANIZATION: NeuronType.ENTITY,
            EntityType.PRODUCT: NeuronType.ENTITY,
            EntityType.EVENT: NeuronType.ACTION,
            EntityType.CODE: NeuronType.CONCEPT,
            EntityType.UNKNOWN: NeuronType.CONCEPT,
        }
        return mapping.get(entity_type, NeuronType.CONCEPT)

    async def _find_similar_entity(
        self,
        text: str,
    ) -> Neuron | None:
        """Find existing entity neuron with similar content."""
        # Try exact match first
        existing = await self._storage.find_neurons(
            content_exact=text,
            limit=1,
        )
        if existing:
            return existing[0]

        # Try normalized match (handles "Claude-Code" vs "Claude Code")
        normalized = text.lower().replace("-", " ").replace("_", " ").strip()
        candidates = await self._storage.find_neurons(
            content_contains=normalized,
            limit=5,
        )
        for candidate in candidates:
            candidate_norm = candidate.content.lower().replace("-", " ").replace("_", " ").strip()
            if candidate_norm == normalized:
                return candidate

        return None

    async def _extract_concept_neurons(
        self,
        content: str,
        language: str = "auto",
    ) -> list[Neuron]:
        """Extract and create concept neurons from keywords."""
        neurons: list[Neuron] = []

        keywords = extract_keywords(content, language=language)

        # Dynamic limit based on content length
        concept_limit = min(20, max(5, len(content) // 100))
        for keyword in keywords[:concept_limit]:
            if len(keyword) < 3:
                continue

            # Check for existing
            existing = await self._storage.find_neurons(
                type=NeuronType.CONCEPT,
                content_exact=keyword,
                limit=1,
            )
            if existing:
                continue

            neuron = Neuron.create(
                type=NeuronType.CONCEPT,
                content=keyword,
            )
            await self._storage.add_neuron(neuron)
            neurons.append(neuron)

        return neurons

    async def _link_temporal_neighbors(
        self,
        anchor: Neuron,
        timestamp: datetime,
    ) -> list[str]:
        """Link to temporally nearby memories."""
        from datetime import timedelta

        linked: list[str] = []

        # Find fibers in nearby time window
        start = timestamp - timedelta(hours=24)
        end = timestamp + timedelta(hours=24)

        nearby_fibers = await self._storage.find_fibers(
            time_overlaps=(start, end),
            limit=5,
        )

        for fiber in nearby_fibers:
            if fiber.anchor_neuron_id == anchor.id:
                continue

            # Create temporal synapse
            synapse = Synapse.create(
                source_id=anchor.id,
                target_id=fiber.anchor_neuron_id,
                type=SynapseType.RELATED_TO,
                weight=0.3,
                metadata={"temporal_link": True},
            )

            try:
                await self._storage.add_synapse(synapse)
                linked.append(fiber.anchor_neuron_id)
            except ValueError:
                # Synapse might already exist
                pass

        return linked
