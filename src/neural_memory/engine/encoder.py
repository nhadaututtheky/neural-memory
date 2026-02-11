"""Memory encoder for converting experiences into neural structures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import suggest_memory_type
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.extraction.entities import EntityExtractor, EntityType
from neural_memory.extraction.keywords import extract_keywords, extract_weighted_keywords
from neural_memory.extraction.relations import RelationExtractor
from neural_memory.extraction.sentiment import SentimentExtractor, Valence
from neural_memory.extraction.temporal import TemporalExtractor
from neural_memory.utils.simhash import is_near_duplicate, simhash
from neural_memory.utils.tag_normalizer import TagNormalizer
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

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
    conflicts_detected: int = 0


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
        relation_extractor: RelationExtractor | None = None,
    ) -> None:
        """
        Initialize the encoder.

        Args:
            storage: Storage backend
            config: Brain configuration
            temporal_extractor: Custom temporal extractor
            entity_extractor: Custom entity extractor
            relation_extractor: Custom relation extractor
        """
        self._storage = storage
        self._config = config
        self._temporal = temporal_extractor or TemporalExtractor()
        self._entity = entity_extractor or EntityExtractor()
        self._relation = relation_extractor or RelationExtractor()
        self._sentiment = SentimentExtractor()
        self._tag_normalizer = TagNormalizer()

    async def encode(
        self,
        content: str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        tags: set[str] | None = None,
        language: str = "auto",
        *,
        skip_conflicts: bool = False,
        skip_time_neurons: bool = False,
        initial_stage: str = "",
        salience_ceiling: float = 0.0,
    ) -> EncodingResult:
        """
        Encode content into neural structures.

        Args:
            content: The text content to encode
            timestamp: When this memory occurred (default: now)
            metadata: Additional metadata to attach
            tags: Optional tags for the fiber
            language: Language hint ("vi", "en", or "auto")
            skip_conflicts: Skip conflict detection (for bulk doc training).
            skip_time_neurons: Skip TIME neuron creation (for bulk doc training).
            initial_stage: Override maturation stage (e.g. "episodic" for doc training).
            salience_ceiling: Cap initial fiber salience (0 = no cap).

        Returns:
            EncodingResult with created structures
        """
        if timestamp is None:
            timestamp = utcnow()

        neurons_created: list[Neuron] = []
        neurons_linked: list[str] = []
        synapses_created: list[Synapse] = []

        # 1. Extract time neurons (skipped for bulk doc training)
        if skip_time_neurons:
            time_neurons: list[Neuron] = []
        else:
            time_neurons = await self._extract_time_neurons(content, timestamp)
        neurons_created.extend(time_neurons)

        # 2. Extract entity neurons
        entity_neurons = await self._extract_entity_neurons(content, language)
        neurons_created.extend(entity_neurons)

        # 3. Extract concept/keyword neurons
        concept_neurons = await self._extract_concept_neurons(content, language)
        neurons_created.extend(concept_neurons)

        # 4. Generate auto-tags from extracted neurons
        auto_tags = self._generate_auto_tags(
            entity_neurons=entity_neurons,
            concept_neurons=concept_neurons,
            content=content,
            language=language,
        )
        agent_tags = self._tag_normalizer.normalize_set(tags) if tags else set()
        merged_tags = auto_tags | agent_tags

        # 4b. Auto-infer memory type if not provided
        effective_metadata = dict(metadata or {})
        if "type" not in effective_metadata or not effective_metadata["type"]:
            effective_metadata["type"] = suggest_memory_type(content).value

        # 5. Create the anchor neuron (main content)
        anchor_neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content=content,
            metadata={
                "is_anchor": True,
                "timestamp": timestamp.isoformat(),
                **effective_metadata,
            },
            content_hash=simhash(content),
        )
        await self._storage.add_neuron(anchor_neuron)
        neurons_created.append(anchor_neuron)

        # 6. Create synapses between neurons
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

        # 6a. Extract sentiment and create emotional synapses
        emotion_synapses, emotion_neurons, valence_meta = await self._extract_emotion_synapses(
            content=content,
            anchor_neuron=anchor_neuron,
            language=language,
            metadata=effective_metadata,
        )
        synapses_created.extend(emotion_synapses)
        neurons_created.extend(emotion_neurons)
        effective_metadata = {**effective_metadata, **valence_meta}

        # 6b. Extract relation-based synapses (causal, comparative, sequential)
        relation_synapses = await self._extract_relation_synapses(
            content=content,
            anchor_neuron=anchor_neuron,
            entity_neurons=entity_neurons,
            concept_neurons=concept_neurons,
            language=language,
        )
        synapses_created.extend(relation_synapses)

        # 6c. Confirmatory weight boost (Hebbian tag confirmation)
        boost_synapses = await self._apply_confirmatory_boost(
            auto_tags=auto_tags,
            agent_tags=agent_tags,
            anchor_neuron=anchor_neuron,
            synapses=synapses_created,
            entity_neurons=entity_neurons,
            concept_neurons=concept_neurons,
        )
        synapses_created.extend(boost_synapses)

        # 7. Detect and resolve conflicts (skipped for bulk doc training)
        _conflicts_detected = 0
        if not skip_conflicts:
            from neural_memory.engine.conflict_detection import detect_conflicts, resolve_conflicts

            conflict_tags = merged_tags
            memory_type_str = effective_metadata.get("type", "")
            conflicts = await detect_conflicts(
                content=content,
                tags=conflict_tags,
                storage=self._storage,
                memory_type=memory_type_str,
            )
            _conflicts_detected = len(conflicts)
            if conflicts:
                resolutions = await resolve_conflicts(
                    conflicts=conflicts,
                    new_neuron_id=anchor_neuron.id,
                    storage=self._storage,
                )
                for resolution in resolutions:
                    synapses_created.append(resolution.contradicts_synapse)

        # 8. Link to nearby temporal memories
        linked = await self._link_temporal_neighbors(anchor_neuron, timestamp)
        neurons_linked.extend(linked)

        # 9. Create fiber with meaningful pathway
        neuron_ids = {n.id for n in all_neurons}
        synapse_ids = {s.id for s in synapses_created}

        # Build pathway: time → entities → concepts → anchor
        # This represents the activation order during encoding
        pathway = self._build_pathway(
            time_neurons=time_neurons,
            entity_neurons=entity_neurons,
            concept_neurons=concept_neurons,
            anchor_neuron=anchor_neuron,
        )

        fiber = Fiber.create(
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=anchor_neuron.id,
            time_start=timestamp,
            time_end=timestamp,
            auto_tags=auto_tags,
            agent_tags=agent_tags,
            metadata=effective_metadata,
            pathway=pathway,
        )

        # Calculate coherence (simple: edges / possible edges)
        possible_edges = len(neuron_ids) * (len(neuron_ids) - 1) / 2
        coherence = len(synapse_ids) / max(1, possible_edges)
        salience = min(1.0, coherence + 0.3)
        if salience_ceiling > 0:
            salience = min(salience, salience_ceiling)
        fiber = fiber.with_salience(salience)

        await self._storage.add_fiber(fiber)

        # Initialize maturation tracking
        from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage

        stage = MemoryStage(initial_stage) if initial_stage else MemoryStage.SHORT_TERM
        maturation = MaturationRecord(
            fiber_id=fiber.id,
            brain_id=self._storage.current_brain_id or "",
            stage=stage,
        )
        try:
            await self._storage.save_maturation(maturation)
        except Exception:
            logger.debug("Maturation init failed (non-critical)", exc_info=True)

        return EncodingResult(
            fiber=fiber,
            neurons_created=neurons_created,
            neurons_linked=neurons_linked,
            synapses_created=synapses_created,
            conflicts_detected=_conflicts_detected,
        )

    def _build_pathway(
        self,
        time_neurons: list[Neuron],
        entity_neurons: list[Neuron],
        concept_neurons: list[Neuron],
        anchor_neuron: Neuron,
    ) -> list[str]:
        """Build a meaningful activation pathway for the fiber.

        The pathway represents the order neurons would activate during
        retrieval: time context → specific entities → abstract concepts → anchor.
        This enables the reflex activation engine to traverse fibers
        in a semantically meaningful order.
        """
        pathway: list[str] = []
        seen: set[str] = set()

        # Time neurons first (temporal context activates first)
        for n in time_neurons[:2]:
            if n.id not in seen:
                pathway.append(n.id)
                seen.add(n.id)

        # Entity neurons next (specific references)
        for n in entity_neurons[:3]:
            if n.id not in seen:
                pathway.append(n.id)
                seen.add(n.id)

        # Top concept neurons (abstract associations)
        for n in concept_neurons[:2]:
            if n.id not in seen:
                pathway.append(n.id)
                seen.add(n.id)

        # Anchor neuron last (destination)
        if anchor_neuron.id not in seen:
            pathway.append(anchor_neuron.id)

        return pathway

    def _generate_auto_tags(
        self,
        entity_neurons: list[Neuron],
        concept_neurons: list[Neuron],
        content: str,
        language: str = "auto",
    ) -> set[str]:
        """
        Generate tags from extracted entities and top keywords.

        Auto-tags ensure every fiber has a baseline tag set for clustering
        and pattern extraction, regardless of whether the calling agent
        provides tags.

        Args:
            entity_neurons: Extracted entity neurons
            concept_neurons: Extracted concept neurons
            content: Original content text
            language: Language hint

        Returns:
            Set of normalized tag strings
        """
        auto_tags: set[str] = set()

        # Entity names as tags (normalized lowercase)
        for neuron in entity_neurons:
            tag = neuron.content.lower().strip()
            if len(tag) >= 2:
                auto_tags.add(tag)

        # Top-5 keywords as tags
        weighted = extract_weighted_keywords(content, language=language)
        for kw in weighted[:5]:
            tag = kw.text.lower().strip()
            if len(tag) >= 2:
                auto_tags.add(tag)

        return self._tag_normalizer.normalize_set(auto_tags)

    async def _extract_relation_synapses(
        self,
        content: str,
        anchor_neuron: Neuron,
        entity_neurons: list[Neuron],
        concept_neurons: list[Neuron],
        language: str = "auto",
    ) -> list[Synapse]:
        """Create synapses from extracted relations between entities/concepts.

        Matches relation source/target spans to existing neurons and creates
        typed synapses (CAUSED_BY, LEADS_TO, BEFORE, AFTER, etc.).

        Args:
            content: Original content text
            anchor_neuron: The anchor neuron for this memory
            entity_neurons: Extracted entity neurons
            concept_neurons: Extracted concept neurons
            language: Language hint

        Returns:
            List of created relation synapses
        """
        synapses: list[Synapse] = []
        relations = self._relation.extract(content, language=language)

        all_extracted = entity_neurons + concept_neurons
        if len(all_extracted) < 2:
            return synapses

        for relation in relations:
            source_neuron = self._match_span_to_neuron(relation.source_span, all_extracted)
            target_neuron = self._match_span_to_neuron(relation.target_span, all_extracted)

            if source_neuron is None or target_neuron is None:
                continue
            if source_neuron.id == target_neuron.id:
                continue

            synapse = Synapse.create(
                source_id=source_neuron.id,
                target_id=target_neuron.id,
                type=relation.synapse_type,
                weight=relation.confidence,
                metadata={"relation_type": relation.relation_type.value},
            )
            try:
                await self._storage.add_synapse(synapse)
                synapses.append(synapse)
            except ValueError:
                logger.debug("Relation synapse already exists, skipping")

        return synapses

    def _match_span_to_neuron(
        self,
        span: str,
        neurons: list[Neuron],
    ) -> Neuron | None:
        """Match a text span to the best-matching neuron by content overlap.

        Uses case-insensitive substring matching. Returns the neuron
        whose content has the best overlap with the span.
        """
        span_lower = span.lower().strip()
        best_match: Neuron | None = None
        best_score: float = 0.0

        for neuron in neurons:
            neuron_lower = neuron.content.lower()
            if neuron_lower in span_lower or span_lower in neuron_lower:
                score = len(neuron_lower) / max(len(span_lower), 1)
                if score > best_score:
                    best_score = score
                    best_match = neuron

        return best_match

    async def _apply_confirmatory_boost(
        self,
        auto_tags: set[str],
        agent_tags: set[str],
        anchor_neuron: Neuron,
        synapses: list[Synapse],
        entity_neurons: list[Neuron],
        concept_neurons: list[Neuron],
    ) -> list[Synapse]:
        """Apply Hebbian confirmatory weight boost when agent tags overlap auto tags.

        When agent_tag matches auto_tag (confirmation):
            Boost weight of all synapses FROM anchor by +0.1

        When agent_tag is NOT in auto_tags (divergence):
            Create RELATED_TO synapse with weight 0.3 for novel associations

        Returns:
            List of any newly created synapses (from divergent tags)
        """
        new_synapses: list[Synapse] = []
        overlap = auto_tags & agent_tags

        if overlap:
            for syn in synapses:
                if syn.source_id == anchor_neuron.id:
                    boosted_weight = min(1.0, syn.weight + 0.1)
                    if boosted_weight != syn.weight:
                        boosted = Synapse(
                            id=syn.id,
                            source_id=syn.source_id,
                            target_id=syn.target_id,
                            type=syn.type,
                            weight=boosted_weight,
                            direction=syn.direction,
                            metadata=syn.metadata,
                            reinforced_count=syn.reinforced_count,
                            last_activated=syn.last_activated,
                            created_at=syn.created_at,
                        )
                        try:
                            await self._storage.update_synapse(boosted)
                        except (ValueError, AttributeError):
                            logger.debug("Synapse boost update failed (non-critical)")

        divergent = agent_tags - auto_tags
        if divergent:
            all_neurons = entity_neurons + concept_neurons
            for tag in divergent:
                tag_lower = tag.lower()
                matching = [n for n in all_neurons if tag_lower in n.content.lower()]
                for neuron in matching:
                    synapse = Synapse.create(
                        source_id=anchor_neuron.id,
                        target_id=neuron.id,
                        type=SynapseType.RELATED_TO,
                        weight=0.3,
                        metadata={"divergent_agent_tag": tag},
                    )
                    try:
                        await self._storage.add_synapse(synapse)
                        new_synapses.append(synapse)
                    except ValueError:
                        logger.debug("Divergent tag synapse already exists, skipping")

        return new_synapses

    async def _extract_emotion_synapses(
        self,
        content: str,
        anchor_neuron: Neuron,
        language: str = "auto",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[Synapse], list[Neuron], dict[str, Any]]:
        """Extract sentiment and create FELT synapses to emotion STATE neurons.

        Args:
            content: Original content text
            anchor_neuron: The anchor neuron for this memory
            language: Language hint
            metadata: Metadata dict (not mutated)

        Returns:
            Tuple of (created synapses, created emotion neurons, valence metadata)
        """
        synapses: list[Synapse] = []
        neurons: list[Neuron] = []
        valence_metadata: dict[str, Any] = {}

        result = self._sentiment.extract(content, language=language)
        if result.valence == Valence.NEUTRAL or not result.emotion_tags:
            return synapses, neurons, valence_metadata

        # Capture valence info for caller to merge into fiber metadata
        valence_metadata["_valence"] = result.valence.value
        valence_metadata["_intensity"] = result.intensity

        weight_scale = self._config.emotional_weight_scale

        for emotion_tag in result.emotion_tags:
            # Find or create shared STATE neuron for this emotion
            existing = await self._storage.find_neurons(
                type=NeuronType.STATE,
                content_exact=emotion_tag,
                limit=1,
            )

            if existing:
                emotion_neuron = existing[0]
            else:
                emotion_neuron = Neuron.create(
                    type=NeuronType.STATE,
                    content=emotion_tag,
                    metadata={"emotion_category": True},
                )
                await self._storage.add_neuron(emotion_neuron)
                neurons.append(emotion_neuron)

            synapse = Synapse.create(
                source_id=anchor_neuron.id,
                target_id=emotion_neuron.id,
                type=SynapseType.FELT,
                weight=result.intensity * weight_scale,
                metadata={
                    "_valence": result.valence.value,
                    "_intensity": result.intensity,
                    "_emotion": emotion_tag,
                },
            )
            try:
                await self._storage.add_synapse(synapse)
                synapses.append(synapse)
            except ValueError:
                logger.debug("Emotion synapse already exists, skipping")

        return synapses, neurons, valence_metadata

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
                content_hash=simhash(entity.text),
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

        # Try SimHash near-duplicate detection
        text_hash = simhash(text)
        if text_hash != 0:
            # Check content_contains with first word as approximation
            first_word = text.split()[0] if text.split() else ""
            if len(first_word) >= 3:
                nearby = await self._storage.find_neurons(
                    content_contains=first_word,
                    limit=10,
                )
                for candidate in nearby:
                    if candidate.content_hash and is_near_duplicate(
                        text_hash, candidate.content_hash
                    ):
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
        """Link to temporally nearby memories with directional synapses.

        Creates BEFORE/AFTER synapses for memories within 24 hours,
        improving synapse type diversity and enabling temporal reasoning.
        """
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

            # Determine temporal direction: is this fiber before or after?
            if fiber.time_start is not None and fiber.time_start < timestamp:
                synapse_type = SynapseType.AFTER
            elif fiber.time_start is not None and fiber.time_start > timestamp:
                synapse_type = SynapseType.BEFORE
            else:
                synapse_type = SynapseType.RELATED_TO

            synapse = Synapse.create(
                source_id=anchor.id,
                target_id=fiber.anchor_neuron_id,
                type=synapse_type,
                weight=0.3,
                metadata={"temporal_link": True},
            )

            try:
                await self._storage.add_synapse(synapse)
                linked.append(fiber.anchor_neuron_id)
            except ValueError:
                logger.debug("Synapse already exists, skipping")

        return linked
