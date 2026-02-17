"""Concrete pipeline steps extracted from MemoryEncoder.encode().

Each step corresponds to a logical phase of memory encoding. Steps are
designed to be composable: skip, reorder, or replace any step without
affecting others (as long as inter-step dependencies are satisfied via
PipelineContext fields).

Step dependency graph::

    ExtractTimeNeuronsStep ─┐
    ExtractEntityNeuronsStep ─┼─ AutoTagStep ─ DedupCheckStep ─ CreateAnchorStep
    ExtractConceptNeuronsStep─┘                                       │
                                                   ┌──────────────────┤
                                                   │                  │
                                              CreateSynapsesStep  CoOccurrenceStep
                                                   │                  │
                                              EmotionStep        RelationStep
                                                   │                  │
                                              ConfirmatoryBoostStep   │
                                                   │                  │
                                              ConflictDetectionStep   │
                                                   │                  │
                                              TemporalLinkingStep     │
                                                   │                  │
                                              BuildFiberStep ─────────┘
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any

from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import suggest_memory_type
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.pipeline import PipelineContext
from neural_memory.extraction.entities import EntityExtractor, EntityType
from neural_memory.extraction.keywords import extract_keywords, extract_weighted_keywords
from neural_memory.extraction.relations import RelationExtractor
from neural_memory.extraction.sentiment import SentimentExtractor, Valence
from neural_memory.extraction.temporal import TemporalExtractor
from neural_memory.utils.simhash import is_near_duplicate, simhash
from neural_memory.utils.tag_normalizer import TagNormalizer

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.engine.dedup.pipeline import DedupPipeline
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


# ── Helpers shared across steps ──


def _entity_type_to_neuron_type(entity_type: EntityType) -> NeuronType:
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


def _match_span_to_neuron(span: str, neurons: list[Neuron]) -> Neuron | None:
    """Match a text span to the best-matching neuron by content overlap."""
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


# ── Step 1: Extract Time Neurons ──


@dataclass
class ExtractTimeNeuronsStep:
    """Extract temporal neurons from content."""

    temporal_extractor: TemporalExtractor

    @property
    def name(self) -> str:
        return "extract_time_neurons"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        if ctx.skip_time_neurons:
            return ctx

        time_hints = self.temporal_extractor.extract(ctx.content, ctx.timestamp)
        for hint in time_hints:
            existing = await _find_similar_time_neuron(storage, hint.midpoint)
            if existing:
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
            await storage.add_neuron(neuron)
            ctx.time_neurons.append(neuron)
            ctx.neurons_created.append(neuron)

        # Always create a timestamp neuron for reference time
        timestamp_neuron = Neuron.create(
            type=NeuronType.TIME,
            content=ctx.timestamp.strftime("%Y-%m-%d %H:%M"),
            metadata={
                "absolute_start": ctx.timestamp.isoformat(),
                "absolute_end": ctx.timestamp.isoformat(),
                "granularity": "minute",
            },
        )
        await storage.add_neuron(timestamp_neuron)
        ctx.time_neurons.append(timestamp_neuron)
        ctx.neurons_created.append(timestamp_neuron)

        return ctx


async def _find_similar_time_neuron(storage: NeuralStorage, timestamp: Any) -> Neuron | None:
    """Find existing time neuron close to given timestamp."""
    from datetime import timedelta

    start = timestamp - timedelta(hours=1)
    end = timestamp + timedelta(hours=1)
    existing = await storage.find_neurons(type=NeuronType.TIME, time_range=(start, end), limit=1)
    return existing[0] if existing else None


# ── Step 2: Extract Entity Neurons ──


@dataclass
class ExtractEntityNeuronsStep:
    """Extract named entity neurons from content."""

    entity_extractor: EntityExtractor

    @property
    def name(self) -> str:
        return "extract_entity_neurons"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        entities = self.entity_extractor.extract(ctx.content, language=ctx.language)
        for entity in entities:
            neuron_type = _entity_type_to_neuron_type(entity.type)
            existing = await _find_similar_entity(storage, entity.text)
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
            await storage.add_neuron(neuron)
            ctx.entity_neurons.append(neuron)
            ctx.neurons_created.append(neuron)
        return ctx


async def _find_similar_entity(storage: NeuralStorage, text: str) -> Neuron | None:
    """Find existing entity neuron with similar content."""
    existing = await storage.find_neurons(content_exact=text, limit=1)
    if existing:
        return existing[0]

    normalized = text.lower().replace("-", " ").replace("_", " ").strip()
    candidates = await storage.find_neurons(content_contains=normalized, limit=5)
    for candidate in candidates:
        candidate_norm = candidate.content.lower().replace("-", " ").replace("_", " ").strip()
        if candidate_norm == normalized:
            return candidate

    text_hash = simhash(text)
    if text_hash != 0:
        first_word = text.split()[0] if text.split() else ""
        if len(first_word) >= 3:
            nearby = await storage.find_neurons(content_contains=first_word, limit=10)
            for candidate in nearby:
                if candidate.content_hash and is_near_duplicate(text_hash, candidate.content_hash):
                    return candidate
    return None


# ── Step 3: Extract Concept Neurons ──


class ExtractConceptNeuronsStep:
    """Extract keyword/concept neurons from content."""

    @property
    def name(self) -> str:
        return "extract_concept_neurons"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        keywords = extract_keywords(ctx.content, language=ctx.language)
        concept_limit = min(20, max(5, len(ctx.content) // 100))

        # Filter valid keywords first
        valid_keywords = [kw for kw in keywords[:concept_limit] if len(kw) >= 3]

        # Batch check existing concepts in one query
        existing_map = await storage.find_neurons_exact_batch(
            valid_keywords, type=NeuronType.CONCEPT
        )

        for keyword in valid_keywords:
            if keyword in existing_map:
                continue
            neuron = Neuron.create(type=NeuronType.CONCEPT, content=keyword)
            await storage.add_neuron(neuron)
            ctx.concept_neurons.append(neuron)
            ctx.neurons_created.append(neuron)
        return ctx


# ── Step 4: Auto-Tag + Metadata ──


@dataclass
class AutoTagStep:
    """Generate auto-tags and infer memory type."""

    tag_normalizer: TagNormalizer

    @property
    def name(self) -> str:
        return "auto_tag"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        # Auto-tags from entities + keywords
        auto_tags: set[str] = set()
        for neuron in ctx.entity_neurons:
            tag = neuron.content.lower().strip()
            if len(tag) >= 2:
                auto_tags.add(tag)

        weighted = extract_weighted_keywords(ctx.content, language=ctx.language)
        for kw in weighted[:5]:
            tag = kw.text.lower().strip()
            if len(tag) >= 2:
                auto_tags.add(tag)

        ctx.auto_tags = self.tag_normalizer.normalize_set(auto_tags)
        ctx.agent_tags = self.tag_normalizer.normalize_set(ctx.tags) if ctx.tags else set()
        ctx.merged_tags = ctx.auto_tags | ctx.agent_tags

        # Effective metadata
        ctx.effective_metadata = dict(ctx.metadata)
        if "type" not in ctx.effective_metadata or not ctx.effective_metadata["type"]:
            ctx.effective_metadata["type"] = suggest_memory_type(ctx.content).value

        # Content hash
        ctx.content_hash = simhash(ctx.content)

        return ctx


# ── Step 5: Dedup Check ──


@dataclass
class DedupCheckStep:
    """Check for duplicate anchors via dedup pipeline."""

    dedup_pipeline: DedupPipeline | None = None

    @property
    def name(self) -> str:
        return "dedup_check"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        if self.dedup_pipeline is None:
            return ctx

        from neural_memory.engine.dedup.pipeline import DedupResult

        dedup_result: DedupResult = await self.dedup_pipeline.check_duplicate(
            ctx.content, content_hash=ctx.content_hash
        )
        if dedup_result.is_duplicate and dedup_result.existing_neuron_id:
            batch = await storage.get_neurons_batch([dedup_result.existing_neuron_id])
            existing_neuron = batch.get(dedup_result.existing_neuron_id)
            if existing_neuron is not None:
                # Store reused anchor in metadata for CreateAnchorStep
                ctx.effective_metadata["_dedup_reused_anchor"] = existing_neuron
                logger.debug(
                    "Dedup: reusing anchor %s (tier=%d, score=%.3f)",
                    dedup_result.existing_neuron_id,
                    dedup_result.tier,
                    dedup_result.similarity_score,
                )
        return ctx


# ── Step 6: Create Anchor Neuron ──


class CreateAnchorStep:
    """Create (or reuse) the anchor neuron for this memory."""

    @property
    def name(self) -> str:
        return "create_anchor"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        dedup_reused = ctx.effective_metadata.pop("_dedup_reused_anchor", None)

        if dedup_reused is not None:
            anchor_neuron = dedup_reused
            # Create alias neuron pointing to existing anchor
            alias_neuron = Neuron.create(
                type=NeuronType.CONCEPT,
                content=ctx.content,
                metadata={
                    "is_anchor": True,
                    "timestamp": ctx.timestamp.isoformat(),
                    "_dedup_alias_of": anchor_neuron.id,
                    **ctx.effective_metadata,
                },
                content_hash=ctx.content_hash,
            )
            await storage.add_neuron(alias_neuron)
            ctx.neurons_created.append(alias_neuron)

            alias_synapse = Synapse.create(
                source_id=alias_neuron.id,
                target_id=anchor_neuron.id,
                type=SynapseType.ALIAS,
                weight=0.9,
                metadata={"_dedup": True},
            )
            try:
                await storage.add_synapse(alias_synapse)
                ctx.synapses_created.append(alias_synapse)
            except ValueError:
                logger.debug("ALIAS synapse already exists")

            ctx.anchor_neuron = alias_neuron
        else:
            anchor_neuron = Neuron.create(
                type=NeuronType.CONCEPT,
                content=ctx.content,
                metadata={
                    "is_anchor": True,
                    "timestamp": ctx.timestamp.isoformat(),
                    **ctx.effective_metadata,
                },
                content_hash=ctx.content_hash,
            )
            await storage.add_neuron(anchor_neuron)
            ctx.neurons_created.append(anchor_neuron)
            ctx.anchor_neuron = anchor_neuron

        return ctx


# ── Step 7: Create Synapses (anchor → time/entity/concept) ──


class CreateSynapsesStep:
    """Wire anchor to time, entity, and concept neurons."""

    @property
    def name(self) -> str:
        return "create_synapses"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        assert ctx.anchor_neuron is not None
        anchor = ctx.anchor_neuron
        synapses_to_add: list[Synapse] = []

        # Anchor → time neurons
        for time_neuron in ctx.time_neurons:
            synapses_to_add.append(
                Synapse.create(
                    source_id=anchor.id,
                    target_id=time_neuron.id,
                    type=SynapseType.HAPPENED_AT,
                    weight=0.9,
                )
            )

        # Anchor → entity neurons (weight by mention frequency)
        content_lower = ctx.content.lower()
        for entity_neuron in ctx.entity_neurons:
            mention_count = content_lower.count(entity_neuron.content.lower())
            entity_weight = min(0.95, 0.7 + 0.05 * mention_count)
            synapses_to_add.append(
                Synapse.create(
                    source_id=anchor.id,
                    target_id=entity_neuron.id,
                    type=SynapseType.INVOLVES,
                    weight=entity_weight,
                )
            )

        # Anchor → concept neurons (weight by keyword importance)
        kw_weight_map = {
            kw.text: kw.weight
            for kw in extract_weighted_keywords(ctx.content, language=ctx.language)
        }
        for concept_neuron in ctx.concept_neurons:
            kw_weight = kw_weight_map.get(concept_neuron.content.lower(), 0.5)
            concept_weight = min(0.8, 0.4 + 0.3 * kw_weight)
            synapses_to_add.append(
                Synapse.create(
                    source_id=anchor.id,
                    target_id=concept_neuron.id,
                    type=SynapseType.RELATED_TO,
                    weight=concept_weight,
                )
            )

        # Batch add all synapses in parallel
        if synapses_to_add:
            results = await asyncio.gather(
                *[storage.add_synapse(s) for s in synapses_to_add],
                return_exceptions=True,
            )
            for synapse, result in zip(synapses_to_add, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning("Synapse add failed in CreateSynapsesStep: %s", result)
                else:
                    ctx.synapses_created.append(synapse)

        return ctx


# ── Step 8: Co-Occurrence Synapses ──


class CoOccurrenceStep:
    """Create CO_OCCURS synapses between co-occurring entities."""

    @property
    def name(self) -> str:
        return "co_occurrence"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        synapses_to_add: list[Synapse] = []
        for i, neuron_a in enumerate(ctx.entity_neurons):
            for neuron_b in ctx.entity_neurons[i + 1 :]:
                synapses_to_add.append(
                    Synapse.create(
                        source_id=neuron_a.id,
                        target_id=neuron_b.id,
                        type=SynapseType.CO_OCCURS,
                        weight=0.5,
                    )
                )

        if synapses_to_add:
            results = await asyncio.gather(
                *[storage.add_synapse(s) for s in synapses_to_add],
                return_exceptions=True,
            )
            for synapse, result in zip(synapses_to_add, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning("Co-occurrence synapse add failed: %s", result)
                else:
                    ctx.synapses_created.append(synapse)

        return ctx


# ── Step 9: Emotion / Sentiment ──


@dataclass
class EmotionStep:
    """Extract sentiment and create FELT synapses."""

    sentiment_extractor: SentimentExtractor

    @property
    def name(self) -> str:
        return "emotion"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        assert ctx.anchor_neuron is not None
        result = self.sentiment_extractor.extract(ctx.content, language=ctx.language)

        if result.valence == Valence.NEUTRAL or not result.emotion_tags:
            return ctx

        ctx.effective_metadata["_valence"] = result.valence.value
        ctx.effective_metadata["_intensity"] = result.intensity
        weight_scale = config.emotional_weight_scale

        for emotion_tag in result.emotion_tags:
            existing = await storage.find_neurons(
                type=NeuronType.STATE, content_exact=emotion_tag, limit=1
            )
            if existing:
                emotion_neuron = existing[0]
            else:
                emotion_neuron = Neuron.create(
                    type=NeuronType.STATE,
                    content=emotion_tag,
                    metadata={"emotion_category": True},
                )
                await storage.add_neuron(emotion_neuron)
                ctx.neurons_created.append(emotion_neuron)

            synapse = Synapse.create(
                source_id=ctx.anchor_neuron.id,
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
                await storage.add_synapse(synapse)
                ctx.synapses_created.append(synapse)
            except ValueError:
                logger.debug("Emotion synapse already exists, skipping")

        return ctx


# ── Step 10: Relation Extraction ──


@dataclass
class RelationExtractionStep:
    """Extract relation-based synapses (causal, sequential, etc.)."""

    relation_extractor: RelationExtractor

    @property
    def name(self) -> str:
        return "relation_extraction"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        assert ctx.anchor_neuron is not None
        relations = self.relation_extractor.extract(ctx.content, language=ctx.language)
        all_extracted = ctx.entity_neurons + ctx.concept_neurons

        if len(all_extracted) < 2:
            return ctx

        for relation in relations:
            source_neuron = _match_span_to_neuron(relation.source_span, all_extracted)
            target_neuron = _match_span_to_neuron(relation.target_span, all_extracted)

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
                await storage.add_synapse(synapse)
                ctx.synapses_created.append(synapse)
            except ValueError:
                logger.debug("Relation synapse already exists, skipping")

        return ctx


# ── Step 11: Confirmatory Boost ──


class ConfirmatoryBoostStep:
    """Apply Hebbian confirmatory weight boost for tag overlap."""

    @property
    def name(self) -> str:
        return "confirmatory_boost"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        assert ctx.anchor_neuron is not None
        anchor = ctx.anchor_neuron
        overlap = ctx.auto_tags & ctx.agent_tags

        if overlap:
            for syn in ctx.synapses_created:
                if syn.source_id == anchor.id:
                    boosted_weight = min(1.0, syn.weight + 0.1)
                    if boosted_weight != syn.weight:
                        boosted = dc_replace(syn, weight=boosted_weight)
                        try:
                            await storage.update_synapse(boosted)
                        except (ValueError, AttributeError):
                            logger.debug("Synapse boost update failed (non-critical)")

        divergent = ctx.agent_tags - ctx.auto_tags
        if divergent:
            all_neurons = ctx.entity_neurons + ctx.concept_neurons
            for tag in divergent:
                tag_lower = tag.lower()
                matching = [n for n in all_neurons if tag_lower in n.content.lower()]
                for neuron in matching:
                    synapse = Synapse.create(
                        source_id=anchor.id,
                        target_id=neuron.id,
                        type=SynapseType.RELATED_TO,
                        weight=0.3,
                        metadata={"divergent_agent_tag": tag},
                    )
                    try:
                        await storage.add_synapse(synapse)
                        ctx.synapses_created.append(synapse)
                    except ValueError:
                        logger.debug("Divergent tag synapse already exists, skipping")

        return ctx


# ── Step 12: Conflict Detection ──


class ConflictDetectionStep:
    """Detect and resolve conflicts with existing memories."""

    @property
    def name(self) -> str:
        return "conflict_detection"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        if ctx.skip_conflicts:
            return ctx

        assert ctx.anchor_neuron is not None

        from neural_memory.engine.conflict_detection import (
            detect_conflicts,
            resolve_conflicts,
        )

        memory_type_str = ctx.effective_metadata.get("type", "")
        conflicts = await detect_conflicts(
            content=ctx.content,
            tags=ctx.merged_tags,
            storage=storage,
            memory_type=memory_type_str,
        )
        ctx.conflicts_detected = len(conflicts)

        if not conflicts:
            return ctx

        # Try auto-resolve first
        remaining_conflicts = conflicts
        try:
            from neural_memory.engine.conflict_auto_resolve import try_auto_resolve

            auto_resolved: list[object] = []
            still_manual: list[object] = []
            for conflict in conflicts:
                result = await try_auto_resolve(conflict, storage, new_confidence=0.5)
                if result.auto_resolved:
                    auto_resolved.append(result)
                else:
                    still_manual.append(conflict)
            remaining_conflicts = still_manual  # type: ignore[assignment]
            if auto_resolved:
                logger.debug(
                    "Auto-resolved %d/%d conflicts",
                    len(auto_resolved),
                    len(conflicts),
                )
        except Exception:
            logger.debug("Auto-resolve failed, falling back to standard", exc_info=True)

        # Standard resolution for remaining
        if remaining_conflicts:
            resolutions = await resolve_conflicts(
                conflicts=remaining_conflicts,
                new_neuron_id=ctx.anchor_neuron.id,
                storage=storage,
            )
            for resolution in resolutions:
                ctx.synapses_created.append(resolution.contradicts_synapse)

        return ctx


# ── Step 13: Temporal Linking ──


class TemporalLinkingStep:
    """Link to temporally nearby memories."""

    @property
    def name(self) -> str:
        return "temporal_linking"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        assert ctx.anchor_neuron is not None
        from datetime import timedelta

        start = ctx.timestamp - timedelta(hours=24)
        end = ctx.timestamp + timedelta(hours=24)
        nearby_fibers = await storage.find_fibers(time_overlaps=(start, end), limit=5)

        for fiber in nearby_fibers:
            if fiber.anchor_neuron_id == ctx.anchor_neuron.id:
                continue

            if fiber.time_start is not None and fiber.time_start < ctx.timestamp:
                synapse_type = SynapseType.AFTER
            elif fiber.time_start is not None and fiber.time_start > ctx.timestamp:
                synapse_type = SynapseType.BEFORE
            else:
                synapse_type = SynapseType.RELATED_TO

            synapse = Synapse.create(
                source_id=ctx.anchor_neuron.id,
                target_id=fiber.anchor_neuron_id,
                type=synapse_type,
                weight=0.3,
                metadata={"temporal_link": True},
            )
            try:
                await storage.add_synapse(synapse)
                ctx.neurons_linked.append(fiber.anchor_neuron_id)
            except ValueError:
                logger.debug("Synapse already exists, skipping")

        return ctx


# ── Step 14: Build Fiber ──


class BuildFiberStep:
    """Assemble the final Fiber from accumulated context."""

    @property
    def name(self) -> str:
        return "build_fiber"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        assert ctx.anchor_neuron is not None

        neuron_ids = {n.id for n in ctx.neurons_created}
        synapse_ids = {s.id for s in ctx.synapses_created}

        pathway = _build_pathway(
            time_neurons=ctx.time_neurons,
            entity_neurons=ctx.entity_neurons,
            concept_neurons=ctx.concept_neurons,
            anchor_neuron=ctx.anchor_neuron,
        )

        fiber = Fiber.create(
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=ctx.anchor_neuron.id,
            time_start=ctx.timestamp,
            time_end=ctx.timestamp,
            auto_tags=ctx.auto_tags,
            agent_tags=ctx.agent_tags,
            metadata=ctx.effective_metadata,
            pathway=pathway,
        )

        # Coherence + salience
        possible_edges = len(neuron_ids) * (len(neuron_ids) - 1) / 2
        coherence = len(synapse_ids) / max(1, possible_edges)
        salience = min(1.0, coherence + 0.3)
        if ctx.salience_ceiling > 0:
            salience = min(salience, ctx.salience_ceiling)
        fiber = fiber.with_salience(salience)

        await storage.add_fiber(fiber)

        # Maturation tracking
        from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage

        stage = MemoryStage(ctx.initial_stage) if ctx.initial_stage else MemoryStage.SHORT_TERM
        maturation = MaturationRecord(
            fiber_id=fiber.id,
            brain_id=storage.current_brain_id or "",
            stage=stage,
        )
        try:
            await storage.save_maturation(maturation)
        except Exception:
            logger.debug("Maturation init failed (non-critical)", exc_info=True)

        # Store fiber in metadata for encoder to retrieve
        ctx.effective_metadata["_pipeline_fiber"] = fiber

        return ctx


def _build_pathway(
    time_neurons: list[Neuron],
    entity_neurons: list[Neuron],
    concept_neurons: list[Neuron],
    anchor_neuron: Neuron,
) -> list[str]:
    """Build activation pathway: time → entities → concepts → anchor."""
    pathway: list[str] = []
    seen: set[str] = set()

    for n in time_neurons[:2]:
        if n.id not in seen:
            pathway.append(n.id)
            seen.add(n.id)

    for n in entity_neurons[:3]:
        if n.id not in seen:
            pathway.append(n.id)
            seen.add(n.id)

    for n in concept_neurons[:2]:
        if n.id not in seen:
            pathway.append(n.id)
            seen.add(n.id)

    if anchor_neuron.id not in seen:
        pathway.append(anchor_neuron.id)

    return pathway
