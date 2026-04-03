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
import math
import re
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from functools import lru_cache
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
        lazy_enabled = getattr(config, "lazy_entity_enabled", True)
        promotion_threshold = getattr(config, "lazy_entity_promotion_threshold", 2)

        for entity in entities:
            neuron_type = _entity_type_to_neuron_type(entity.type)
            existing = await _find_similar_entity(storage, entity.text)
            if existing:
                continue

            # Exceptions: always promote high-confidence or user-tagged entities
            is_exception = entity.confidence >= 0.9 or entity.text.lower() in {
                t.lower() for t in ctx.tags
            }

            if lazy_enabled and not is_exception:
                # Check mention count from entity_refs
                try:
                    ref_count = await storage.count_entity_refs(entity.text)
                except Exception:
                    ref_count = promotion_threshold  # Fallback: promote immediately
                if ref_count < promotion_threshold - 1:
                    # Not enough mentions yet — defer as ref
                    ctx.deferred_entity_refs.append(entity.text)
                    continue
                # Reached threshold — promote! (fall through to neuron creation)

            meta: dict[str, Any] = {
                "entity_type": entity.type.value,
                "confidence": entity.confidence,
            }
            if entity.subtype:
                meta["entity_subtype"] = entity.subtype.value
            if entity.raw_value:
                meta["raw_value"] = entity.raw_value
                meta["_verbatim"] = True
            if entity.unit:
                meta["unit"] = entity.unit

            neuron = Neuron.create(
                type=neuron_type,
                content=entity.text,
                metadata=meta,
                content_hash=simhash(entity.text),
            )
            await storage.add_neuron(neuron)
            ctx.entity_neurons.append(neuron)
            ctx.neurons_created.append(neuron)

            # If promoted via lazy path, mark refs and do retroactive linking
            if lazy_enabled and not is_exception:
                try:
                    await storage.mark_entity_refs_promoted(entity.text)
                except Exception:
                    pass  # Non-critical: ref table may not exist yet
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


# ── Step 3b: Extract Action Neurons ──


@lru_cache(maxsize=1)
def _action_pattern() -> re.Pattern[str]:
    verbs = (
        r"(?:decided|implemented|fixed|deployed|created|built|added|removed|"
        r"refactored|migrated|updated|configured|resolved|shipped|completed|"
        r"released|installed|debugged|wrote|merged|tested)"
    )
    return re.compile(
        rf"\b{verbs}\s+(.{{3,60}}?)(?:[.,;!?\n]|$)",
        re.IGNORECASE,
    )


class ExtractActionNeuronsStep:
    """Extract ACTION neurons from verb phrases in content."""

    MAX_ACTIONS: int = 5

    @property
    def name(self) -> str:
        return "extract_action_neurons"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        matches = _action_pattern().findall(ctx.content)
        seen: set[str] = set()

        valid_actions: list[str] = []
        for match in matches[: self.MAX_ACTIONS]:
            action_text = match.strip()
            if len(action_text) < 3 or action_text.lower() in seen:
                continue
            seen.add(action_text.lower())
            valid_actions.append(action_text)

        if not valid_actions:
            return ctx

        # Batch check existing action neurons in one query
        existing_map = await storage.find_neurons_exact_batch(valid_actions, type=NeuronType.ACTION)

        for action_text in valid_actions:
            if action_text in existing_map:
                continue
            neuron = Neuron.create(type=NeuronType.ACTION, content=action_text)
            await storage.add_neuron(neuron)
            ctx.action_neurons.append(neuron)
            ctx.neurons_created.append(neuron)

        return ctx


# ── Step 3c: Extract Intent Neurons ──


@lru_cache(maxsize=1)
def _intent_pattern() -> re.Pattern[str]:
    prefixes = (
        r"(?:want to|need to|plan to|going to|trying to|"
        r"goal:\s*|objective:\s*|aim to|intend to|should)"
    )
    return re.compile(
        rf"\b{prefixes}\s+(.{{2,60}}?)(?:[.,;!?\n]|$)",
        re.IGNORECASE,
    )


class ExtractIntentNeuronsStep:
    """Extract INTENT neurons from goal/intention phrases."""

    MAX_INTENTS: int = 3

    @property
    def name(self) -> str:
        return "extract_intent_neurons"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        matches = _intent_pattern().findall(ctx.content)
        seen: set[str] = set()

        valid_intents: list[str] = []
        for match in matches[: self.MAX_INTENTS]:
            intent_text = match.strip()
            if len(intent_text) < 2 or intent_text.lower() in seen:
                continue
            seen.add(intent_text.lower())
            valid_intents.append(intent_text)

        if not valid_intents:
            return ctx

        # Batch check existing intent neurons in one query
        existing_map = await storage.find_neurons_exact_batch(valid_intents, type=NeuronType.INTENT)

        for intent_text in valid_intents:
            if intent_text in existing_map:
                continue
            neuron = Neuron.create(type=NeuronType.INTENT, content=intent_text)
            await storage.add_neuron(neuron)
            ctx.intent_neurons.append(neuron)
            ctx.neurons_created.append(neuron)

        return ctx


# ── Step 3.5: Structure Detection ──


@dataclass
class StructureDetectionStep:
    """Detect structured content (CSV, JSON, key-value, table).

    Runs before auto-tagging so structure tags are included in merged_tags.
    Stores structure metadata in effective_metadata for downstream steps.
    """

    @property
    def name(self) -> str:
        return "structure_detection"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        from neural_memory.extraction.structure_detector import detect_structure

        result = detect_structure(ctx.content)
        if result.is_structured:
            # Store structure info in metadata for neuron persistence
            ctx.metadata["_structure"] = {
                "format": result.format.value,
                "fields": [
                    {"name": f.name, "value": f.value, "type": f.field_type} for f in result.fields
                ],
                "confidence": result.confidence,
            }
            # Add structure tag for filtering
            ctx.tags.add(f"_structured:{result.format.value}")

        return ctx


# ── Step 3.6: Structured Data Encoder ──


_MAX_TABLE_ROWS = 50  # Cap table rows to avoid neuron explosion


@dataclass
class StructuredDataEncoderStep:
    """Encode structured data (table, key-value, JSON) as graph neurons.

    Reads ``_structure`` metadata set by StructureDetectionStep and creates
    cell/row/column neurons with domain synapses (IN_ROW, IN_COLUMN, HAS_VALUE).
    Cells have ``_verbatim: True`` so they return exact values on recall.

    Caps at _MAX_TABLE_ROWS to avoid neuron explosion from large datasets.
    """

    @property
    def name(self) -> str:
        return "structured_data_encoder"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        structure = ctx.metadata.get("_structure")
        if not structure:
            return ctx

        fmt = structure.get("format", "")
        fields = structure.get("fields", [])

        if not fields:
            return ctx

        if fmt == "key_value":
            await self._encode_key_value(ctx, storage, fields)
        elif fmt in ("table_row", "csv_row"):
            await self._encode_table(ctx, storage, fields, fmt)
        elif fmt == "json_object":
            await self._encode_key_value(ctx, storage, fields)

        return ctx

    async def _encode_key_value(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        fields: list[dict[str, str]],
    ) -> None:
        """Encode key-value pairs as cell neurons with HAS_VALUE synapses."""
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType
        from neural_memory.utils.simhash import simhash

        for field_dict in fields[:_MAX_TABLE_ROWS]:
            name = field_dict.get("name", "")
            value = field_dict.get("value", "")
            field_type = field_dict.get("type", "text")

            if not name or not value:
                continue

            cell_content = f"{name} = {value}"
            cell_neuron = Neuron.create(
                type=NeuronType.ENTITY,
                content=cell_content,
                metadata={
                    "entity_type": "entity",
                    "_verbatim": True,
                    "raw_value": value,
                    "field_name": name,
                    "field_type": field_type,
                    "_structured_cell": True,
                },
                content_hash=simhash(cell_content),
            )
            await storage.add_neuron(cell_neuron)
            ctx.neurons_created.append(cell_neuron)
            ctx.entity_neurons.append(cell_neuron)

            # Link cell to anchor if available
            if ctx.anchor_neuron:
                synapse = Synapse.create(
                    type=SynapseType.HAS_VALUE,
                    source_id=ctx.anchor_neuron.id,
                    target_id=cell_neuron.id,
                    weight=0.8,
                )
                await storage.add_synapse(synapse)
                ctx.synapses_created.append(synapse)

    async def _encode_table(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        fields: list[dict[str, str]],
        fmt: str,
    ) -> None:
        """Encode table rows as graph: row header → cell neurons with IN_ROW/IN_COLUMN."""
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType
        from neural_memory.utils.simhash import simhash

        # Group fields by row (fields come as flat list; group by name pattern)
        # For table_row format, each field is a column in a single row
        for field_dict in fields[:_MAX_TABLE_ROWS]:
            name = field_dict.get("name", "")
            value = field_dict.get("value", "")
            field_type = field_dict.get("type", "text")

            if not name or not value:
                continue

            cell_content = f"{name}: {value}"
            cell_neuron = Neuron.create(
                type=NeuronType.ENTITY,
                content=cell_content,
                metadata={
                    "entity_type": "entity",
                    "_verbatim": True,
                    "raw_value": value,
                    "field_name": name,
                    "field_type": field_type,
                    "_structured_cell": True,
                    "_table_format": fmt,
                },
                content_hash=simhash(cell_content),
            )
            await storage.add_neuron(cell_neuron)
            ctx.neurons_created.append(cell_neuron)
            ctx.entity_neurons.append(cell_neuron)

            # Link to anchor with IN_COLUMN (column name is the structural key)
            if ctx.anchor_neuron:
                synapse = Synapse.create(
                    type=SynapseType.IN_COLUMN,
                    source_id=cell_neuron.id,
                    target_id=ctx.anchor_neuron.id,
                    weight=0.7,
                    metadata={"column": name},
                )
                await storage.add_synapse(synapse)
                ctx.synapses_created.append(synapse)


# ── Step 3.7: Decision Component Extraction ──


@dataclass
class DecisionComponentStep:
    """Extract structured decision components from DECISION-type memories.

    Reads content and context to extract chosen/rejected/reasoning fields,
    stores as ``_decision`` metadata on the anchor neuron. Adds searchable
    tags like ``_decision:chosen:X``.
    """

    @property
    def name(self) -> str:
        return "decision_component"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        mem_type = ctx.metadata.get("type", "")
        if mem_type != "decision":
            return ctx

        from neural_memory.engine.decision_intel import extract_decision_components

        context_dict = ctx.metadata.get("_raw_context") or {}
        components = extract_decision_components(ctx.content, context_dict)
        if components is None:
            return ctx

        ctx.metadata["_decision"] = components.to_dict()

        # Add searchable tag for chosen option
        if components.chosen:
            chosen_tag = components.chosen.lower().strip()[:50]
            ctx.tags.add(f"_decision:chosen:{chosen_tag}")

        return ctx


# ── Step 4: Auto-Tag + Metadata ──

# A8 T3.3: Confidence patterns for type classification
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "error": ["error", "bug", "crash", "exception", "traceback", "failed", "broken", "fix", "TypeError", "stack trace"],
    "decision": ["chose", "decided", "picked", "selected", "opted", "rejected", "went with", "instead of", "over"],
    "insight": ["discovered", "realized", "learned", "found that", "turns out", "root cause", "key insight", "pattern"],
    "workflow": ["step 1", "step 2", "after that", "pipeline", "process", "deploy", "release", "sequence"],
    "preference": ["prefer", "like", "favorite", "hate", "dislike", "always use", "never use"],
    "instruction": ["always", "never", "must", "make sure", "don't forget", "remember to"],
}


def _classify_confidence(content: str, suggested_type: str) -> float:
    """Score how well content matches its suggested type (0.0-1.0).

    Counts how many type-specific keywords appear in content.
    More matches = higher confidence.
    """
    keywords = _TYPE_KEYWORDS.get(suggested_type, [])
    if not keywords:
        return 0.5  # Unknown type — neutral confidence

    content_lower = content.lower()
    matches = sum(1 for kw in keywords if kw in content_lower)

    if matches == 0:
        return 0.2
    if matches == 1:
        return 0.6
    if matches == 2:
        return 0.8
    return 0.95  # 3+ matches = high confidence


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
        agent_type = ctx.effective_metadata.get("type", "")
        suggested = suggest_memory_type(ctx.content)

        if not agent_type:
            # A8 T3.3: Auto-classify with confidence check
            confidence = _classify_confidence(ctx.content, suggested.value)
            if confidence >= 0.7:
                ctx.effective_metadata["type"] = suggested.value
            else:
                # Low confidence — default to fact, store hint for agent
                ctx.effective_metadata["type"] = "fact"
                ctx.effective_metadata["_type_hint"] = (
                    f"Content may be '{suggested.value}' (confidence {confidence:.0%})"
                )
                ctx.effective_metadata["_type_confidence"] = round(confidence, 2)
        else:
            # Agent specified type — check for mismatch
            if agent_type != suggested.value and agent_type != "fact":
                confidence = _classify_confidence(ctx.content, agent_type)
                if confidence < 0.3:
                    ctx.effective_metadata["_type_hint"] = (
                        f"Content looks more like '{suggested.value}' than '{agent_type}'"
                    )

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

        # Skip dedup for very short content (high false-positive risk)
        if len(ctx.content.strip()) < 20:
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
                # A8 T3.2: Propagate dedup details for agent-facing feedback
                ctx.effective_metadata["_dedup_similarity"] = dedup_result.similarity_score
                ctx.effective_metadata["_dedup_tier"] = dedup_result.tier
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
            # Extract raw keywords from content for fidelity-aware recall
            from neural_memory.extraction.keywords import extract_keywords

            raw_kws = extract_keywords(ctx.content)[:10]

            anchor_neuron = Neuron.create(
                type=NeuronType.CONCEPT,
                content=ctx.content,
                metadata={
                    "is_anchor": True,
                    "timestamp": ctx.timestamp.isoformat(),
                    **({"_raw_keywords": raw_kws} if raw_kws else {}),
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

        # Anchor -> concept neurons (weight by keyword importance * IDF)
        weighted_keywords = extract_weighted_keywords(ctx.content, language=ctx.language)
        kw_weight_map: dict[str, float] = {}
        if weighted_keywords:
            kw_texts = [kw.text.lower() for kw in weighted_keywords]
            try:
                total_fibers = await storage.get_total_fiber_count()
                if not isinstance(total_fibers, (int, float)):
                    total_fibers = 0
            except Exception:
                total_fibers = 0
            if total_fibers >= 5:
                try:
                    df_map = await storage.get_keyword_df_batch(kw_texts)
                    if not isinstance(df_map, dict):
                        df_map = {}
                    idf_max = math.log(total_fibers + 1)
                    for kw in weighted_keywords:
                        df = df_map.get(kw.text.lower(), 0)
                        if not isinstance(df, (int, float)):
                            df = 0
                        idf_raw = math.log((total_fibers + 1) / (1 + df))
                        idf_factor = max(0.2, idf_raw / idf_max) if idf_max > 0 else 1.0
                        kw_weight_map[kw.text.lower()] = kw.weight * idf_factor
                except Exception:
                    kw_weight_map = {kw.text.lower(): kw.weight for kw in weighted_keywords}
            else:
                # Cold start: not enough corpus for meaningful IDF
                kw_weight_map = {kw.text.lower(): kw.weight for kw in weighted_keywords}
            # Update DF table for this memory's keywords
            try:
                await storage.increment_keyword_df(kw_texts)
            except Exception:
                pass  # Non-critical: DF update failure doesn't block encoding

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

        # Anchor → action neurons
        for action_neuron in ctx.action_neurons:
            synapses_to_add.append(
                Synapse.create(
                    source_id=anchor.id,
                    target_id=action_neuron.id,
                    type=SynapseType.INVOLVES,
                    weight=0.6,
                )
            )

        # Anchor → intent neurons
        for intent_neuron in ctx.intent_neurons:
            synapses_to_add.append(
                Synapse.create(
                    source_id=anchor.id,
                    target_id=intent_neuron.id,
                    type=SynapseType.INVOLVES,
                    weight=0.7,
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

        # Record deferred entity refs (lazy promotion B7)
        deferred_refs = getattr(ctx, "deferred_entity_refs", [])
        if deferred_refs:
            for entity_text in deferred_refs:
                try:
                    await storage.add_entity_ref(entity_text, anchor.id)
                except Exception:
                    logger.debug("Failed to add entity ref for %s", entity_text)

        # Retroactive linking: connect previous anchors to newly promoted entities
        neurons_created = getattr(ctx, "neurons_created", [])
        if getattr(config, "lazy_entity_enabled", True) and neurons_created:
            await _retroactive_entity_link(ctx, storage)

        return ctx


async def _retroactive_entity_link(
    ctx: PipelineContext,
    storage: NeuralStorage,
) -> None:
    """Link previous anchors to newly promoted entity neurons.

    When an entity reaches the promotion threshold, find all previous
    anchors that mentioned it (via entity_refs table) and create
    INVOLVES synapses retroactively.
    """
    for entity_neuron in ctx.entity_neurons:
        # Only retroactive-link neurons that were just created (not pre-existing)
        if entity_neuron not in ctx.neurons_created:
            continue
        try:
            prev_anchor_ids = await storage.get_entity_ref_fiber_ids(entity_neuron.content)
        except Exception:
            continue
        for anchor_id in prev_anchor_ids:
            # Skip current anchor (already linked above)
            if ctx.anchor_neuron and anchor_id == ctx.anchor_neuron.id:
                continue
            try:
                synapse = Synapse.create(
                    source_id=anchor_id,
                    target_id=entity_neuron.id,
                    type=SynapseType.INVOLVES,
                    weight=0.6,  # Lower weight for retroactive links
                )
                await storage.add_synapse(synapse)
                ctx.synapses_created.append(synapse)
            except Exception:
                logger.debug(
                    "Retroactive link failed: %s -> %s",
                    anchor_id,
                    entity_neuron.id,
                )


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


# ── Step 13b: Semantic Linking ──


class SemanticLinkingStep:
    """Link entity/concept neurons to existing similar neurons.

    Reduces orphan rate by cross-linking new neurons to previously stored
    neurons with matching content. Only creates RELATED_TO synapses.
    """

    MAX_NEURONS_PER_ENCODE: int = 20
    MAX_LINKS_PER_NEURON: int = 5

    @property
    def name(self) -> str:
        return "semantic_linking"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        # Collect entity and concept neurons from this encode
        linkable = [
            n
            for n in ctx.entity_neurons
            + ctx.concept_neurons
            + ctx.action_neurons
            + ctx.intent_neurons
            if n.content and len(n.content) >= 3
        ]
        if not linkable:
            return ctx

        new_ids = {
            n.id
            for n in ctx.entity_neurons
            + ctx.concept_neurons
            + ctx.time_neurons
            + ctx.action_neurons
            + ctx.intent_neurons
        }
        if ctx.anchor_neuron is not None:
            new_ids.add(ctx.anchor_neuron.id)

        for neuron in linkable[: self.MAX_NEURONS_PER_ENCODE]:
            try:
                existing = await storage.find_neurons(
                    type=neuron.type,
                    content_exact=neuron.content,
                    limit=self.MAX_LINKS_PER_NEURON + 1,
                )
            except Exception:
                logger.debug("Semantic linking lookup failed for %s", neuron.id)
                continue

            links_created = 0
            for target in existing:
                if target.id == neuron.id or target.id in new_ids:
                    continue
                if links_created >= self.MAX_LINKS_PER_NEURON:
                    break

                synapse = Synapse.create(
                    source_id=neuron.id,
                    target_id=target.id,
                    type=SynapseType.RELATED_TO,
                    weight=0.4,
                    metadata={"semantic_link": True},
                )
                try:
                    await storage.add_synapse(synapse)
                    ctx.neurons_linked.append(target.id)
                    links_created += 1
                except ValueError:
                    logger.debug("Synapse already exists, skipping")

        return ctx


# ── Step 14: Cross-Memory Linking ──


class CrossMemoryLinkStep:
    """Link new anchor to existing anchors that share entity neurons.

    When a new memory mentions entities that appear in older memories,
    creates RELATED_TO synapses between anchors. This enables multi-hop
    reasoning: Memory A → shared entity → Memory B.

    Brain test alignment:
    - Activation: anchor-to-anchor paths improve spreading activation reach
    - SA center: creates graph edges, not a search index
    - No-embedding: entity content match is keyword-based
    - Detail→Speed: specific entities = fewer, stronger links
    - Source traceable: metadata tracks which entity caused the link
    - Brain analogy: hippocampal memory binding via shared representations
    - Lifecycle: weak initial weight (0.3), survives only if reinforced
    """

    MAX_LINKS_PER_ENTITY: int = 5
    MAX_LINKS_PER_ENCODE: int = 15
    ENTITY_FREQUENCY_CAP: int = 50
    BASE_WEIGHT: float = 0.3
    WEIGHT_BONUS_PER_ENTITY: float = 0.1
    WEIGHT_CAP: float = 0.7

    @property
    def name(self) -> str:
        return "cross_memory_link"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        if ctx.anchor_neuron is None or not ctx.entity_neurons:
            return ctx

        anchor_id = ctx.anchor_neuron.id
        new_ids = {n.id for n in ctx.neurons_created}
        new_ids.add(anchor_id)

        # Track which old anchors we've already linked to + shared entity count
        linked_anchors: dict[str, int] = {}  # old_anchor_id → shared_entity_count
        total_links = 0

        for entity_neuron in ctx.entity_neurons:
            if len(linked_anchors) >= self.MAX_LINKS_PER_ENCODE:
                break

            # Find existing INVOLVES synapses pointing to this entity
            try:
                existing_synapses = await storage.get_synapses(
                    target_id=entity_neuron.id,
                    type=SynapseType.INVOLVES,
                )
            except Exception:
                logger.debug(
                    "Cross-memory link: lookup failed for entity %s",
                    entity_neuron.id,
                )
                continue

            # Skip "stop entities" that are too common (prevent hub explosion)
            if len(existing_synapses) > self.ENTITY_FREQUENCY_CAP:
                continue

            links_for_entity = 0
            for syn in existing_synapses:
                if links_for_entity >= self.MAX_LINKS_PER_ENTITY:
                    break
                if len(linked_anchors) >= self.MAX_LINKS_PER_ENCODE:
                    break

                old_anchor_id = syn.source_id
                # Skip self-links and links to neurons from this encode
                if old_anchor_id == anchor_id or old_anchor_id in new_ids:
                    continue

                # Count shared entities for weight calculation
                linked_anchors[old_anchor_id] = linked_anchors.get(old_anchor_id, 0) + 1
                links_for_entity += 1

        # Create synapses for accumulated anchor pairs
        for old_anchor_id, shared_count in linked_anchors.items():
            if total_links >= self.MAX_LINKS_PER_ENCODE:
                break

            weight = min(
                self.WEIGHT_CAP,
                self.BASE_WEIGHT + self.WEIGHT_BONUS_PER_ENTITY * (shared_count - 1),
            )
            synapse = Synapse.create(
                source_id=anchor_id,
                target_id=old_anchor_id,
                type=SynapseType.RELATED_TO,
                weight=weight,
                metadata={
                    "_cross_memory": True,
                    "_shared_entity_count": shared_count,
                },
            )
            try:
                await storage.add_synapse(synapse)
                ctx.synapses_created.append(synapse)
                ctx.neurons_linked.append(old_anchor_id)
                total_links += 1
            except ValueError:
                logger.debug("Cross-memory synapse already exists, skipping")

        return ctx


# ── Step 15: Build Fiber ──


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

        # Copy metadata to avoid leaking non-serializable objects into the fiber
        fiber_metadata = {k: v for k, v in ctx.effective_metadata.items() if k != "_pipeline_fiber"}

        # Propagate _verbatim flag if any neuron has structured cell data
        has_verbatim = any(n.metadata.get("_verbatim") for n in ctx.neurons_created)
        if has_verbatim:
            fiber_metadata = {**fiber_metadata, "_verbatim": True}

        # Encode-time token estimation: stored in metadata for budget-aware retrieval.
        # Uses word-count * 1.3 heuristic (same as retrieval_context._TOKEN_RATIO).
        if "estimated_tokens" not in fiber_metadata:
            token_estimate = int(len(ctx.content.split()) * 1.3)
            fiber_metadata = {**fiber_metadata, "estimated_tokens": token_estimate}

        fiber = Fiber.create(
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=ctx.anchor_neuron.id,
            time_start=ctx.timestamp,
            time_end=ctx.timestamp,
            auto_tags=ctx.auto_tags,
            agent_tags=ctx.agent_tags,
            metadata=fiber_metadata,
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

        # Record tag co-occurrence for drift detection
        if ctx.merged_tags and len(ctx.merged_tags) >= 2:
            try:
                await storage.record_tag_cooccurrence(ctx.merged_tags)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Tag co-occurrence recording failed (non-critical)", exc_info=True)

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
