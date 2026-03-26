"""Memory encoder for converting experiences into neural structures."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron
from neural_memory.core.synapse import Synapse
from neural_memory.engine.arousal import ArousalStep
from neural_memory.engine.context_retrieval import ContextFingerprintStep
from neural_memory.engine.pipeline import Pipeline, PipelineContext
from neural_memory.engine.pipeline_steps import (
    AutoTagStep,
    BuildFiberStep,
    ConfirmatoryBoostStep,
    ConflictDetectionStep,
    CoOccurrenceStep,
    CreateAnchorStep,
    CreateSynapsesStep,
    CrossMemoryLinkStep,
    DedupCheckStep,
    EmotionStep,
    ExtractActionNeuronsStep,
    ExtractConceptNeuronsStep,
    ExtractEntityNeuronsStep,
    ExtractIntentNeuronsStep,
    ExtractTimeNeuronsStep,
    RelationExtractionStep,
    SemanticLinkingStep,
    StructuredDataEncoderStep,
    StructureDetectionStep,
    TemporalLinkingStep,
)
from neural_memory.engine.prediction_error import PredictionErrorStep
from neural_memory.engine.temporal_binding import TemporalBindingStep
from neural_memory.extraction.entities import EntityExtractor
from neural_memory.extraction.relations import RelationExtractor
from neural_memory.extraction.sentiment import SentimentExtractor
from neural_memory.extraction.temporal import TemporalExtractor
from neural_memory.utils.tag_normalizer import TagNormalizer
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.core.fiber import Fiber
    from neural_memory.engine.dedup.pipeline import DedupPipeline
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


def build_default_pipeline(
    temporal_extractor: TemporalExtractor,
    entity_extractor: EntityExtractor,
    relation_extractor: RelationExtractor,
    sentiment_extractor: SentimentExtractor,
    tag_normalizer: TagNormalizer,
    dedup_pipeline: DedupPipeline | None = None,
) -> Pipeline:
    """Build the default encoding pipeline with all 15 steps.

    This is the standard pipeline that reproduces the original monolithic
    ``encode()`` behavior. Users can customize by removing, replacing,
    or reordering steps.

    Args:
        temporal_extractor: Temporal extraction instance
        entity_extractor: Entity extraction instance
        relation_extractor: Relation extraction instance
        sentiment_extractor: Sentiment extraction instance
        tag_normalizer: Tag normalization instance
        dedup_pipeline: Optional dedup pipeline

    Returns:
        A Pipeline with all default steps.
    """
    return Pipeline(
        [
            ExtractTimeNeuronsStep(temporal_extractor=temporal_extractor),
            ExtractEntityNeuronsStep(entity_extractor=entity_extractor),
            ExtractConceptNeuronsStep(),
            ExtractActionNeuronsStep(),
            ExtractIntentNeuronsStep(),
            StructureDetectionStep(),
            AutoTagStep(tag_normalizer=tag_normalizer),
            DedupCheckStep(dedup_pipeline=dedup_pipeline),
            PredictionErrorStep(),
            CreateAnchorStep(),
            StructuredDataEncoderStep(),
            CreateSynapsesStep(),
            CoOccurrenceStep(),
            EmotionStep(sentiment_extractor=sentiment_extractor),
            ArousalStep(),
            RelationExtractionStep(relation_extractor=relation_extractor),
            ConfirmatoryBoostStep(),
            ConflictDetectionStep(),
            TemporalLinkingStep(),
            SemanticLinkingStep(),
            CrossMemoryLinkStep(),
            ContextFingerprintStep(),
            BuildFiberStep(),
            TemporalBindingStep(),
        ]
    )


_INSTRUCTION_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "we",
        "you",
        "they",
        "he",
        "she",
        "my",
        "our",
        "your",
        "their",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "such",
        "just",
        "than",
        "then",
        "when",
        "where",
        "which",
        "who",
        "what",
        "how",
        "why",
        "also",
        "very",
        "too",
        "up",
        "out",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "only",
        "own",
        "same",
        "other",
        "there",
        "here",
        "new",
        "now",
        "always",
        "never",
    }
)


def _extract_trigger_patterns(content: str, max_patterns: int = 5) -> list[str]:
    """Extract significant keywords from instruction content as trigger patterns.

    Takes the top N significant keywords (filtered against stop words) from
    the instruction text. These words will be used to boost the instruction
    during recall when the query overlaps with them.

    Args:
        content: The instruction text.
        max_patterns: Maximum number of trigger patterns to return.

    Returns:
        List of lowercase keyword strings.
    """
    # Normalize: lowercase, remove punctuation except apostrophes
    normalized = re.sub(r"[^\w\s']", " ", content.lower())
    words = normalized.split()
    seen: dict[str, int] = {}
    for word in words:
        word = word.strip("'")
        if len(word) >= 4 and word not in _INSTRUCTION_STOP_WORDS:
            seen[word] = seen.get(word, 0) + 1
    # Sort by frequency descending, then alphabetically for stability
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return [kw for kw, _ in ranked[:max_patterns]]


def _inject_instruction_metadata(
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge instruction tracking fields into metadata (non-destructive).

    Only adds keys that are not already present — preserves any existing
    instruction metadata set by the caller.

    Args:
        content: Instruction text (used for trigger extraction).
        metadata: Existing metadata dict.

    Returns:
        New metadata dict with instruction fields merged in.
    """
    defaults: dict[str, Any] = {
        "version": 1,
        "execution_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "success_rate": None,
        "last_executed_at": None,
        "failure_modes": [],
        "trigger_patterns": _extract_trigger_patterns(content),
        "refinement_history": [],
    }
    # Merge: existing keys win; only add missing ones
    merged = {**defaults, **metadata}
    return merged


class MemoryEncoder:
    """
    Encoder for converting experiences into neural structures.

    The encoder:
    1. Extracts neurons from content (time, entities, actions, concepts)
    2. Finds existing similar neurons for de-duplication
    3. Creates synapses based on relationships
    4. Bundles everything into a Fiber
    5. Auto-links with nearby temporal neurons

    Internally delegates to a composable :class:`Pipeline` of steps.
    The default pipeline reproduces the original behavior. Pass a custom
    ``pipeline`` to ``__init__`` to customize encoding.
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
        temporal_extractor: TemporalExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        dedup_pipeline: DedupPipeline | None = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        """
        Initialize the encoder.

        Args:
            storage: Storage backend
            config: Brain configuration
            temporal_extractor: Custom temporal extractor
            entity_extractor: Custom entity extractor
            relation_extractor: Custom relation extractor
            dedup_pipeline: Optional DedupPipeline for anchor deduplication
            pipeline: Custom pipeline (overrides default step composition)
        """
        self._storage = storage
        self._config = config
        self._temporal = temporal_extractor or TemporalExtractor()
        self._entity = entity_extractor or EntityExtractor()
        self._relation = relation_extractor or RelationExtractor()
        self._sentiment = SentimentExtractor()
        self._tag_normalizer = TagNormalizer()
        self._dedup_pipeline = dedup_pipeline

        self._pipeline = pipeline or build_default_pipeline(
            temporal_extractor=self._temporal,
            entity_extractor=self._entity,
            relation_extractor=self._relation,
            sentiment_extractor=self._sentiment,
            tag_normalizer=self._tag_normalizer,
            dedup_pipeline=self._dedup_pipeline,
        )

    @property
    def pipeline(self) -> Pipeline:
        """Access the encoding pipeline (read-only)."""
        return self._pipeline

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

        # Auto-populate instruction metadata for instruction/workflow types
        merged_metadata = dict(metadata or {})
        mem_type = merged_metadata.get("type", "")
        if mem_type in ("instruction", "workflow"):
            merged_metadata = _inject_instruction_metadata(content, merged_metadata)

        ctx = PipelineContext(
            content=content,
            timestamp=timestamp,
            metadata=merged_metadata,
            tags=tags or set(),
            language=language,
            skip_conflicts=skip_conflicts,
            skip_time_neurons=skip_time_neurons,
            initial_stage=initial_stage,
            salience_ceiling=salience_ceiling,
        )

        ctx = await self._pipeline.run(ctx, self._storage, self._config)

        # Extract fiber from context (set by BuildFiberStep) — use .get() to avoid mutation
        fiber = ctx.effective_metadata.get("_pipeline_fiber")
        if fiber is None:
            msg = "Pipeline did not produce a fiber (missing BuildFiberStep?)"
            raise RuntimeError(msg)

        # Post-encode: schema assimilation + interference detection (non-critical)
        if ctx.anchor_neuron is not None:
            await self._post_encode_neuro(ctx.anchor_neuron)

        return EncodingResult(
            fiber=fiber,
            neurons_created=ctx.neurons_created,
            neurons_linked=ctx.neurons_linked,
            synapses_created=ctx.synapses_created,
            conflicts_detected=ctx.conflicts_detected,
        )

    async def _post_encode_neuro(self, anchor: Neuron) -> None:
        """Run post-encode neuroscience hooks (schema assimilation + interference).

        Non-critical: failures are logged and swallowed so encoding always succeeds.
        Schema assimilation skips small brains (< schema_min_cluster_size) since
        there aren't enough neurons to form meaningful schemas.
        """
        # Schema assimilation: auto-wire when brain has enough memories
        schema_enabled = getattr(self._config, "schema_assimilation_enabled", False)
        if isinstance(schema_enabled, bool) and schema_enabled:
            try:
                # Skip small brains — not enough neurons for schema formation
                min_cluster = getattr(self._config, "schema_min_cluster_size", 10)
                min_cluster = int(min_cluster) if isinstance(min_cluster, (int, float)) else 10
                stats = await self._storage.get_stats(self._storage.brain_id or "")
                neuron_count = stats.get("neuron_count", 0)
                if neuron_count >= min_cluster:
                    from neural_memory.engine.schema_assimilation import assimilate_or_accommodate

                    await assimilate_or_accommodate(anchor, self._storage, self._config)
            except Exception:
                logger.debug("Post-encode schema assimilation failed (non-critical)", exc_info=True)

        # Interference detection: detect and resolve competing memories
        interference_enabled = getattr(self._config, "interference_detection_enabled", False)
        if isinstance(interference_enabled, bool) and interference_enabled:
            try:
                from neural_memory.engine.interference import (
                    detect_interference,
                    resolve_interference,
                )

                results = await detect_interference(anchor, self._storage, self._config)
                if results:
                    await resolve_interference(results, anchor, self._storage, self._config)
            except Exception:
                logger.debug(
                    "Post-encode interference detection failed (non-critical)", exc_info=True
                )
