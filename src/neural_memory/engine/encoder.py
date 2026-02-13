"""Memory encoder for converting experiences into neural structures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron
from neural_memory.core.synapse import Synapse
from neural_memory.engine.pipeline import Pipeline, PipelineContext
from neural_memory.engine.pipeline_steps import (
    AutoTagStep,
    BuildFiberStep,
    ConfirmatoryBoostStep,
    ConflictDetectionStep,
    CoOccurrenceStep,
    CreateAnchorStep,
    CreateSynapsesStep,
    DedupCheckStep,
    EmotionStep,
    ExtractConceptNeuronsStep,
    ExtractEntityNeuronsStep,
    ExtractTimeNeuronsStep,
    RelationExtractionStep,
    TemporalLinkingStep,
)
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
    """Build the default encoding pipeline with all 14 steps.

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
            AutoTagStep(tag_normalizer=tag_normalizer),
            DedupCheckStep(dedup_pipeline=dedup_pipeline),
            CreateAnchorStep(),
            CreateSynapsesStep(),
            CoOccurrenceStep(),
            EmotionStep(sentiment_extractor=sentiment_extractor),
            RelationExtractionStep(relation_extractor=relation_extractor),
            ConfirmatoryBoostStep(),
            ConflictDetectionStep(),
            TemporalLinkingStep(),
            BuildFiberStep(),
        ]
    )


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

        ctx = PipelineContext(
            content=content,
            timestamp=timestamp,
            metadata=dict(metadata or {}),
            tags=tags or set(),
            language=language,
            skip_conflicts=skip_conflicts,
            skip_time_neurons=skip_time_neurons,
            initial_stage=initial_stage,
            salience_ceiling=salience_ceiling,
        )

        ctx = await self._pipeline.run(ctx, self._storage, self._config)

        # Extract fiber from context (set by BuildFiberStep)
        fiber = ctx.effective_metadata.pop("_pipeline_fiber", None)
        if fiber is None:
            msg = "Pipeline did not produce a fiber (missing BuildFiberStep?)"
            raise RuntimeError(msg)

        return EncodingResult(
            fiber=fiber,
            neurons_created=ctx.neurons_created,
            neurons_linked=ctx.neurons_linked,
            synapses_created=ctx.synapses_created,
            conflicts_detected=ctx.conflicts_detected,
        )
