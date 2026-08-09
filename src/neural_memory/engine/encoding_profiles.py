"""Encoding profiles — lean (minimal + outbox) vs cognitive (full pipeline).

Compatibility policy:
- Existing brains with missing ``encoding_profile`` key → cognitive (current).
- Fresh installs / explicit new profiles may choose lean.
- High-priority decision/hypothesis/instruction may force cognitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from neural_memory.engine.pipeline import Pipeline
from neural_memory.engine.pipeline_steps import (
    AutoTagStep,
    BuildFiberStep,
    CreateAnchorStep,
    CreateSynapsesStep,
    DedupCheckStep,
    ExtractConceptNeuronsStep,
    ExtractEntityNeuronsStep,
    ExtractTimeNeuronsStep,
)
from neural_memory.extraction.entities import EntityExtractor
from neural_memory.extraction.temporal import TemporalExtractor
from neural_memory.utils.tag_normalizer import TagNormalizer


class EncodingProfile(StrEnum):
    """Write-path profile controlling ack latency vs cognitive depth."""

    LEAN = "lean"
    COGNITIVE = "cognitive"


# Memory types that force cognitive encoding even under lean default
_COGNITIVE_FORCE_TYPES: frozenset[str] = frozenset(
    {
        "decision",
        "hypothesis",
        "instruction",
        "boundary",
    }
)


@dataclass(frozen=True)
class ProfileResolution:
    """Resolved encoding profile for one write."""

    profile: EncodingProfile
    async_enrichment: bool
    reason: str


def parse_encoding_profile(value: str | None) -> EncodingProfile:
    """Parse profile string; invalid values raise ValueError."""
    if value is None or value == "":
        return EncodingProfile.COGNITIVE  # missing key → preserve current behavior
    normalized = str(value).strip().lower()
    try:
        return EncodingProfile(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported encoding_profile {value!r}; expected lean|cognitive"
        ) from exc


def resolve_profile(
    *,
    configured: str | EncodingProfile | None,
    async_enrichment: bool | None,
    memory_type: str | None = None,
    priority: int | None = None,
    force_cognitive: bool = False,
) -> ProfileResolution:
    """Resolve effective profile for a single encode.

    Rules:
    - Missing configured profile → cognitive + sync (compatibility).
    - force_cognitive / high-priority decision types → cognitive.
    - lean profile enables async enrichment when flag is True.
    """
    if isinstance(configured, EncodingProfile):
        profile = configured
    else:
        profile = parse_encoding_profile(configured if configured is not None else None)

    mem_type = (memory_type or "").strip().lower()
    if force_cognitive or mem_type in _COGNITIVE_FORCE_TYPES:
        return ProfileResolution(
            profile=EncodingProfile.COGNITIVE,
            async_enrichment=False,
            reason="high_priority_type" if mem_type else "force_cognitive",
        )

    if priority is not None and priority >= 9 and profile is EncodingProfile.LEAN:
        return ProfileResolution(
            profile=EncodingProfile.COGNITIVE,
            async_enrichment=False,
            reason="high_priority",
        )

    if profile is EncodingProfile.LEAN:
        # New lean profiles default async enrichment on unless explicitly disabled
        async_flag = True if async_enrichment is None else bool(async_enrichment)
        return ProfileResolution(
            profile=EncodingProfile.LEAN,
            async_enrichment=async_flag,
            reason="lean_default",
        )

    # Cognitive path: preserve synchronous enrichment unless opted in
    async_flag = False if async_enrichment is None else bool(async_enrichment)
    return ProfileResolution(
        profile=EncodingProfile.COGNITIVE,
        async_enrichment=async_flag,
        reason="cognitive_compat" if configured in (None, "") else "cognitive_explicit",
    )


def build_lean_pipeline(
    *,
    temporal_extractor: TemporalExtractor | None = None,
    entity_extractor: EntityExtractor | None = None,
    tag_normalizer: TagNormalizer | None = None,
    dedup_pipeline: Any | None = None,
) -> Pipeline:
    """Minimal searchable write path: time/entity/concept → anchor → fiber.

    Heavy stages (relations, conflict, temporal binding, embedding) are
    deferred to the enrichment outbox when async_enrichment is enabled.
    """
    temporal_extractor = temporal_extractor or TemporalExtractor()
    entity_extractor = entity_extractor or EntityExtractor()
    tag_normalizer = tag_normalizer or TagNormalizer()

    return Pipeline(
        [
            ExtractTimeNeuronsStep(temporal_extractor=temporal_extractor),
            ExtractEntityNeuronsStep(entity_extractor=entity_extractor),
            ExtractConceptNeuronsStep(),
            AutoTagStep(tag_normalizer=tag_normalizer),
            DedupCheckStep(dedup_pipeline=dedup_pipeline),
            CreateAnchorStep(),
            CreateSynapsesStep(),
            BuildFiberStep(),
        ]
    )


def build_pipeline(
    profile: EncodingProfile | str,
    *,
    build_cognitive: Any | None = None,
    **lean_kwargs: Any,
) -> Pipeline:
    """Build pipeline for the given profile.

    Args:
        profile: lean or cognitive.
        build_cognitive: Callable returning the full cognitive pipeline.
            Required when profile is cognitive.
        **lean_kwargs: Forwarded to ``build_lean_pipeline``.
    """
    resolved = (
        profile if isinstance(profile, EncodingProfile) else parse_encoding_profile(str(profile))
    )
    if resolved is EncodingProfile.LEAN:
        return build_lean_pipeline(**lean_kwargs)
    if build_cognitive is None:
        raise ValueError("build_cognitive callable required for cognitive profile")
    result = build_cognitive()
    if not isinstance(result, Pipeline):
        raise TypeError("build_cognitive must return a Pipeline")
    return result
