"""Context-dependent retrieval — project-scoped scoring boost.

Memories encoded in the same project/topic context as the current query
receive a scoring boost, while cross-project memories get a mild penalty.
This mimics the brain's context-dependent memory effect where recall
is stronger when encoding and retrieval contexts match.

Neuroscience basis: encoding specificity principle (Tulving & Thomson, 1973)
— memory retrieval is enhanced when context at retrieval matches context
at encoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextFingerprint:
    """Lightweight context snapshot for encoding/retrieval matching."""

    project_name: str = ""
    dominant_topics: tuple[str, ...] = ()
    active_entities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "dominant_topics": list(self.dominant_topics),
            "active_entities": list(self.active_entities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextFingerprint:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            project_name=data.get("project_name", ""),
            dominant_topics=tuple(data.get("dominant_topics", [])),
            active_entities=tuple(data.get("active_entities", [])),
        )


def build_context_fingerprint(
    tags: set[str],
    entities: list[str] | None = None,
    project_name: str = "",
) -> ContextFingerprint:
    """Build a context fingerprint from available encoding context.

    Args:
        tags: Merged tags from the encoding pipeline
        entities: Entity names extracted during encoding
        project_name: Project name if available from session/config
    """
    # Use top tags as dominant topics (sorted for stability)
    topics = tuple(sorted(tags))[:10] if tags else ()
    ents = tuple(entities[:10]) if entities else ()

    return ContextFingerprint(
        project_name=project_name,
        dominant_topics=topics,
        active_entities=ents,
    )


def context_match_score(
    encoding_ctx: ContextFingerprint,
    retrieval_ctx: ContextFingerprint,
) -> float:
    """Score how well encoding and retrieval contexts match.

    Returns:
        Score multiplier in [0.5, 1.5]:
        - 1.0 = neutral (no context info or no match/mismatch)
        - >1.0 = context match (same project, overlapping topics)
        - <1.0 = context mismatch (different project)
    """
    if not encoding_ctx.dominant_topics and not retrieval_ctx.dominant_topics:
        return 1.0  # No context info → neutral

    score = 1.0

    # Project match: +0.2 (same) / -0.1 (different, both non-empty)
    if encoding_ctx.project_name and retrieval_ctx.project_name:
        if encoding_ctx.project_name.lower() == retrieval_ctx.project_name.lower():
            score += 0.2
        else:
            score -= 0.1

    # Topic overlap: +0.15 * Jaccard similarity
    if encoding_ctx.dominant_topics and retrieval_ctx.dominant_topics:
        enc_set = set(encoding_ctx.dominant_topics)
        ret_set = set(retrieval_ctx.dominant_topics)
        union = enc_set | ret_set
        if union:
            jaccard = len(enc_set & ret_set) / len(union)
            score += 0.15 * jaccard

    # Entity overlap: +0.1 * overlap ratio
    if encoding_ctx.active_entities and retrieval_ctx.active_entities:
        enc_ents = {e.lower() for e in encoding_ctx.active_entities}
        ret_ents = {e.lower() for e in retrieval_ctx.active_entities}
        union = enc_ents | ret_ents
        if union:
            overlap = len(enc_ents & ret_ents) / len(union)
            score += 0.1 * overlap

    return max(0.5, min(1.5, score))


@dataclass
class ContextFingerprintStep:
    """Pipeline step: store context fingerprint in fiber metadata.

    Must run BEFORE BuildFiberStep so the fingerprint lands in fiber.metadata.
    """

    @property
    def name(self) -> str:
        return "context_fingerprint"

    async def execute(
        self,
        ctx: Any,  # PipelineContext
        storage: Any,  # NeuralStorage
        config: Any,  # BrainConfig
    ) -> Any:
        if not getattr(config, "context_retrieval_enabled", True):
            return ctx

        entities = [n.content for n in getattr(ctx, "entity_neurons", [])]
        fingerprint = build_context_fingerprint(
            tags=getattr(ctx, "merged_tags", set()) or getattr(ctx, "auto_tags", set()),
            entities=entities,
        )
        ctx.effective_metadata["_context_fingerprint"] = fingerprint.to_dict()
        return ctx
