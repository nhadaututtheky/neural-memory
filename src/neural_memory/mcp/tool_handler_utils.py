"""Shared utilities for MCP tool handlers.

Module-level helpers used by multiple handler mixins.
Extracted from tool_handlers.py to keep individual handler files focused.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Max tags per recall query (remember allows 50 for storage, recall caps at 20 for filtering)
_MAX_RECALL_TAGS = 20
_MAX_TAG_LENGTH = 100


def _parse_tags(args: dict[str, Any], *, max_items: int = _MAX_RECALL_TAGS) -> set[str] | None:
    """Parse and validate tags from MCP tool arguments.

    Returns a set of valid tag strings, or None if no valid tags provided.
    """
    raw_tags = args.get("tags")
    if not raw_tags or not isinstance(raw_tags, list):
        return None
    tags = {t for t in raw_tags[:max_items] if isinstance(t, str) and 0 < len(t) <= _MAX_TAG_LENGTH}
    return tags or None


def _require_brain_id(storage: NeuralStorage) -> str:
    """Return the current brain ID or raise ValueError if not set."""
    brain_id = storage.brain_id
    if not brain_id:
        raise ValueError("No brain context set")
    return brain_id


async def _get_brain_or_error(
    storage: NeuralStorage,
) -> tuple[Any, dict[str, Any] | None]:
    """Get brain object or return (None, error_dict)."""
    try:
        brain_id = _require_brain_id(storage)
    except ValueError:
        return None, {"error": "No brain configured"}
    brain = await storage.get_brain(brain_id)
    if not brain:
        return None, {"error": "No brain configured"}
    return brain, None


async def _build_citation_audit(
    storage: NeuralStorage,
    neuron_id: str,
    include_citations: bool = True,
) -> dict[str, Any]:
    """Build citation and audit trail for a neuron from its synapses.

    Looks up SOURCE_OF, STORED_BY, VERIFIED_AT, APPROVED_BY synapses
    connected to the neuron and returns citation + audit dicts.
    """
    from neural_memory.core.synapse import SynapseType

    result: dict[str, Any] = {}

    # Fetch incoming synapses for this neuron
    synapses = await storage.get_synapses(target_id=neuron_id)

    # Build citation from SOURCE_OF synapse
    if include_citations:
        source_synapses = [s for s in synapses if s.type == SynapseType.SOURCE_OF]
        if source_synapses:
            source_syn = source_synapses[0]
            try:
                source_obj = await storage.get_source(source_syn.source_id)
                if source_obj:
                    from neural_memory.engine.citation import (
                        CitationFormat,
                        CitationInput,
                        format_citation,
                    )

                    citation_input = CitationInput(
                        source_name=source_obj.name,
                        source_type=source_obj.source_type.value,
                        source_version=source_obj.version,
                        effective_date=(
                            source_obj.effective_date.isoformat()
                            if source_obj.effective_date
                            else None
                        ),
                        neuron_id=neuron_id,
                        metadata=source_obj.metadata,
                    )
                    result["citation"] = {
                        "inline": format_citation(citation_input, CitationFormat.INLINE),
                        "footnote": format_citation(citation_input, CitationFormat.FOOTNOTE),
                        "source_id": source_obj.id,
                        "source_name": source_obj.name,
                        "source_type": source_obj.source_type.value,
                    }
            except Exception:
                logger.debug("Citation generation failed", exc_info=True)

    # Build audit trail from STORED_BY, VERIFIED_AT, APPROVED_BY synapses
    stored_by_syns = [s for s in synapses if s.type == SynapseType.STORED_BY]
    verified_syns = [s for s in synapses if s.type == SynapseType.VERIFIED_AT]
    approved_syns = [s for s in synapses if s.type == SynapseType.APPROVED_BY]

    if stored_by_syns or verified_syns or approved_syns:
        audit: dict[str, Any] = {}
        if stored_by_syns:
            syn = stored_by_syns[0]
            audit["stored_by"] = syn.metadata.get("actor", syn.source_id)
            audit["stored_at"] = syn.created_at.isoformat() if syn.created_at else None
        if verified_syns:
            syn = verified_syns[0]
            audit["verified"] = True
            audit["verified_by"] = syn.metadata.get("actor", syn.source_id)
            audit["verified_at"] = syn.created_at.isoformat() if syn.created_at else None
        else:
            audit["verified"] = False
        if approved_syns:
            syn = approved_syns[0]
            audit["approved_by"] = syn.metadata.get("actor", syn.source_id)
            audit["approved_at"] = syn.created_at.isoformat() if syn.created_at else None
        result["audit"] = audit

    return result
