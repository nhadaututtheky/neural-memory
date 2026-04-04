"""Decision intelligence — extract structured components and detect overlaps.

When a DECISION-type memory is stored, this module:
1. Extracts structured fields (chosen, alternatives, reasoning, confidence, context_tags)
2. Finds overlapping prior decisions for contradiction/evolution tracking
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionComponents:
    """Structured decision components extracted from content or context."""

    chosen: str
    rejected_alternatives: list[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: str = ""
    context_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "rejected_alternatives": list(self.rejected_alternatives),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "context_tags": list(self.context_tags),
        }


@dataclass(frozen=True)
class DecisionOverlap:
    """An overlapping prior decision found during storage."""

    fiber_id: str
    content_preview: str
    overlap_score: float
    relationship: str  # "confirms" | "contradicts" | "evolves"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fiber_id": self.fiber_id,
            "content_preview": self.content_preview,
            "overlap_score": round(self.overlap_score, 3),
            "relationship": self.relationship,
        }


# --- Extraction ---

# Regex patterns for heuristic extraction from content
_CHOSE_OVER = re.compile(
    r"(?:chose|chosen|picked|selected|went with|going with)\s+(.+?)\s+(?:over|instead of|rather than)\s+(.+?)(?:\s+because\b|\.|,|$)",
    re.IGNORECASE,
)
_DECIDED_BECAUSE = re.compile(
    r"(?:decided to|chose|choosing)\s+(.+?)\s+because\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)
_REJECTED = re.compile(
    r"(?:rejected|ruled out|dismissed|dropped)\s+(.+?)(?:\s+(?:because|due to|since)\s+(.+?))?(?:\.|,|$)",
    re.IGNORECASE,
)


def extract_decision_components(
    content: str,
    context: dict[str, Any] | None = None,
) -> DecisionComponents | None:
    """Extract structured decision components from content and/or context dict.

    Priority: structured context dict > regex heuristics on content.
    Returns None if no decision structure can be extracted.
    """
    # Try structured context first
    if context:
        chosen = _str_or_none(context.get("chosen"))
        if chosen:
            alternatives_raw = context.get("alternatives") or context.get("rejected") or []
            if isinstance(alternatives_raw, str):
                alternatives_raw = [a.strip() for a in alternatives_raw.split(",") if a.strip()]
            return DecisionComponents(
                chosen=chosen,
                rejected_alternatives=list(alternatives_raw),
                reasoning=_str_or_none(context.get("reason") or context.get("reasoning")) or "",
                confidence=_str_or_none(context.get("confidence")) or "",
                context_tags=_extract_context_tags(context),
            )

    # Fallback: regex heuristics on content
    if not content:
        return None

    # Pattern 1: "chose X over Y"
    match = _CHOSE_OVER.search(content)
    if match:
        chosen = match.group(1).strip()
        rejected = [alt.strip() for alt in match.group(2).split(",") if alt.strip()]
        reasoning = _extract_reasoning_suffix(content, match.end())
        return DecisionComponents(
            chosen=chosen,
            rejected_alternatives=rejected,
            reasoning=reasoning,
        )

    # Pattern 2: "decided to X because Y"
    match = _DECIDED_BECAUSE.search(content)
    if match:
        return DecisionComponents(
            chosen=match.group(1).strip(),
            reasoning=match.group(2).strip(),
        )

    # Pattern 3: standalone rejection mentions
    rejected_items: list[str] = []
    reasoning_parts: list[str] = []
    for match in _REJECTED.finditer(content):
        rejected_items.append(match.group(1).strip())
        if match.group(2):
            reasoning_parts.append(match.group(2).strip())

    if rejected_items:
        return DecisionComponents(
            chosen="",  # No explicit chosen from rejection-only pattern
            rejected_alternatives=rejected_items,
            reasoning="; ".join(reasoning_parts),
        )

    return None


def _str_or_none(val: Any) -> str | None:
    """Coerce to str or return None for falsy values."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _extract_context_tags(context: dict[str, Any]) -> list[str]:
    """Extract context tags from a context dict."""
    tags: list[str] = []
    if "tags" in context and isinstance(context["tags"], list):
        tags.extend(str(t).lower() for t in context["tags"] if t)
    if "project" in context:
        tags.append(str(context["project"]).lower())
    if "domain" in context:
        tags.append(str(context["domain"]).lower())
    return tags


def _extract_reasoning_suffix(content: str, match_end: int) -> str:
    """Try to extract 'because ...' reasoning after the match position."""
    remainder = content[match_end:].strip()
    because_match = re.match(r"because\s+(.+?)(?:\.|$)", remainder, re.IGNORECASE)
    if because_match:
        return because_match.group(1).strip()
    return ""


# --- Overlap scoring ---


async def find_overlapping_decisions(
    storage: NeuralStorage,
    new_components: DecisionComponents,
    new_tags: set[str] | None = None,
    limit: int = 10,
) -> list[DecisionOverlap]:
    """Find prior DECISION memories that overlap with a new decision.

    Scores overlap via tag intersection (Jaccard) + text similarity on chosen/alternatives.
    Returns top matches above threshold (0.3), classified as confirms/contradicts/evolves.
    """
    from neural_memory.core.memory_types import MemoryType

    existing = await storage.find_typed_memories(memory_type=MemoryType.DECISION, limit=200)
    if not existing:
        return []

    new_chosen_lower = new_components.chosen.lower().strip()
    new_rejected_lower = {a.lower().strip() for a in new_components.rejected_alternatives}
    new_tag_set = {t.lower() for t in (new_tags or set())}
    new_context_tags = {t.lower() for t in new_components.context_tags}
    all_new_tags = new_tag_set | new_context_tags

    overlaps: list[DecisionOverlap] = []

    for mem in existing:
        fiber = await storage.get_fiber(mem.fiber_id)
        if fiber is None or not fiber.anchor_neuron_id:
            continue

        neuron = await storage.get_neuron(fiber.anchor_neuron_id)
        if neuron is None:
            continue

        old_decision: dict[str, Any] = neuron.metadata.get("_decision", {})
        old_chosen = str(old_decision.get("chosen", "")).lower().strip()
        old_rejected = {
            str(a).lower().strip() for a in old_decision.get("rejected_alternatives", [])
        }
        old_context_tags = {str(t).lower() for t in old_decision.get("context_tags", [])}

        # --- Score ---
        # 1. Tag overlap (Jaccard on context_tags + memory tags)
        old_tag_set = old_context_tags | {str(t).lower() for t in (mem.tags or set())}
        tag_union = all_new_tags | old_tag_set
        tag_intersection = all_new_tags & old_tag_set
        tag_score = len(tag_intersection) / len(tag_union) if tag_union else 0.0

        # 2. Text similarity on chosen/alternatives (token overlap)
        text_score = _token_overlap_score(
            new_chosen_lower, new_rejected_lower, old_chosen, old_rejected
        )

        # Combined score (weighted average)
        overlap_score = 0.4 * tag_score + 0.6 * text_score
        if overlap_score < 0.3:
            continue

        # --- Classify relationship ---
        relationship = _classify_relationship(
            new_chosen_lower, new_rejected_lower, old_chosen, old_rejected
        )

        content_preview = neuron.content[:120] if neuron.content else ""
        overlaps.append(
            DecisionOverlap(
                fiber_id=mem.fiber_id,
                content_preview=content_preview,
                overlap_score=overlap_score,
                relationship=relationship,
            )
        )

    # Sort by score descending, cap at limit
    overlaps.sort(key=lambda o: o.overlap_score, reverse=True)
    return overlaps[:limit]


def _token_overlap_score(
    new_chosen: str,
    new_rejected: set[str],
    old_chosen: str,
    old_rejected: set[str],
) -> float:
    """Score text similarity between two decisions based on token overlap."""
    new_tokens = _tokenize(new_chosen) | _tokenize_set(new_rejected)
    old_tokens = _tokenize(old_chosen) | _tokenize_set(old_rejected)

    if not new_tokens and not old_tokens:
        return 0.0

    union = new_tokens | old_tokens
    intersection = new_tokens & old_tokens
    return len(intersection) / len(union) if union else 0.0


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return {w for w in re.split(r"\W+", text.lower()) if len(w) > 1}


def _tokenize_set(items: set[str]) -> set[str]:
    """Tokenize a set of strings into a single token set."""
    tokens: set[str] = set()
    for item in items:
        tokens |= _tokenize(item)
    return tokens


def _classify_relationship(
    new_chosen: str,
    new_rejected: set[str],
    old_chosen: str,
    old_rejected: set[str],
) -> str:
    """Classify the relationship between two decisions.

    - confirms: same chosen option
    - contradicts: new chosen was previously rejected (or vice versa)
    - evolves: different chosen but overlapping domain
    """
    if not new_chosen or not old_chosen:
        return "evolves"

    # Same chosen → confirms
    new_chosen_tokens = _tokenize(new_chosen)
    old_chosen_tokens = _tokenize(old_chosen)
    if new_chosen_tokens and old_chosen_tokens:
        chosen_overlap = len(new_chosen_tokens & old_chosen_tokens) / max(
            len(new_chosen_tokens), len(old_chosen_tokens)
        )
        if chosen_overlap >= 0.6:
            return "confirms"

    # New chosen was previously rejected → contradicts
    if new_chosen and old_rejected:
        for rejected in old_rejected:
            rejected_tokens = _tokenize(rejected)
            if rejected_tokens and new_chosen_tokens:
                overlap = len(new_chosen_tokens & rejected_tokens) / max(
                    len(new_chosen_tokens), len(rejected_tokens)
                )
                if overlap >= 0.6:
                    return "contradicts"

    # Old chosen is now in new rejected → contradicts
    if old_chosen and new_rejected:
        for rejected in new_rejected:
            rejected_tokens = _tokenize(rejected)
            if rejected_tokens and old_chosen_tokens:
                overlap = len(old_chosen_tokens & rejected_tokens) / max(
                    len(old_chosen_tokens), len(rejected_tokens)
                )
                if overlap >= 0.6:
                    return "contradicts"

    return "evolves"
