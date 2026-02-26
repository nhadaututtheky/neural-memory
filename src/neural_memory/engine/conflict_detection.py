"""Real-time conflict detection for contradictory memories.

Detects two conflict types at encode time (no LLM needed):

1. FACTUAL_CONTRADICTION: "We use PostgreSQL" vs "We use MySQL"
   - Same subject X, different predicate Y in "X is/uses/chose Y" patterns
   - Tag overlap > 50% confirms same topic

2. DECISION_REVERSAL: New DECISION with overlapping tags but different conclusion
   - Detects when a decision is reversed without explicit acknowledgment

Resolution actions:
- Reduce existing neuron confidence via anti-Hebbian update
- Mark existing neuron metadata: _disputed: true
- Create CONTRADICTS synapse between old and new neurons
- If confidence drops below threshold: mark _superseded: true
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from enum import StrEnum
from typing import TYPE_CHECKING

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.learning_rule import LearningConfig, anti_hebbian_update
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

# Shared stop words used by both _extract_search_terms and _extract_implicit_tags.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
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
        "shall",
        "can",
        "we",
        "i",
        "they",
        "you",
        "he",
        "she",
        "it",
        "our",
        "my",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "that",
        "this",
        "not",
        "but",
        "and",
        "or",
        "if",
        "then",
        "use",
        "using",
        "used",
        "chose",
        "decided",
    }
)


class ConflictType(StrEnum):
    """Types of memory conflicts."""

    FACTUAL_CONTRADICTION = "factual_contradiction"
    DECISION_REVERSAL = "decision_reversal"


@dataclass(frozen=True)
class Conflict:
    """A detected conflict between an existing and new memory.

    Attributes:
        type: The kind of conflict detected
        existing_neuron_id: ID of the existing conflicting neuron
        existing_content: Content of the existing neuron
        new_content: Content of the new memory
        confidence: How confident we are this is a real conflict (0-1)
        subject: The shared subject of the contradiction
        existing_predicate: The predicate in the existing memory
        new_predicate: The predicate in the new memory
    """

    type: ConflictType
    existing_neuron_id: str
    existing_content: str
    new_content: str
    confidence: float
    subject: str = ""
    existing_predicate: str = ""
    new_predicate: str = ""


@dataclass(frozen=True)
class ConflictResolution:
    """Actions taken to resolve a detected conflict.

    Attributes:
        conflict: The original conflict
        contradicts_synapse: The CONTRADICTS synapse created
        existing_neuron_updated: The updated existing neuron (with _disputed)
        confidence_reduced_by: How much confidence was reduced
        superseded: Whether the existing neuron was marked as superseded
    """

    conflict: Conflict
    contradicts_synapse: Synapse
    existing_neuron_updated: Neuron
    confidence_reduced_by: float
    superseded: bool


@dataclass
class ConflictReport:
    """Summary of conflict detection and resolution for an encoding.

    Attributes:
        conflicts_detected: Number of conflicts found
        resolutions_applied: Number of resolutions applied
        neurons_disputed: Number of neurons marked as disputed
        neurons_superseded: Number of neurons marked as superseded
        conflicts: Detailed conflict list
        resolutions: Detailed resolution list
    """

    conflicts_detected: int = 0
    resolutions_applied: int = 0
    neurons_disputed: int = 0
    neurons_superseded: int = 0
    conflicts: list[Conflict] = field(default_factory=list)
    resolutions: list[ConflictResolution] = field(default_factory=list)


# Regex patterns for extracting subject-predicate relationships.
# Matches patterns like "X is Y", "X uses Y", "X chose Y", "X selected Y"
_PREDICATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:we|i|they|the team|our team|the project)\s+"
        r"(?:use|uses|chose|selected|picked|decided on|switched to|migrated to|adopted)\s+"
        r"(.+?)(?:\s+(?:for|in|as|on|with|at|from|because|instead|rather)\s|\.|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(.+?)\s+(?:is|are|was|were)\s+(?:using|running|built with|powered by)\s+"
        r"(.+?)(?:\s+(?:for|in|as|on|with|at|from|because)\s|\.|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:we|i|they)\s+(?:decided|agreed|concluded)\s+(?:to\s+)?(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
]


@dataclass(frozen=True)
class _PredicateExtraction:
    """Internal: extracted subject-predicate pair from content."""

    subject: str
    predicate: str
    pattern_index: int


def _extract_predicates(content: str) -> list[_PredicateExtraction]:
    """Extract subject-predicate pairs from content using regex patterns."""
    results: list[_PredicateExtraction] = []

    for idx, pattern in enumerate(_PREDICATE_PATTERNS):
        for match in pattern.finditer(content):
            groups = match.groups()
            if len(groups) >= 2:
                subject = groups[0].strip().lower()
                predicate = groups[1].strip().lower()
            elif len(groups) == 1:
                # For patterns where subject is implicit (we/I/they)
                subject = "_implicit_agent"
                predicate = groups[0].strip().lower()
            else:
                continue

            if predicate and len(predicate) >= 2:
                results.append(
                    _PredicateExtraction(
                        subject=subject,
                        predicate=predicate,
                        pattern_index=idx,
                    )
                )

    return results


def _tag_overlap(tags_a: set[str], tags_b: set[str]) -> float:
    """Compute Jaccard similarity between two tag sets."""
    if not tags_a or not tags_b:
        return 0.0
    intersection = len(tags_a & tags_b)
    union = len(tags_a | tags_b)
    return intersection / union if union > 0 else 0.0


async def detect_conflicts(
    content: str,
    tags: set[str],
    storage: NeuralStorage,
    memory_type: str = "",
    tag_overlap_threshold: float = 0.15,
    max_candidates: int = 20,
) -> list[Conflict]:
    """Detect conflicts between new content and existing memories.

    Checks for:
    1. Factual contradictions via predicate extraction
    2. Decision reversals via tag overlap + different conclusions

    Args:
        content: The new memory content to check
        tags: Tags for the new memory
        storage: Storage backend for querying existing memories
        memory_type: Type of the new memory (e.g., "decision")
        tag_overlap_threshold: Minimum Jaccard similarity for tag match
        max_candidates: Maximum existing neurons to check

    Returns:
        List of detected conflicts (may be empty)
    """
    conflicts: list[Conflict] = []

    new_predicates = _extract_predicates(content)

    # Find existing neurons with overlapping content
    # Search by key terms from the new content
    search_terms = _extract_search_terms(content)

    candidates: list[Neuron] = []
    # Batch-fetch candidates for all search terms in parallel (avoid N+1)
    term_tasks = [
        storage.find_neurons(content_contains=term, limit=max_candidates)
        for term in search_terms[:5]
    ]
    term_results = await asyncio.gather(*term_tasks) if term_tasks else []
    seen_ids: set[str] = set()
    for found in term_results:
        for neuron in found:
            if neuron.content != content and neuron.id not in seen_ids:
                seen_ids.add(neuron.id)
                candidates.append(neuron)

    # Check each candidate for conflicts
    for candidate in candidates[:max_candidates]:
        # Skip TIME neurons — they can't contradict
        if candidate.type == NeuronType.TIME:
            continue

        # Already disputed or resolved — don't re-flag
        if candidate.metadata.get("_disputed") or candidate.metadata.get("_conflict_resolved"):
            continue

        existing_predicates = _extract_predicates(candidate.content)

        # Check for factual contradictions
        for new_pred in new_predicates:
            for existing_pred in existing_predicates:
                if _subjects_match(new_pred.subject, existing_pred.subject):
                    if _predicates_conflict(new_pred.predicate, existing_pred.predicate):
                        # Verify tag overlap if tags available
                        candidate_tags = _extract_implicit_tags(candidate)
                        overlap = (
                            _tag_overlap(tags, candidate_tags) if tags and candidate_tags else 0.5
                        )

                        if overlap >= tag_overlap_threshold or (not tags and not candidate_tags):
                            conflicts.append(
                                Conflict(
                                    type=ConflictType.FACTUAL_CONTRADICTION,
                                    existing_neuron_id=candidate.id,
                                    existing_content=candidate.content,
                                    new_content=content,
                                    confidence=min(1.0, 0.5 + overlap),
                                    subject=new_pred.subject,
                                    existing_predicate=existing_pred.predicate,
                                    new_predicate=new_pred.predicate,
                                )
                            )

        # Check for decision reversals
        if memory_type == "decision" or _is_decision_content(content):
            if _is_decision_content(candidate.content):
                candidate_tags = _extract_implicit_tags(candidate)
                overlap = _tag_overlap(tags, candidate_tags) if tags and candidate_tags else 0.0

                if overlap >= tag_overlap_threshold:
                    # Different content + overlapping tags = potential reversal
                    if not _content_agrees(content, candidate.content):
                        conflicts.append(
                            Conflict(
                                type=ConflictType.DECISION_REVERSAL,
                                existing_neuron_id=candidate.id,
                                existing_content=candidate.content,
                                new_content=content,
                                confidence=min(1.0, 0.4 + overlap),
                            )
                        )

    return conflicts


async def resolve_conflicts(
    conflicts: list[Conflict],
    new_neuron_id: str,
    storage: NeuralStorage,
    confidence_delta: float = 0.3,
    supersede_threshold: float = 0.2,
    existing_memory_type: str = "",
) -> list[ConflictResolution]:
    """Apply resolution actions for detected conflicts.

    For each conflict:
    1. Reduce existing neuron confidence via anti-Hebbian
    2. Mark _disputed: true in metadata
    3. Create CONTRADICTS synapse
    4. If confidence drops below threshold: mark _superseded
    5. If existing is ERROR type: create RESOLVED_BY synapse, mark resolved,
       and apply stronger activation demotion (error resolution learning)

    Args:
        conflicts: List of detected conflicts
        new_neuron_id: ID of the new neuron that conflicts
        storage: Storage backend
        confidence_delta: How much to reduce confidence
        supersede_threshold: Confidence below which to mark superseded
        existing_memory_type: Type of the existing memory (e.g. "error")

    Returns:
        List of resolution actions taken
    """
    resolutions: list[ConflictResolution] = []

    for conflict in conflicts:
        # Determine if existing memory is an error (from param or neuron metadata)
        existing_neuron = await storage.get_neuron(conflict.existing_neuron_id)
        if existing_neuron is None:
            continue

        is_error_resolution = _is_error_memory(existing_neuron, existing_memory_type)

        # 1. Get existing neuron state for confidence reduction
        existing_state = await storage.get_neuron_state(conflict.existing_neuron_id)
        current_confidence = existing_state.activation_level if existing_state else 0.5

        # 2. Compute anti-Hebbian reduction
        # Error resolution uses stronger demotion (2x learning rate)
        effective_delta = confidence_delta * 2.0 if is_error_resolution else confidence_delta
        update = anti_hebbian_update(
            current_weight=current_confidence,
            strength=conflict.confidence,
            config=LearningConfig(learning_rate=effective_delta),
        )
        confidence_reduced = abs(update.delta)

        # For error resolution, ensure at least 50% reduction
        if is_error_resolution:
            max_allowed = current_confidence * 0.5
            effective_activation = min(update.new_weight, max_allowed)
        else:
            effective_activation = update.new_weight

        # 3. Update neuron state with reduced confidence
        if existing_state:
            new_state = dc_replace(existing_state, activation_level=effective_activation)
            await storage.update_neuron_state(new_state)

        # 4. Mark existing neuron as disputed (or resolved for errors)
        is_superseded = effective_activation < supersede_threshold

        if is_error_resolution:
            updated_neuron = existing_neuron.with_metadata(
                _disputed=True,
                _disputed_at=utcnow().isoformat(),
                _disputed_by=new_neuron_id,
                _superseded=is_superseded,
                _pre_dispute_activation=current_confidence,
                _conflict_resolved=True,
                _resolved_by=new_neuron_id,
            )
        else:
            updated_neuron = existing_neuron.with_metadata(
                _disputed=True,
                _disputed_at=utcnow().isoformat(),
                _disputed_by=new_neuron_id,
                _superseded=is_superseded,
                _pre_dispute_activation=current_confidence,
            )
        await storage.update_neuron(updated_neuron)

        # 5. Create CONTRADICTS synapse
        contradicts_synapse = Synapse.create(
            source_id=new_neuron_id,
            target_id=conflict.existing_neuron_id,
            type=SynapseType.CONTRADICTS,
            weight=conflict.confidence,
            metadata={
                "conflict_type": conflict.type.value,
                "subject": conflict.subject,
                "detected_at": utcnow().isoformat(),
            },
        )

        try:
            await storage.add_synapse(contradicts_synapse)
        except ValueError:
            # Synapse may already exist if same pair conflicts multiple ways
            pass

        # 6. Error Resolution Learning: create RESOLVED_BY synapse
        if is_error_resolution:
            resolved_by_synapse = Synapse.create(
                source_id=new_neuron_id,
                target_id=conflict.existing_neuron_id,
                type=SynapseType.RESOLVED_BY,
                weight=conflict.confidence,
                metadata={
                    "error_resolution": True,
                    "resolved_at": utcnow().isoformat(),
                    "original_error": conflict.existing_content[:200],
                },
            )
            try:
                await storage.add_synapse(resolved_by_synapse)
            except ValueError:
                logger.debug("RESOLVED_BY synapse already exists, skipping")

        resolutions.append(
            ConflictResolution(
                conflict=conflict,
                contradicts_synapse=contradicts_synapse,
                existing_neuron_updated=updated_neuron,
                confidence_reduced_by=confidence_reduced,
                superseded=is_superseded,
            )
        )

    return resolutions


def _is_error_memory(neuron: Neuron, memory_type_hint: str) -> bool:
    """Check if a neuron represents an error memory.

    Checks both the explicit memory_type_hint and the neuron's own metadata.

    Args:
        neuron: The existing neuron to check
        memory_type_hint: Explicit type hint from caller (e.g. "error")

    Returns:
        True if the memory is an error type
    """
    if memory_type_hint == "error":
        return True
    neuron_type = str(neuron.metadata.get("type", ""))
    return neuron_type == "error"


def _extract_search_terms(content: str) -> list[str]:
    """Extract key terms from content for searching existing memories."""
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_.-]+\b", content)
    terms = [w for w in words if w.lower() not in _STOP_WORDS and len(w) >= 3]

    # Deduplicate (case-insensitive) while preserving original case and order
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        lower = term.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(term)
    return unique


def _subjects_match(subject_a: str, subject_b: str) -> bool:
    """Check if two subjects refer to the same entity."""
    if subject_a == subject_b:
        return True
    # Both implicit agents match
    if subject_a == "_implicit_agent" and subject_b == "_implicit_agent":
        return True
    # One implicit + one explicit: match if explicit is generic
    if "_implicit_agent" in (subject_a, subject_b):
        return True
    # Fuzzy: check if one contains the other
    return subject_a in subject_b or subject_b in subject_a


def _predicates_conflict(pred_a: str, pred_b: str) -> bool:
    """Check if two predicates are contradictory (different answers to same question)."""
    # Same predicate = agreement, not conflict
    if pred_a == pred_b:
        return False

    # Check if both are non-empty and substantively different
    if not pred_a or not pred_b:
        return False

    # If predicates share many words, they likely agree (not contradict)
    words_a = set(pred_a.lower().split())
    words_b = set(pred_b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    if overlap > 0.7:
        return False

    return True


def _is_decision_content(content: str) -> bool:
    """Check if content represents a decision."""
    decision_markers = [
        "decided",
        "chose",
        "selected",
        "picked",
        "going with",
        "will use",
        "switched to",
        "migrated to",
        "adopted",
    ]
    content_lower = content.lower()
    return any(marker in content_lower for marker in decision_markers)


def _content_agrees(content_a: str, content_b: str) -> bool:
    """Check if two pieces of content express the same conclusion."""
    # Simple heuristic: high word overlap = agreement
    words_a = set(content_a.lower().split())
    words_b = set(content_b.lower().split())

    if not words_a or not words_b:
        return False

    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap > 0.8


def _extract_implicit_tags(neuron: Neuron) -> set[str]:
    """Extract implicit tags from neuron metadata (for fibers we don't have direct access to)."""
    tags: set[str] = set()

    # Check for tags in metadata
    if "tags" in neuron.metadata:
        meta_tags = neuron.metadata["tags"]
        if isinstance(meta_tags, (list, set, frozenset)):
            tags.update(str(t) for t in meta_tags)

    # Extract key terms as implicit tags (exclude common stop words)
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_.-]+\b", neuron.content)
    tags.update(w.lower() for w in words if w.lower() not in _STOP_WORDS and len(w) >= 3)

    return tags
