"""Query pattern mining — learn recall topic associations.

Mines action_context from recall events to discover topic co-occurrence
patterns and materialize them as CONCEPT neurons + BEFORE synapses.

Same substrate as action habits (no new types):
- Neurons: NeuronType.CONCEPT (vs ACTION for habits)
- Synapses: SynapseType.BEFORE
- Fibers: WORKFLOW with metadata {"_query_pattern": True}

Zero LLM dependency — pure frequency-based topic mining.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.extraction.keywords import extract_keywords
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from datetime import datetime

    from neural_memory.core.action_event import ActionEvent
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicPair:
    """A pair of query topics observed in sequence.

    Attributes:
        topic_a: First topic keyword
        topic_b: Second topic keyword
        count: Number of times this pair was observed
        avg_gap_seconds: Average time gap between queries containing A and B
    """

    topic_a: str
    topic_b: str
    count: int
    avg_gap_seconds: float


@dataclass(frozen=True)
class QueryPatternCandidate:
    """A candidate query pattern for materialization.

    Attributes:
        topics: Ordered topic pair (a, b)
        frequency: Number of co-occurrences
        confidence: Frequency / total_sessions
    """

    topics: tuple[str, str]
    frequency: int
    confidence: float


@dataclass
class QueryPatternReport:
    """Report from query pattern learning.

    Attributes:
        topics_extracted: Number of unique topics found
        pairs_found: Number of topic pairs above threshold
        patterns_learned: Number of new CONCEPT neurons + BEFORE synapses created
    """

    topics_extracted: int = 0
    pairs_found: int = 0
    patterns_learned: int = 0


def extract_topics(action_context: str) -> list[str]:
    """Extract topic keywords from a recall query's action_context.

    Uses the keyword extractor with min_length=3 for meaningful topics.

    Args:
        action_context: The query string from an action event

    Returns:
        List of normalized topic keywords (lowercase)
    """
    if not action_context or len(action_context) < 3:
        return []

    keywords = extract_keywords(action_context, min_length=3)
    # Normalize and deduplicate
    seen: set[str] = set()
    result: list[str] = []
    for kw in keywords:
        normalized = kw.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result[:10]  # Cap at 10 topics per query


def mine_query_topic_pairs(
    events: list[ActionEvent],
    window_seconds: float = 600.0,
) -> list[TopicPair]:
    """Mine topic pairs from recall events within sessions.

    Groups recall events by session, extracts topics from action_context,
    and counts topic pairs across consecutive queries within the time window.

    Args:
        events: Recall action events (filtered to action_type="recall")
        window_seconds: Maximum gap between queries for pair counting

    Returns:
        List of TopicPair sorted by count descending
    """
    # Group by session
    sessions: dict[str | None, list[ActionEvent]] = defaultdict(list)
    for event in events:
        if event.action_type == "recall" and event.action_context:
            sessions[event.session_id].append(event)

    # Mine topic pairs across consecutive recall queries
    pair_gaps: dict[tuple[str, str], list[float]] = defaultdict(list)

    for session_events in sessions.values():
        sorted_events = sorted(session_events, key=lambda e: e.created_at)

        for i in range(len(sorted_events) - 1):
            a = sorted_events[i]
            b = sorted_events[i + 1]
            gap = (b.created_at - a.created_at).total_seconds()
            if gap > window_seconds:
                continue

            topics_a = extract_topics(a.action_context)
            topics_b = extract_topics(b.action_context)

            # Create directed pairs: topic from query A -> topic from query B
            for ta in topics_a:
                for tb in topics_b:
                    if ta != tb:
                        pair_gaps[(ta, tb)].append(gap)

    results: list[TopicPair] = []
    for (topic_a, topic_b), gaps in pair_gaps.items():
        results.append(
            TopicPair(
                topic_a=topic_a,
                topic_b=topic_b,
                count=len(gaps),
                avg_gap_seconds=sum(gaps) / len(gaps) if gaps else 0.0,
            )
        )

    results.sort(key=lambda p: p.count, reverse=True)
    return results


def extract_pattern_candidates(
    pairs: list[TopicPair],
    min_frequency: int = 3,
    total_sessions: int = 1,
) -> list[QueryPatternCandidate]:
    """Filter topic pairs into pattern candidates.

    Args:
        pairs: Topic pairs from mine_query_topic_pairs
        min_frequency: Minimum count for a pair to qualify
        total_sessions: Total sessions for confidence calculation

    Returns:
        List of QueryPatternCandidate sorted by frequency descending
    """
    candidates: list[QueryPatternCandidate] = []
    for pair in pairs:
        if pair.count >= min_frequency:
            confidence = min(pair.count / max(total_sessions, 1), 1.0)
            candidates.append(
                QueryPatternCandidate(
                    topics=(pair.topic_a, pair.topic_b),
                    frequency=pair.count,
                    confidence=confidence,
                )
            )

    candidates.sort(key=lambda c: c.frequency, reverse=True)
    return candidates[:20]  # Cap at 20 patterns


async def learn_query_patterns(
    storage: NeuralStorage,
    config: BrainConfig,
    reference_time: datetime,
) -> QueryPatternReport:
    """Mine query patterns and materialize as CONCEPT neurons + BEFORE synapses.

    Called during LEARN_HABITS consolidation strategy alongside action habits.

    Args:
        storage: Storage backend
        config: Brain configuration
        reference_time: Current time for freshness

    Returns:
        Report with mining statistics
    """
    report = QueryPatternReport()

    # Fetch recall events
    events = await storage.get_action_sequences(limit=1000)
    recall_events = [e for e in events if e.action_type == "recall" and e.action_context]

    if len(recall_events) < 3:
        return report

    # Extract unique topics
    all_topics: set[str] = set()
    for event in recall_events:
        topics = extract_topics(event.action_context)
        all_topics.update(topics)
    report.topics_extracted = len(all_topics)

    # Mine pairs
    pairs = mine_query_topic_pairs(recall_events)
    if not pairs:
        return report

    # Count sessions for confidence
    session_ids = {e.session_id for e in recall_events if e.session_id}
    total_sessions = max(len(session_ids), 1)

    candidates = extract_pattern_candidates(
        pairs,
        min_frequency=getattr(config, "query_pattern_min_frequency", 3),
        total_sessions=total_sessions,
    )
    report.pairs_found = len(candidates)

    if not candidates:
        return report

    # Materialize patterns
    now = utcnow()
    for candidate in candidates:
        topic_a, topic_b = candidate.topics

        # Find or create CONCEPT neurons for each topic
        neuron_a = await _get_or_create_concept_neuron(storage, topic_a, now)
        neuron_b = await _get_or_create_concept_neuron(storage, topic_b, now)

        if neuron_a.id == neuron_b.id:
            continue

        # Create or strengthen BEFORE synapse
        await _strengthen_topic_synapse(storage, neuron_a.id, neuron_b.id, candidate.frequency)

        report.patterns_learned += 1

    if report.patterns_learned:
        logger.info(
            "Learned %d query patterns from %d topics",
            report.patterns_learned,
            report.topics_extracted,
        )

    return report


async def suggest_follow_up_queries(
    storage: NeuralStorage,
    topics: list[str],
    config: BrainConfig,
) -> list[str]:
    """Suggest follow-up query topics based on learned patterns.

    Finds CONCEPT neurons for current topics, follows BEFORE synapses
    to discover related topics via spreading activation.

    Args:
        storage: Storage backend
        topics: Current query topics (from extract_topics)
        config: Brain configuration

    Returns:
        List of suggested follow-up topic strings
    """
    if not topics:
        return []

    suggestions: list[tuple[str, float]] = []
    seen_topics = set(topics)

    for topic in topics[:5]:  # Limit input topics
        neurons = await storage.find_neurons(
            content_exact=topic,
            type=NeuronType.CONCEPT,
            limit=1,
        )
        if not neurons:
            continue

        neuron = neurons[0]

        # Follow BEFORE synapses (outgoing)
        try:
            neighbors = await storage.get_neighbors(
                neuron.id,
                direction="out",
                synapse_types=[SynapseType.BEFORE],
                min_weight=0.1,
            )

            for neighbor_neuron, synapse in neighbors:
                if neighbor_neuron.content in seen_topics:
                    continue
                # Only include CONCEPT neurons from query patterns
                if neighbor_neuron.type != NeuronType.CONCEPT:
                    continue
                suggestions.append((neighbor_neuron.content, synapse.weight))
                seen_topics.add(neighbor_neuron.content)
        except Exception:
            logger.debug("Query pattern neighbor lookup failed", exc_info=True)

    # Sort by weight, return top 5
    suggestions.sort(key=lambda s: s[1], reverse=True)
    return [s[0] for s in suggestions[:5]]


async def _get_or_create_concept_neuron(
    storage: NeuralStorage,
    topic: str,
    now: datetime,
) -> Neuron:
    """Find existing CONCEPT neuron or create a new one."""
    existing = await storage.find_neurons(
        content_exact=topic,
        type=NeuronType.CONCEPT,
        limit=1,
    )
    if existing:
        return existing[0]

    neuron = Neuron(
        id=f"concept-{topic}",
        type=NeuronType.CONCEPT,
        content=topic,
        metadata={"_query_pattern": True},
        created_at=now,
    )
    await storage.add_neuron(neuron)
    return neuron


async def _strengthen_topic_synapse(
    storage: NeuralStorage,
    source_id: str,
    target_id: str,
    frequency: int,
) -> None:
    """Create or strengthen a BEFORE synapse between topic neurons."""
    # Check for existing synapse
    try:
        neighbors = await storage.get_neighbors(
            source_id,
            direction="out",
            synapse_types=[SynapseType.BEFORE],
        )
        for neighbor_neuron, existing_synapse in neighbors:
            if neighbor_neuron.id == target_id:
                # Strengthen existing synapse
                from dataclasses import replace

                new_weight = min(existing_synapse.weight + 0.05, 1.0)
                new_count = existing_synapse.metadata.get("sequential_count", 0) + frequency
                updated = replace(
                    existing_synapse,
                    weight=new_weight,
                    metadata={**existing_synapse.metadata, "sequential_count": new_count},
                )
                await storage.update_synapse(updated)
                return
    except Exception:
        logger.debug("Synapse lookup failed, creating new", exc_info=True)

    # Create new synapse
    synapse = Synapse(
        id=uuid4().hex[:16],
        source_id=source_id,
        target_id=target_id,
        type=SynapseType.BEFORE,
        weight=min(0.3 + frequency * 0.05, 1.0),
        metadata={"_query_pattern": True, "sequential_count": frequency},
        created_at=utcnow(),
    )
    try:
        await storage.add_synapse(synapse)
    except ValueError:
        logger.debug("Query pattern synapse already exists")
