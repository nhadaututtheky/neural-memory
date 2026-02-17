"""Sequence mining — detect habitual action patterns and create workflow fibers.

Mines action event sequences to discover repeated patterns:
1. Group events by session, find consecutive pairs within time window
2. Extract bigram/trigram candidates meeting frequency threshold
3. Create ACTION neurons + BEFORE synapses + WORKFLOW fibers

Zero LLM dependency — pure frequency-based pattern detection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.action_event import ActionEvent
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class SequencePair:
    """A consecutive pair of actions observed in sessions.

    Attributes:
        action_a: First action type
        action_b: Second action type
        count: Number of times this pair was observed
        avg_gap_seconds: Average time gap between A and B
    """

    action_a: str
    action_b: str
    count: int
    avg_gap_seconds: float


@dataclass(frozen=True)
class HabitCandidate:
    """A candidate habit pattern extracted from sequential pairs.

    Attributes:
        steps: Ordered tuple of action types forming the habit
        frequency: Number of times this pattern was observed
        avg_duration_seconds: Average total duration of the pattern
        confidence: Frequency / total sessions (how consistently it appears)
    """

    steps: tuple[str, ...]
    frequency: int
    avg_duration_seconds: float
    confidence: float


@dataclass(frozen=True)
class LearnedHabit:
    """A fully materialized habit in the neural graph.

    Attributes:
        name: Heuristic name (e.g., "recall-edit-test")
        steps: Ordered action types
        frequency: How often this pattern occurs
        workflow_fiber: The WORKFLOW fiber created for this habit
        sequence_synapses: BEFORE synapses connecting the action neurons
    """

    name: str
    steps: tuple[str, ...]
    frequency: int
    workflow_fiber: Fiber
    sequence_synapses: list[Synapse]


@dataclass
class HabitReport:
    """Report from habit learning operations.

    Attributes:
        sequences_analyzed: Total action events processed
        pairs_strengthened: Sequential pairs that had existing synapses reinforced
        habits_learned: New habits materialized in the graph
        action_events_pruned: Old action events cleaned up
    """

    sequences_analyzed: int = 0
    pairs_strengthened: int = 0
    habits_learned: int = 0
    action_events_pruned: int = 0


def mine_sequential_pairs(
    events: list[ActionEvent],
    window_seconds: float,
) -> list[SequencePair]:
    """Mine consecutive action pairs from event sequences.

    Groups events by session_id, sorts by created_at, and counts
    pairs of consecutive actions within the time window.

    Args:
        events: List of action events (any order)
        window_seconds: Maximum gap between A and B to count as sequential

    Returns:
        List of SequencePair sorted by count descending
    """
    # Group by session
    sessions: dict[str | None, list[ActionEvent]] = defaultdict(list)
    for event in events:
        sessions[event.session_id].append(event)

    # Count pairs
    pair_gaps: dict[tuple[str, str], list[float]] = defaultdict(list)

    for session_events in sessions.values():
        sorted_events = sorted(session_events, key=lambda e: e.created_at)
        for i in range(len(sorted_events) - 1):
            a = sorted_events[i]
            b = sorted_events[i + 1]
            gap = (b.created_at - a.created_at).total_seconds()
            if gap <= window_seconds:
                pair_gaps[(a.action_type, b.action_type)].append(gap)

    results: list[SequencePair] = []
    for (action_a, action_b), gaps in pair_gaps.items():
        results.append(
            SequencePair(
                action_a=action_a,
                action_b=action_b,
                count=len(gaps),
                avg_gap_seconds=sum(gaps) / len(gaps) if gaps else 0.0,
            )
        )

    results.sort(key=lambda p: p.count, reverse=True)
    return results


def extract_habit_candidates(
    pairs: list[SequencePair],
    min_frequency: int,
    total_sessions: int = 1,
) -> list[HabitCandidate]:
    """Extract habit candidates from sequential pairs.

    Builds bigrams and trigrams from pairs meeting the frequency threshold.

    Args:
        pairs: Sequential pairs from mine_sequential_pairs
        min_frequency: Minimum count for a pair to be considered
        total_sessions: Total number of sessions for confidence calculation

    Returns:
        List of HabitCandidate sorted by frequency descending
    """
    # Filter pairs by min_frequency
    frequent_pairs = [p for p in pairs if p.count >= min_frequency]
    if not frequent_pairs:
        return []

    candidates: list[HabitCandidate] = []
    seen_steps: set[tuple[str, ...]] = set()

    # Build bigrams
    for pair in frequent_pairs:
        steps = (pair.action_a, pair.action_b)
        if steps not in seen_steps:
            seen_steps.add(steps)
            candidates.append(
                HabitCandidate(
                    steps=steps,
                    frequency=pair.count,
                    avg_duration_seconds=pair.avg_gap_seconds,
                    confidence=pair.count / max(total_sessions, 1),
                )
            )

    # Build trigrams: A→B + B→C = A→B→C
    pair_map: dict[str, list[SequencePair]] = defaultdict(list)
    for pair in frequent_pairs:
        pair_map[pair.action_a].append(pair)

    for pair_ab in frequent_pairs:
        for pair_bc in pair_map.get(pair_ab.action_b, []):
            if pair_bc.action_b == pair_ab.action_a:
                continue  # Skip cycles
            tri_steps = (pair_ab.action_a, pair_ab.action_b, pair_bc.action_b)
            if tri_steps in seen_steps:
                continue
            seen_steps.add(tri_steps)
            freq = min(pair_ab.count, pair_bc.count)
            if freq >= min_frequency:
                candidates.append(
                    HabitCandidate(
                        steps=tri_steps,
                        frequency=freq,
                        avg_duration_seconds=pair_ab.avg_gap_seconds + pair_bc.avg_gap_seconds,
                        confidence=freq / max(total_sessions, 1),
                    )
                )

    candidates.sort(key=lambda c: c.frequency, reverse=True)
    return candidates


def heuristic_habit_name(steps: tuple[str, ...]) -> str:
    """Generate a human-readable name from action steps.

    Args:
        steps: Ordered action types

    Returns:
        Hyphen-joined name (e.g., "recall-edit-test")
    """
    return "-".join(steps)


async def strengthen_sequential_pair(
    storage: NeuralStorage,
    action_a: str,
    action_b: str,
    config: BrainConfig,
) -> Synapse | None:
    """Find or create a BEFORE synapse between two action neurons.

    Looks up ACTION neurons by exact content match. If both exist
    and a BEFORE synapse connects them, reinforces it. Otherwise
    creates the synapse.

    Args:
        storage: Storage backend
        action_a: First action type
        action_b: Second action type
        config: Brain configuration

    Returns:
        The created or reinforced synapse, or None if neurons don't exist
    """
    neurons_a = await storage.find_neurons(content_exact=action_a, type=NeuronType.ACTION)
    neurons_b = await storage.find_neurons(content_exact=action_b, type=NeuronType.ACTION)

    if not neurons_a or not neurons_b:
        return None

    neuron_a = neurons_a[0]
    neuron_b = neurons_b[0]

    # Check for existing BEFORE synapse
    existing = await storage.get_synapses(
        source_id=neuron_a.id,
        target_id=neuron_b.id,
        type=SynapseType.BEFORE,
    )

    if existing:
        synapse = existing[0]
        seq_count = synapse.metadata.get("sequential_count", 0) + 1
        reinforced = Synapse(
            id=synapse.id,
            source_id=synapse.source_id,
            target_id=synapse.target_id,
            type=synapse.type,
            weight=min(1.0, synapse.weight + config.reinforcement_delta),
            direction=synapse.direction,
            metadata={**synapse.metadata, "sequential_count": seq_count},
            reinforced_count=synapse.reinforced_count + 1,
            last_activated=utcnow(),
            created_at=synapse.created_at,
        )
        await storage.update_synapse(reinforced)
        return reinforced

    # Create new BEFORE synapse
    synapse = Synapse.create(
        source_id=neuron_a.id,
        target_id=neuron_b.id,
        type=SynapseType.BEFORE,
        weight=config.default_synapse_weight,
        metadata={"sequential_count": 1, "_habit": True},
    )
    await storage.add_synapse(synapse)
    return synapse


async def learn_habits(
    storage: NeuralStorage,
    config: BrainConfig,
    reference_time: datetime,
) -> tuple[list[LearnedHabit], HabitReport]:
    """Learn habits from action event sequences.

    Full pipeline:
    1. Get action sequences from last 30 days
    2. Mine sequential pairs within time window
    3. Extract habit candidates meeting frequency threshold
    4. For qualifying candidates: create neurons, synapses, fibers
    5. Prune old action events (>60 days)

    Args:
        storage: Storage backend
        config: Brain configuration
        reference_time: Current time for age calculations

    Returns:
        Tuple of (learned habits, report)
    """
    report = HabitReport()

    # 1. Get recent action sequences
    since = reference_time - timedelta(days=30)
    events = await storage.get_action_sequences(since=since)
    report.sequences_analyzed = len(events)

    if len(events) < 2:
        return [], report

    # 2. Mine sequential pairs
    pairs = mine_sequential_pairs(events, config.sequential_window_seconds)
    if not pairs:
        return [], report

    # Count unique sessions for confidence calculation
    session_ids = {e.session_id for e in events if e.session_id}
    total_sessions = max(len(session_ids), 1)

    # 3. Extract candidates
    candidates = extract_habit_candidates(pairs, config.habit_min_frequency, total_sessions)
    if not candidates:
        return [], report

    # 4. Materialize qualifying candidates
    learned: list[LearnedHabit] = []

    # Batch-fetch all unique step names across all candidates
    all_steps = {step for candidate in candidates for step in candidate.steps}
    existing_actions = await storage.find_neurons_exact_batch(
        list(all_steps), type=NeuronType.ACTION
    )

    for candidate in candidates:
        # Ensure ACTION neurons exist for each step
        neuron_ids: list[str] = []
        for step in candidate.steps:
            if step in existing_actions:
                neuron_ids.append(existing_actions[step].id)
            else:
                neuron = Neuron.create(
                    type=NeuronType.ACTION,
                    content=step,
                    metadata={"_habit_action": True},
                )
                await storage.add_neuron(neuron)
                existing_actions[step] = neuron  # Cache for next candidate
                neuron_ids.append(neuron.id)

        # Create BEFORE synapses between consecutive steps
        sequence_synapses: list[Synapse] = []
        for i in range(len(neuron_ids) - 1):
            synapse = await strengthen_sequential_pair(
                storage,
                candidate.steps[i],
                candidate.steps[i + 1],
                config,
            )
            if synapse:
                sequence_synapses.append(synapse)
                report.pairs_strengthened += 1

        # Create WORKFLOW fiber
        name = heuristic_habit_name(candidate.steps)
        synapse_id_set = {s.id for s in sequence_synapses}
        neuron_id_set = set(neuron_ids)

        workflow_fiber = Fiber.create(
            neuron_ids=neuron_id_set,
            synapse_ids=synapse_id_set,
            anchor_neuron_id=neuron_ids[0],
            pathway=neuron_ids,
            summary=name,
            tags=set(),
            metadata={
                "_workflow_actions": list(candidate.steps),
                "_habit_pattern": True,
                "_habit_frequency": candidate.frequency,
                "_habit_confidence": candidate.confidence,
            },
        )
        await storage.add_fiber(workflow_fiber)

        learned.append(
            LearnedHabit(
                name=name,
                steps=candidate.steps,
                frequency=candidate.frequency,
                workflow_fiber=workflow_fiber,
                sequence_synapses=sequence_synapses,
            )
        )
        report.habits_learned += 1

    # 5. Prune old action events (>60 days)
    prune_cutoff = reference_time - timedelta(days=60)
    pruned = await storage.prune_action_events(prune_cutoff)
    report.action_events_pruned = pruned

    return learned, report
