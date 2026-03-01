"""Tool memory processing engine.

Reads raw tool events from the staging table (or JSONL buffer),
detects usage patterns, and promotes them to neurons and synapses:
- USED_WITH: Tools frequently used together within a time window.
- EFFECTIVE_FOR: Tools that succeed in the context of a specific task.

Processing is designed to be idempotent and batched.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType

if TYPE_CHECKING:
    # Storage is typed as Any because tool_memory functions accept
    # SQLiteStorage (which mixes in SQLiteToolEventsMixin) — the base
    # NeuralStorage protocol doesn't declare tool-event methods.
    from typing import Protocol

    from neural_memory.unified_config import ToolMemoryConfig

    class _ToolEventStorage(Protocol):
        async def insert_tool_events(self, brain_id: str, events: list[dict[str, Any]]) -> int: ...
        async def get_unprocessed_events(
            self, brain_id: str, limit: int = ...
        ) -> list[dict[str, Any]]: ...
        async def mark_events_processed(self, brain_id: str, event_ids: list[int]) -> None: ...
        async def find_neurons(self, *, content_exact: str, limit: int = ...) -> list[Neuron]: ...
        async def add_neuron(self, neuron: Any) -> None: ...
        async def get_synapses(
            self,
            *,
            source_id: str | None = ...,
            target_id: str | None = ...,
            type: Any | None = ...,
        ) -> list[Any]: ...
        async def add_synapse(self, synapse: Any) -> None: ...
        async def update_synapse(self, synapse: Any) -> None: ...
        def set_brain(self, brain_id: str) -> None: ...


logger = logging.getLogger(__name__)

# Max characters for args_summary in JSONL buffer
_MAX_ARGS_SUMMARY = 200


@dataclass(frozen=True)
class IngestResult:
    """Result of ingesting JSONL buffer into tool_events table."""

    events_ingested: int
    events_skipped: int


@dataclass(frozen=True)
class ProcessResult:
    """Result of processing tool events into neurons/synapses."""

    neurons_created: int
    synapses_created: int
    synapses_reinforced: int
    events_processed: int


def _parse_buffer_line(line: str) -> dict[str, Any] | None:
    """Parse a single JSONL line into an event dict.

    Returns None if the line is malformed.
    """
    try:
        data: dict[str, Any] = json.loads(line)
        if not isinstance(data, dict) or "tool_name" not in data:
            return None
        return data
    except (json.JSONDecodeError, TypeError):
        return None


async def ingest_buffer(
    storage: _ToolEventStorage,
    brain_id: str,
    buffer_path: Path,
    max_lines: int = 10000,
) -> IngestResult:
    """Read JSONL buffer file, insert events into tool_events table, truncate buffer.

    Args:
        storage: Storage backend (must have insert_tool_events method).
        brain_id: Brain context.
        buffer_path: Path to the tool_events.jsonl file.
        max_lines: Max lines to read per ingestion cycle.

    Returns:
        IngestResult with counts.
    """
    if not buffer_path.exists():
        return IngestResult(events_ingested=0, events_skipped=0)

    try:
        raw = buffer_path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("Failed to read tool events buffer", exc_info=True)
        return IngestResult(events_ingested=0, events_skipped=0)

    lines = raw.strip().splitlines()
    if not lines:
        return IngestResult(events_ingested=0, events_skipped=0)

    # Cap to max_lines (oldest first)
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    events: list[dict[str, Any]] = []
    skipped = 0
    for line in lines:
        parsed = _parse_buffer_line(line)
        if parsed is None:
            skipped += 1
            continue
        events.append(parsed)

    inserted = 0
    if events:
        inserted = await storage.insert_tool_events(brain_id, events)

    # Truncate buffer after successful ingestion
    try:
        buffer_path.write_text("", encoding="utf-8")
    except OSError:
        logger.debug("Failed to truncate tool events buffer", exc_info=True)

    return IngestResult(events_ingested=inserted, events_skipped=skipped)


def _tool_neuron_content(tool_name: str) -> str:
    """Canonical content string for a tool neuron."""
    return f"tool:{tool_name}"


async def _find_or_create_tool_neuron(
    storage: _ToolEventStorage,
    tool_name: str,
    server_name: str,
) -> str:
    """Find existing tool neuron or create a new one.

    Storage must have brain context set via set_brain() before calling.
    Returns the neuron ID.
    """
    content = _tool_neuron_content(tool_name)

    # Search for existing neuron with this content
    existing = await storage.find_neurons(content_exact=content, limit=1)
    if existing:
        return existing[0].id

    # Create new tool neuron
    neuron = Neuron.create(
        content=content,
        type=NeuronType.ENTITY,
        metadata={"tool_server": server_name, "tool_type": "mcp"},
    )
    await storage.add_neuron(neuron)
    return neuron.id


async def _find_or_create_concept_neuron(
    storage: _ToolEventStorage,
    concept: str,
) -> str:
    """Find existing concept neuron or create a new one for task context."""
    existing = await storage.find_neurons(content_exact=concept, limit=1)
    if existing:
        return existing[0].id

    neuron = Neuron.create(
        content=concept,
        type=NeuronType.CONCEPT,
    )
    await storage.add_neuron(neuron)
    return neuron.id


async def _find_synapse_between(
    storage: _ToolEventStorage,
    source_id: str,
    target_id: str,
    synapse_type: SynapseType,
) -> Synapse | None:
    """Find an existing synapse between two neurons."""
    synapses = await storage.get_synapses(
        source_id=source_id,
        target_id=target_id,
        type=synapse_type,
    )
    return synapses[0] if synapses else None


async def process_events(
    storage: _ToolEventStorage,
    brain_id: str,
    config: ToolMemoryConfig,
) -> ProcessResult:
    """Process unprocessed tool events into neurons and synapses.

    Pattern detection:
    1. USED_WITH: Tools used within cooccurrence_window_s in same session.
    2. EFFECTIVE_FOR: Successful tool used with a task_context.

    Requires storage.set_brain(brain_id) to have been called before.

    Args:
        storage: Storage backend.
        brain_id: Brain context (used for tool_events table queries).
        config: Tool memory configuration.

    Returns:
        ProcessResult with counts.
    """
    events = await storage.get_unprocessed_events(brain_id, config.process_batch_size)
    if not events:
        return ProcessResult(
            neurons_created=0,
            synapses_created=0,
            synapses_reinforced=0,
            events_processed=0,
        )

    neurons_created = 0
    synapses_created = 0
    synapses_reinforced = 0

    # Count tool frequency across all events
    tool_freq: dict[str, int] = defaultdict(int)
    tool_server: dict[str, str] = {}
    for ev in events:
        tool_freq[ev["tool_name"]] += 1
        tool_server[ev["tool_name"]] = ev.get("server_name", "")

    # Only process tools that meet frequency threshold
    frequent_tools = {t for t, c in tool_freq.items() if c >= config.min_frequency}

    # Group events by session for co-occurrence detection
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        sid = ev.get("session_id", "")
        if sid:
            sessions[sid].append(ev)

    # USED_WITH detection: sliding window within each session
    seen_pairs: set[tuple[str, str]] = set()
    for session_events in sessions.values():
        session_events.sort(key=lambda e: e["created_at"])
        for i, ev_a in enumerate(session_events):
            if ev_a["tool_name"] not in frequent_tools:
                continue
            for j in range(i + 1, len(session_events)):
                ev_b = session_events[j]
                if ev_b["tool_name"] not in frequent_tools:
                    continue
                if ev_a["tool_name"] == ev_b["tool_name"]:
                    continue

                # Check time window
                try:
                    ts_a = datetime.fromisoformat(ev_a["created_at"])
                    ts_b = datetime.fromisoformat(ev_b["created_at"])
                    delta = abs((ts_b - ts_a).total_seconds())
                except (ValueError, TypeError):
                    continue

                if delta > config.cooccurrence_window_s:
                    break  # Sorted by time — no more within window

                # Canonical pair (alphabetical order for dedup)
                pair = tuple(sorted([ev_a["tool_name"], ev_b["tool_name"]]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Find or create neurons for both tools
                nid_a = await _find_or_create_tool_neuron(
                    storage, pair[0], tool_server.get(pair[0], "")
                )
                nid_b = await _find_or_create_tool_neuron(
                    storage, pair[1], tool_server.get(pair[1], "")
                )

                # Find or create USED_WITH synapse (check both directions)
                existing_syn = await _find_synapse_between(
                    storage, nid_a, nid_b, SynapseType.USED_WITH
                )
                if existing_syn is None:
                    existing_syn = await _find_synapse_between(
                        storage, nid_b, nid_a, SynapseType.USED_WITH
                    )

                if existing_syn:
                    reinforced = existing_syn.reinforce()
                    await storage.update_synapse(reinforced)
                    synapses_reinforced += 1
                else:
                    synapse = Synapse.create(
                        source_id=nid_a,
                        target_id=nid_b,
                        type=SynapseType.USED_WITH,
                        weight=0.3,
                        direction=Direction.BIDIRECTIONAL,
                    )
                    await storage.add_synapse(synapse)
                    synapses_created += 1

    # EFFECTIVE_FOR detection: tools with task_context
    task_tool_pairs: set[tuple[str, str]] = set()  # (tool_name, task_context)
    for ev in events:
        if ev["tool_name"] not in frequent_tools:
            continue
        task = ev.get("task_context", "").strip()
        if not task:
            continue
        if not ev.get("success", True):
            continue

        pair_key = (ev["tool_name"], task)
        if pair_key in task_tool_pairs:
            continue
        task_tool_pairs.add(pair_key)

        tool_nid = await _find_or_create_tool_neuron(
            storage, ev["tool_name"], ev.get("server_name", "")
        )
        task_nid = await _find_or_create_concept_neuron(storage, task)

        existing_syn = await _find_synapse_between(
            storage, tool_nid, task_nid, SynapseType.EFFECTIVE_FOR
        )
        if existing_syn:
            reinforced = existing_syn.reinforce()
            await storage.update_synapse(reinforced)
            synapses_reinforced += 1
        else:
            synapse = Synapse.create(
                source_id=tool_nid,
                target_id=task_nid,
                type=SynapseType.EFFECTIVE_FOR,
                weight=0.4,
                direction=Direction.UNIDIRECTIONAL,
            )
            await storage.add_synapse(synapse)
            synapses_created += 1

    # Mark all events as processed
    event_ids = [ev["id"] for ev in events]
    await storage.mark_events_processed(brain_id, event_ids)

    return ProcessResult(
        neurons_created=neurons_created,
        synapses_created=synapses_created,
        synapses_reinforced=synapses_reinforced,
        events_processed=len(events),
    )
