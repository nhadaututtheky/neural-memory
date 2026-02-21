"""Memory narrative generator — template-based markdown stories.

Produces structured markdown narratives from the neural graph.
No LLM required — uses fiber metadata, timestamps, and causal chains.

Three modes:
    - timeline: Date-range narrative ordered by time
    - topic:    SA-driven narrative around a topic
    - causal:   Traces CAUSED_BY chains into a causal story
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.core.fiber import Fiber
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Hard cap for fibers in any narrative to prevent runaway
MAX_NARRATIVE_FIBERS = 50


@dataclass(frozen=True)
class NarrativeItem:
    """A single item in a narrative."""

    fiber_id: str
    timestamp: str
    summary: str
    tags: list[str] = field(default_factory=list)
    relevance: float = 0.0


@dataclass(frozen=True)
class Narrative:
    """A generated narrative."""

    mode: str
    title: str
    items: list[NarrativeItem]
    markdown: str


def _fiber_to_item(fiber: Fiber, relevance: float = 0.0) -> NarrativeItem:
    """Convert a fiber to a narrative item."""
    summary = fiber.summary or ""
    if not summary:
        # Fallback to metadata or fiber ID
        summary = fiber.metadata.get("content_preview", f"Memory {fiber.id[:8]}")
    ts = fiber.created_at.isoformat() if fiber.created_at else ""
    return NarrativeItem(
        fiber_id=fiber.id,
        timestamp=ts,
        summary=summary,
        tags=sorted(fiber.tags),
        relevance=relevance,
    )


def _render_timeline_md(items: list[NarrativeItem], start: str, end: str) -> str:
    """Render timeline items as markdown."""
    lines = [f"## Timeline: {start} to {end}", ""]
    for item in items:
        date_str = item.timestamp[:10] if item.timestamp else "Unknown"
        tags_str = f" `{'`, `'.join(item.tags)}`" if item.tags else ""
        lines.append(f"- **{date_str}**: {item.summary}{tags_str}")
    if not items:
        lines.append("_No memories found in this time range._")
    return "\n".join(lines)


def _render_topic_md(items: list[NarrativeItem], topic: str) -> str:
    """Render topic items as markdown."""
    lines = [f"## Topic: {topic}", ""]
    for item in items:
        score_str = f" ({item.relevance:.0%})" if item.relevance > 0 else ""
        lines.append(f"- {item.summary}{score_str}")
    if not items:
        lines.append(f"_No memories found related to '{topic}'._")
    return "\n".join(lines)


def _render_causal_md(items: list[NarrativeItem], topic: str) -> str:
    """Render causal chain as markdown."""
    lines = [f"## Causal Chain: {topic}", ""]
    if items:
        for i, item in enumerate(items):
            prefix = "  " * i + ("-> " if i > 0 else "")
            lines.append(f"{prefix}**{item.summary}**")
    else:
        lines.append(f"_No causal chain found for '{topic}'._")
    return "\n".join(lines)


async def generate_timeline_narrative(
    storage: NeuralStorage,
    start_date: datetime,
    end_date: datetime,
    max_fibers: int = 20,
) -> Narrative:
    """Generate a time-ordered narrative for a date range.

    Args:
        storage: Neural storage backend
        start_date: Start of the time range
        end_date: End of the time range
        max_fibers: Maximum fibers to include

    Returns:
        A Narrative with timeline mode
    """
    safe_limit = min(max_fibers, MAX_NARRATIVE_FIBERS)
    fibers = await storage.find_fibers(
        time_overlaps=(start_date, end_date),
        limit=safe_limit,
    )

    # Sort by creation time
    fibers.sort(key=lambda f: f.created_at)

    items = [_fiber_to_item(f) for f in fibers]
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    md = _render_timeline_md(items, start_str, end_str)

    return Narrative(
        mode="timeline",
        title=f"Timeline: {start_str} to {end_str}",
        items=items,
        markdown=md,
    )


async def generate_topic_narrative(
    storage: NeuralStorage,
    config: BrainConfig,
    topic: str,
    max_fibers: int = 20,
) -> Narrative:
    """Generate a topic-focused narrative using spreading activation.

    Args:
        storage: Neural storage backend
        config: Brain configuration
        topic: Topic to explore
        max_fibers: Maximum fibers to include

    Returns:
        A Narrative with topic mode
    """
    from neural_memory.engine.retrieval import ReflexPipeline

    safe_limit = min(max_fibers, MAX_NARRATIVE_FIBERS)
    pipeline = ReflexPipeline(storage, config)
    result = await pipeline.query(topic, max_tokens=2000)

    items: list[NarrativeItem] = []
    seen_fibers: set[str] = set()

    for fiber_id in result.fibers_matched[:safe_limit]:
        if fiber_id in seen_fibers:
            continue
        seen_fibers.add(fiber_id)
        fiber = await storage.get_fiber(fiber_id)
        if fiber:
            items.append(_fiber_to_item(fiber, relevance=result.confidence))

    md = _render_topic_md(items, topic)
    return Narrative(
        mode="topic",
        title=f"Topic: {topic}",
        items=items,
        markdown=md,
    )


async def generate_causal_narrative(
    storage: NeuralStorage,
    topic: str,
    max_depth: int = 5,
) -> Narrative:
    """Generate a causal chain narrative.

    Finds neurons matching the topic, then traces CAUSED_BY chains.

    Args:
        storage: Neural storage backend
        topic: Starting concept to trace causality from
        max_depth: Maximum causal chain depth

    Returns:
        A Narrative with causal mode
    """
    from neural_memory.engine.causal_traversal import trace_causal_chain

    # Find seed neurons matching topic
    neurons = await storage.find_neurons(
        content_contains=topic,
        limit=5,
    )
    if not neurons:
        return Narrative(
            mode="causal",
            title=f"Causal Chain: {topic}",
            items=[],
            markdown=_render_causal_md([], topic),
        )

    # Use the first matching neuron as seed
    seed = neurons[0]
    chain = await trace_causal_chain(
        storage,
        seed_neuron_id=seed.id,
        direction="causes",
        max_depth=min(max_depth, 10),
    )

    items: list[NarrativeItem] = []
    # Add the seed itself
    items.append(
        NarrativeItem(
            fiber_id="",
            timestamp="",
            summary=seed.content,
            relevance=1.0,
        )
    )

    # Add each causal step
    for step in chain.steps:
        neuron = await storage.get_neuron(step.neuron_id)
        if neuron:
            items.append(
                NarrativeItem(
                    fiber_id="",
                    timestamp="",
                    summary=f"{neuron.content} (weight: {step.weight:.2f})",
                    relevance=step.weight,
                )
            )

    md = _render_causal_md(items, topic)
    return Narrative(
        mode="causal",
        title=f"Causal Chain: {topic}",
        items=items,
        markdown=md,
    )
