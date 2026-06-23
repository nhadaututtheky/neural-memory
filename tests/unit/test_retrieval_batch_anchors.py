"""Regression test for #11: InfinityDB batch retrieval path dropping anchors.

The batch path (`find_neurons_by_content_batch`, used by InfinityDB) populates
`entity_anchors`/`keyword_anchors`, but the blocks appending them to the returned
`anchor_sets`/`ranked_lists` were nested inside the standard (SQLite) `else`, so the
batch backend silently discarded its two primary content retrievers. This test
exercises the batch branch and asserts the entity/keyword anchors survive.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.retrieval import ReflexPipeline
from neural_memory.extraction.entities import Entity, EntityType
from neural_memory.extraction.parser import Perspective, QueryIntent, Stimulus


def _neuron(content: str) -> Neuron:
    return Neuron.create(content=content, type=NeuronType.CONCEPT)


def _stimulus() -> Stimulus:
    return Stimulus(
        time_hints=[],
        keywords=["postgres"],
        entities=[Entity(text="PostgreSQL", type=EntityType.ORGANIZATION, start=0, end=10)],
        intent=QueryIntent.RECALL,
        perspective=Perspective.RECALL,
        raw_query="tell me about PostgreSQL postgres",
    )


@pytest.mark.asyncio
async def test_batch_path_keeps_entity_and_keyword_anchors() -> None:
    config = BrainConfig(
        fuzzy_search_enabled=False,
        graph_expansion_enabled=False,
        embedding_enabled=False,
        activation_strategy="reflex",
    )

    entity_neuron = _neuron("PostgreSQL is a database")
    keyword_neuron = _neuron("postgres tuning notes")

    storage = AsyncMock()
    # No time anchors.
    storage.find_neurons = AsyncMock(return_value=[])

    async def _batch(terms, **_kwargs):  # type: ignore[no-untyped-def]
        out: dict[str, list[Neuron]] = {}
        for t in terms:
            if t == "PostgreSQL":
                out[t] = [entity_neuron]
            else:
                # Any normalized keyword variant of "postgres"
                out[t] = [keyword_neuron]
        return out

    storage.find_neurons_by_content_batch = AsyncMock(side_effect=_batch)

    pipeline = ReflexPipeline(storage, config, use_reflex=True)

    anchor_sets, ranked_lists = await pipeline._find_anchors_ranked(_stimulus())

    # Batch path must have been taken.
    storage.find_neurons_by_content_batch.assert_awaited()

    all_anchor_ids = {nid for s in anchor_sets for nid in s}
    assert entity_neuron.id in all_anchor_ids, "entity anchor dropped on batch path (#11)"
    assert keyword_neuron.id in all_anchor_ids, "keyword anchor dropped on batch path (#11)"

    retrievers = {ra.retriever for rl in ranked_lists for ra in rl}
    assert "entity" in retrievers
    assert "keyword" in retrievers
