"""Tests for neural_memory.engine.context_compiler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.context_compiler import (
    CompiledChunk,
    _dedup_groups,
    _merge_group,
    _rescore,
    compile_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    fiber_id: str,
    content: str,
    activation_score: float = 1.0,
    final_score: float = 0.0,
) -> CompiledChunk:
    return CompiledChunk(
        fiber_id=fiber_id,
        content=content,
        activation_score=activation_score,
        created_at=None,
        summary=None,
        final_score=final_score,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty() -> None:
    result = compile_context([], [])
    assert result == []


# ---------------------------------------------------------------------------
# Single chunk passthrough
# ---------------------------------------------------------------------------


def test_single_chunk_passes_through() -> None:
    c = _chunk("f1", "Hello world.", activation_score=0.8)
    result = compile_context([c], [])
    assert len(result) == 1
    assert result[0].fiber_id == "f1"
    assert result[0].content == "Hello world."
    # final_score should equal activation_score (no boost, no query terms)
    assert result[0].final_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_near_identical_chunks_deduped_to_one() -> None:
    """Two chunks with almost identical content should collapse into one."""
    c1 = _chunk("f1", "The quick brown fox jumps over the lazy dog.", activation_score=0.9)
    c2 = _chunk("f2", "The quick brown fox jumps over the lazy dog.", activation_score=0.5)
    result = compile_context([c1, c2], [])
    assert len(result) == 1
    # Primary should be the higher-activation chunk
    assert result[0].fiber_id == "f1"


def test_all_unique_chunks_preserved() -> None:
    """Completely different chunks should all survive dedup."""
    c1 = _chunk("f1", "Python is a programming language.", activation_score=0.7)
    c2 = _chunk("f2", "The Eiffel Tower is in Paris.", activation_score=0.6)
    c3 = _chunk("f3", "Quantum computing uses qubits.", activation_score=0.5)
    result = compile_context([c1, c2, c3], [])
    assert len(result) == 3


def test_dedup_groups_identical_content() -> None:
    """_dedup_groups should put identical-content chunks in the same group."""
    c1 = _chunk("f1", "same content here", activation_score=0.9)
    c2 = _chunk("f2", "same content here", activation_score=0.4)
    groups = _dedup_groups([c1, c2], threshold=10)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_dedup_groups_different_content() -> None:
    """_dedup_groups should put very different chunks in separate groups."""
    c1 = _chunk("f1", "Python is a programming language.", activation_score=0.9)
    c2 = _chunk("f2", "The Eiffel Tower is located in Paris, France.", activation_score=0.4)
    groups = _dedup_groups([c1, c2], threshold=10)
    assert len(groups) == 2


# ---------------------------------------------------------------------------
# Merge: unique sentences from secondary chunks appended
# ---------------------------------------------------------------------------


def test_merged_chunk_contains_unique_sentence() -> None:
    """After merge, the primary chunk should contain a unique sentence from the duplicate."""
    primary = _chunk("f1", "Cats are mammals.", activation_score=0.9)
    secondary = _chunk("f2", "Cats are mammals. They also purr loudly.", activation_score=0.3)
    merged = _merge_group([primary, secondary], max_extra_sentences=2)
    assert "purr loudly" in merged.content


def test_merge_does_not_duplicate_existing_sentences() -> None:
    """Sentences already in the primary should NOT be appended again."""
    primary = _chunk("f1", "Cats are mammals.", activation_score=0.9)
    secondary = _chunk("f2", "Cats are mammals.", activation_score=0.3)
    merged = _merge_group([primary, secondary], max_extra_sentences=2)
    # "Cats are mammals." appears only once
    assert merged.content.count("Cats are mammals") == 1


def test_max_extra_sentences_cap() -> None:
    """Only up to max_extra_sentences unique sentences are appended."""
    primary = _chunk("f1", "Sentence one.", activation_score=0.9)
    secondary = _chunk(
        "f2",
        "Sentence two. Sentence three. Sentence four. Sentence five.",
        activation_score=0.3,
    )
    merged = _merge_group([primary, secondary], max_extra_sentences=2)
    # Count how many of the secondary sentences appear
    appended_count = sum(
        1
        for s in ["Sentence two", "Sentence three", "Sentence four", "Sentence five"]
        if s in merged.content
    )
    assert appended_count <= 2


def test_merge_single_chunk_returns_unchanged() -> None:
    """A group with one chunk should return that chunk unchanged."""
    c = _chunk("f1", "Only chunk.", activation_score=0.7)
    result = _merge_group([c], max_extra_sentences=2)
    assert result is c


# ---------------------------------------------------------------------------
# Query boost
# ---------------------------------------------------------------------------


def test_query_boost_raises_relevant_chunk() -> None:
    """A chunk containing query terms should rank above one that does not."""
    c_relevant = _chunk("f1", "Python async programming is efficient.", activation_score=0.5)
    c_irrelevant = _chunk("f2", "The Eiffel Tower stands in Paris.", activation_score=0.7)
    result = compile_context([c_relevant, c_irrelevant], query_terms=["python", "async"])
    # c_relevant gets boost; c_irrelevant does not
    # c_irrelevant base=0.7, c_relevant base=0.5 + 0.3 cap = 0.8
    assert result[0].fiber_id == "f1"


def test_query_boost_is_case_insensitive() -> None:
    """Query term matching should be case-insensitive."""
    c = _chunk("f1", "PYTHON is great for scripting.", activation_score=0.5)
    rescored = _rescore([c], ["python"], boost_per_term=0.15, boost_cap=0.3)
    assert rescored[0].final_score == pytest.approx(0.65)


def test_query_boost_capped_at_boost_cap() -> None:
    """Total boost should not exceed boost_cap even with many matching terms."""
    c = _chunk("f1", "alpha beta gamma delta epsilon", activation_score=0.0)
    rescored = _rescore(
        [c],
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        boost_per_term=0.15,
        boost_cap=0.3,
    )
    assert rescored[0].final_score == pytest.approx(0.3)


def test_no_query_terms_preserves_activation_order() -> None:
    """With no query terms, chunks should be returned in activation order (descending)."""
    c1 = _chunk("f1", "First chunk.", activation_score=0.3)
    c2 = _chunk("f2", "Second chunk.", activation_score=0.8)
    c3 = _chunk("f3", "Third chunk.", activation_score=0.5)
    result = compile_context([c1, c2, c3], query_terms=[])
    scores = [r.final_score for r in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0].fiber_id == "f2"


# ---------------------------------------------------------------------------
# Immutability: inputs must not be mutated
# ---------------------------------------------------------------------------


def test_compile_context_does_not_mutate_inputs() -> None:
    """compile_context must not modify the input list or its chunks."""
    c1 = _chunk("f1", "Original content.", activation_score=0.6)
    c2 = _chunk("f2", "Original content.", activation_score=0.4)
    original_list = [c1, c2]
    original_content_1 = c1.content
    original_score_1 = c1.activation_score

    compile_context(original_list, ["original"])

    assert len(original_list) == 2, "Input list must not be mutated"
    assert c1.content == original_content_1
    assert c1.activation_score == original_score_1


# ---------------------------------------------------------------------------
# Integration: format_context_budgeted with compile step
# ---------------------------------------------------------------------------


def _make_fiber(fiber_id: str, anchor_id: str, summary: str | None = None) -> MagicMock:
    """Build a minimal mock Fiber for integration tests."""
    fiber = MagicMock()
    fiber.id = fiber_id
    fiber.anchor_neuron_id = anchor_id
    fiber.neuron_ids = [anchor_id]
    fiber.summary = summary
    fiber.salience = 0.5
    fiber.conductivity = 0.5
    fiber.created_at = None
    fiber.metadata = {}
    return fiber


def _make_anchor_neuron(neuron_id: str, content: str) -> MagicMock:
    """Build a minimal mock Neuron for integration tests."""
    neuron = MagicMock()
    neuron.id = neuron_id
    neuron.content = content
    return neuron


def _make_storage(anchor_map: dict[str, MagicMock]) -> MagicMock:
    """Build a mock NeuralStorage that returns anchor_map from get_neurons_batch."""
    storage = MagicMock()
    storage.get_neurons_batch = AsyncMock(return_value=anchor_map)
    storage.brain_id = "test-brain"
    return storage


@pytest.mark.asyncio
async def test_format_context_budgeted_deduplicates_overlapping_fibers() -> None:
    """Two fibers with nearly identical anchor content should produce one context entry
    when query_terms are provided (compile step runs)."""
    from neural_memory.engine.activation import ActivationResult
    from neural_memory.engine.retrieval_context import format_context_budgeted

    duplicate_text = (
        "The user prefers dark mode for all dashboard views. "
        "Settings are stored in config.toml under [ui] section."
    )

    fiber1 = _make_fiber("f1", "n1")
    fiber2 = _make_fiber("f2", "n2")

    neuron1 = _make_anchor_neuron("n1", duplicate_text)
    neuron2 = _make_anchor_neuron("n2", duplicate_text)

    storage = _make_storage({"n1": neuron1, "n2": neuron2})

    activations = {
        "n1": ActivationResult(
            neuron_id="n1",
            activation_level=0.9,
            hop_distance=0,
            path=["n1"],
            source_anchor="n1",
        ),
        "n2": ActivationResult(
            neuron_id="n2",
            activation_level=0.7,
            hop_distance=0,
            path=["n2"],
            source_anchor="n2",
        ),
    }

    context, _, allocation = await format_context_budgeted(
        storage=storage,
        activations=activations,
        fibers=[fiber1, fiber2],
        max_tokens=500,
        query_terms=["dark", "mode", "config"],
    )

    # With compilation, the near-duplicate fibers should collapse.
    # Parse only the "Relevant Memories" section (before "Related Information").
    relevant_section = context.split("## Related Information")[0]
    fiber_bullet_count = relevant_section.count("- ")
    assert fiber_bullet_count == 1, (
        f"Expected 1 fiber bullet after dedup, got {fiber_bullet_count}. Context:\n{context}"
    )
    assert "dark mode" in context.lower() or "config" in context.lower()
    # Budget allocation still sees both fibers selected before compile
    assert allocation.fibers_dropped == 0


@pytest.mark.asyncio
async def test_format_context_budgeted_backward_compat_no_query_terms() -> None:
    """Without query_terms, format_context_budgeted behaves exactly as before
    (no compilation, both fibers rendered)."""
    from neural_memory.engine.activation import ActivationResult
    from neural_memory.engine.retrieval_context import format_context_budgeted

    duplicate_text = "The user prefers dark mode for all dashboard views."

    fiber1 = _make_fiber("f1", "n1")
    fiber2 = _make_fiber("f2", "n2")

    neuron1 = _make_anchor_neuron("n1", duplicate_text)
    neuron2 = _make_anchor_neuron("n2", duplicate_text)

    storage = _make_storage({"n1": neuron1, "n2": neuron2})

    activations = {
        "n1": ActivationResult(
            neuron_id="n1",
            activation_level=0.9,
            hop_distance=0,
            path=["n1"],
            source_anchor="n1",
        ),
        "n2": ActivationResult(
            neuron_id="n2",
            activation_level=0.7,
            hop_distance=0,
            path=["n2"],
            source_anchor="n2",
        ),
    }

    # No query_terms → compile step skipped → both fibers rendered
    context, _, _ = await format_context_budgeted(
        storage=storage,
        activations=activations,
        fibers=[fiber1, fiber2],
        max_tokens=500,
        query_terms=None,
    )

    relevant_section = context.split("## Related Information")[0]
    fiber_bullet_count = relevant_section.count("- ")
    assert fiber_bullet_count == 2, (
        f"Expected 2 fiber bullets without dedup, got {fiber_bullet_count}. Context:\n{context}"
    )


@pytest.mark.asyncio
async def test_format_context_budgeted_compile_false_skips_dedup() -> None:
    """compile=False disables the compile step even when query_terms provided."""
    from neural_memory.engine.activation import ActivationResult
    from neural_memory.engine.retrieval_context import format_context_budgeted

    duplicate_text = "Same content repeated across fibers."

    fiber1 = _make_fiber("f1", "n1")
    fiber2 = _make_fiber("f2", "n2")

    neuron1 = _make_anchor_neuron("n1", duplicate_text)
    neuron2 = _make_anchor_neuron("n2", duplicate_text)

    storage = _make_storage({"n1": neuron1, "n2": neuron2})

    activations = {
        "n1": ActivationResult(
            neuron_id="n1",
            activation_level=0.8,
            hop_distance=0,
            path=["n1"],
            source_anchor="n1",
        ),
        "n2": ActivationResult(
            neuron_id="n2",
            activation_level=0.6,
            hop_distance=0,
            path=["n2"],
            source_anchor="n2",
        ),
    }

    context, _, _ = await format_context_budgeted(
        storage=storage,
        activations=activations,
        fibers=[fiber1, fiber2],
        max_tokens=500,
        query_terms=["content", "fibers"],
        compile=False,
    )

    # compile=False → no dedup → both fiber bullets present
    relevant_section = context.split("## Related Information")[0]
    fiber_bullet_count = relevant_section.count("- ")
    assert fiber_bullet_count == 2, (
        f"Expected 2 fiber bullets with compile=False, got {fiber_bullet_count}. Context:\n{context}"
    )
