"""Tests for lazy entity promotion (B7).

Entities need 2+ mentions before being promoted to full neurons.
First mention → entity_ref only. Second mention → promote + retroactive link.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.pipeline import PipelineContext
from neural_memory.engine.pipeline_steps import ExtractEntityNeuronsStep
from neural_memory.utils.timeutils import utcnow


@dataclass
class FakeEntity:
    text: str
    type: MagicMock
    confidence: float = 0.5
    subtype: object = None
    raw_value: str = ""
    unit: str = ""


def _make_ctx(content: str = "Test content about PostgreSQL", **kwargs: object) -> PipelineContext:
    return PipelineContext(
        content=content,
        timestamp=utcnow(),
        metadata={},
        tags=set(),
        language="en",
        **kwargs,  # type: ignore[arg-type]
    )


def _make_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=[])
    storage.add_neuron = AsyncMock(return_value="new-id")
    storage.add_entity_ref = AsyncMock()
    storage.count_entity_refs = AsyncMock(return_value=0)
    storage.mark_entity_refs_promoted = AsyncMock(return_value=0)
    storage.get_entity_ref_fiber_ids = AsyncMock(return_value=[])
    return storage


def _make_config(lazy_enabled: bool = True, threshold: int = 2) -> BrainConfig:
    return BrainConfig(
        lazy_entity_enabled=lazy_enabled,
        lazy_entity_promotion_threshold=threshold,
    )


def _make_extractor(entities: list[FakeEntity]) -> MagicMock:
    extractor = MagicMock()
    extractor.extract.return_value = entities
    return extractor


# ── First mention → defer as ref ──


async def test_first_mention_defers_entity():
    """First mention of an entity should NOT create a neuron."""
    entity = FakeEntity(text="PostgreSQL", type=MagicMock(value="technology"))
    storage = _make_storage()
    storage.count_entity_refs.return_value = 0  # No prior mentions

    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor([entity]))
    ctx = _make_ctx()
    result = await step.execute(ctx, storage, _make_config())

    # Entity should be deferred, not created
    assert len(result.entity_neurons) == 0
    assert len(result.neurons_created) == 0
    assert "PostgreSQL" in result.deferred_entity_refs
    storage.add_neuron.assert_not_awaited()


# ── Second mention → promote ──


async def test_second_mention_promotes_entity():
    """Second mention should create the neuron and mark refs promoted."""
    entity = FakeEntity(text="PostgreSQL", type=MagicMock(value="technology"))
    storage = _make_storage()
    storage.count_entity_refs.return_value = 1  # 1 prior mention → threshold met

    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor([entity]))
    ctx = _make_ctx()
    result = await step.execute(ctx, storage, _make_config())

    # Entity should be promoted
    assert len(result.entity_neurons) == 1
    assert len(result.neurons_created) == 1
    storage.add_neuron.assert_awaited_once()
    storage.mark_entity_refs_promoted.assert_awaited_once_with("PostgreSQL")


# ── Existing entity → link only ──


async def test_existing_entity_not_duplicated():
    """Entity that already exists as a neuron should just be linked."""
    existing_neuron = Neuron.create(
        type=NeuronType.ENTITY,
        content="PostgreSQL",
    )
    entity = FakeEntity(text="PostgreSQL", type=MagicMock(value="technology"))
    storage = _make_storage()
    storage.find_neurons.return_value = [existing_neuron]

    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor([entity]))
    ctx = _make_ctx()
    result = await step.execute(ctx, storage, _make_config())

    # Existing entity skipped — not re-added or created
    assert len(result.entity_neurons) == 0
    assert len(result.neurons_created) == 0
    storage.add_neuron.assert_not_awaited()


# ── Lazy disabled → immediate promotion ──


async def test_lazy_disabled_creates_immediately():
    """When lazy entity is disabled, create neuron on first mention."""
    entity = FakeEntity(text="PostgreSQL", type=MagicMock(value="technology"))
    storage = _make_storage()

    config = _make_config(lazy_enabled=False)
    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor([entity]))
    ctx = _make_ctx()
    result = await step.execute(ctx, storage, config)

    assert len(result.entity_neurons) == 1
    assert len(result.neurons_created) == 1
    storage.add_neuron.assert_awaited_once()


# ── High confidence exception ──


async def test_high_confidence_always_promotes():
    """Entities with confidence >= 0.9 bypass lazy promotion."""
    entity = FakeEntity(text="PostgreSQL", type=MagicMock(value="technology"), confidence=0.95)
    storage = _make_storage()
    storage.count_entity_refs.return_value = 0  # No prior mentions

    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor([entity]))
    ctx = _make_ctx()
    result = await step.execute(ctx, storage, _make_config())

    # Should create immediately despite no prior mentions
    assert len(result.entity_neurons) == 1
    assert len(result.neurons_created) == 1


# ── User-tagged exception ──


async def test_user_tagged_always_promotes():
    """Entities matching user-provided tags bypass lazy promotion."""
    entity = FakeEntity(text="PostgreSQL", type=MagicMock(value="technology"))
    storage = _make_storage()
    storage.count_entity_refs.return_value = 0

    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor([entity]))
    ctx = _make_ctx()
    ctx.tags = {"postgresql", "database"}
    result = await step.execute(ctx, storage, _make_config())

    assert len(result.entity_neurons) == 1
    assert len(result.neurons_created) == 1


# ── Custom threshold ──


async def test_custom_threshold():
    """Promotion threshold can be customized."""
    entity = FakeEntity(text="Redis", type=MagicMock(value="technology"))
    storage = _make_storage()
    storage.count_entity_refs.return_value = 1  # Only 1 prior

    config = _make_config(threshold=3)  # Need 3 mentions
    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor([entity]))
    ctx = _make_ctx()
    result = await step.execute(ctx, storage, config)

    # 1 prior + 1 current = 2, threshold = 3 → still deferred
    assert len(result.entity_neurons) == 0
    assert "Redis" in result.deferred_entity_refs


async def test_threshold_3_promotes_on_third():
    """With threshold=3, promote on the 3rd mention."""
    entity = FakeEntity(text="Redis", type=MagicMock(value="technology"))
    storage = _make_storage()
    storage.count_entity_refs.return_value = 2  # 2 prior mentions

    config = _make_config(threshold=3)
    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor([entity]))
    ctx = _make_ctx()
    result = await step.execute(ctx, storage, config)

    assert len(result.entity_neurons) == 1
    assert len(result.neurons_created) == 1


# ── Multiple entities mixed ──


async def test_mixed_entities():
    """Some entities deferred, some promoted, some existing."""
    entities = [
        FakeEntity(text="PostgreSQL", type=MagicMock(value="technology")),  # existing
        FakeEntity(text="Redis", type=MagicMock(value="technology")),  # first mention
        FakeEntity(text="Docker", type=MagicMock(value="technology")),  # second mention
    ]

    existing_pg = Neuron.create(type=NeuronType.ENTITY, content="PostgreSQL")

    storage = _make_storage()

    async def mock_find(
        content_exact: str | None = None, content_contains: str | None = None, **kw: object
    ) -> list[Neuron]:
        if content_exact == "PostgreSQL":
            return [existing_pg]
        return []

    storage.find_neurons = AsyncMock(side_effect=mock_find)

    async def mock_count(entity_text: str) -> int:
        return {"Redis": 0, "Docker": 1}.get(entity_text, 0)

    storage.count_entity_refs = AsyncMock(side_effect=mock_count)

    step = ExtractEntityNeuronsStep(entity_extractor=_make_extractor(entities))
    ctx = _make_ctx()
    result = await step.execute(ctx, storage, _make_config())

    # PostgreSQL: existing → skipped
    # Redis: first mention → deferred
    # Docker: second mention → promoted
    assert len(result.entity_neurons) == 1  # Docker (promoted) only
    assert len(result.neurons_created) == 1  # Docker only
    assert "Redis" in result.deferred_entity_refs
    assert "PostgreSQL" not in result.deferred_entity_refs


# ── PipelineContext has deferred_entity_refs field ──


def test_pipeline_context_has_deferred_refs():
    """PipelineContext should have deferred_entity_refs field."""
    ctx = _make_ctx()
    assert hasattr(ctx, "deferred_entity_refs")
    assert ctx.deferred_entity_refs == []


# ── BrainConfig has lazy entity fields ──


def test_brain_config_lazy_entity_defaults():
    """BrainConfig should have lazy entity promotion settings."""
    config = BrainConfig()
    assert config.lazy_entity_enabled is True
    assert config.lazy_entity_promotion_threshold == 2
    assert config.lazy_entity_prune_days == 90
