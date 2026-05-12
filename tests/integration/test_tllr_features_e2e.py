"""End-to-end integration tests for the TLLR-learnings features.

Real `InMemoryStorage` + full `ReflexPipeline.query()` — no engine-level
mocks. Covers:

- Item #5 provenance footer in recall output
- Item #2 lifecycle status filter (active/superseded/expired)
- Item #3 validity window with `as_of` time-travel
- Item #1 BM25 lexical retrieval as parallel candidate source

Plus cross-feature combinations where bugs love to hide: superseded +
expired neuron, BM25 hit + status filter, legacy `_superseded=True` flag
side-by-side with new `_status` field, aware-TZ ISO round-trip.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.neuron import Neuron, NeuronStatus
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.reflex_conflict import pin_as_reflex
from neural_memory.engine.retrieval import ReflexPipeline
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.utils.timeutils import utcnow


@pytest_asyncio.fixture
async def storage() -> InMemoryStorage:
    s = InMemoryStorage()
    config = BrainConfig(
        activation_threshold=0.1,
        max_spread_hops=4,
        bm25_enabled=False,
    )
    brain = Brain.create(name="tllr-e2e", config=config)
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


@pytest_asyncio.fixture
async def storage_bm25() -> InMemoryStorage:
    """Variant with BM25 enabled for lexical-retrieval tests."""
    s = InMemoryStorage()
    config = BrainConfig(
        activation_threshold=0.1,
        max_spread_hops=4,
        bm25_enabled=True,
        bm25_limit=10,
    )
    brain = Brain.create(name="tllr-bm25", config=config)
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


async def _encode(storage: InMemoryStorage, content: str, *, source: str = "test") -> Neuron:
    """Encode + tag every created neuron with `_source` (mirrors remember_handler).

    Returns the actual fiber anchor (not `neurons_created[0]` which is
    often a TIME neuron). Mirrors the production behaviour where
    `remember_handler` iterates every created neuron — concept / entity
    subordinates also get attribution so their provenance footers show
    a real source instead of the "manual" fallback.
    """
    brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
    encoder = MemoryEncoder(storage, brain.config)  # type: ignore[union-attr]
    result = await encoder.encode(content)
    anchor_id = result.fiber.anchor_neuron_id
    for neuron in result.neurons_created:
        await storage.update_neuron(neuron.with_metadata(_source=source))
    anchor_or_default = next(
        (n for n in result.neurons_created if n.id == anchor_id),
        result.neurons_created[0],
    )
    return anchor_or_default.with_metadata(_source=source)


# ─────────────────── Item #5 — Provenance footer ───────────────────


class TestProvenanceE2E:
    @pytest.mark.asyncio
    async def test_provenance_footer_in_recall_output(self, storage: InMemoryStorage) -> None:
        await _encode(storage, "neural memory uses spreading activation", source="workflow")
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("spreading activation")
        assert "[src=" in result.context, (
            f"Provenance footer missing from recall output:\n{result.context}"
        )


# ─────────────────── Item #2 — Status filter ───────────────────


class TestStatusFilterE2E:
    @pytest.mark.asyncio
    async def test_superseded_neuron_filtered_from_default_recall(
        self, storage: InMemoryStorage
    ) -> None:
        await _encode(storage, "API key rotates monthly via vault")
        loser = await _encode(storage, "API key is sk-old-leaked-12345")
        # Mark loser superseded directly (matching `pin_as_reflex` outcome).
        await storage.update_neuron(loser.with_status(NeuronStatus.SUPERSEDED))

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("API key")

        # Default recall must not surface the superseded leaked key.
        assert "sk-old-leaked-12345" not in result.context

    @pytest.mark.asyncio
    async def test_include_status_override_surfaces_superseded(
        self, storage: InMemoryStorage
    ) -> None:
        loser = await _encode(storage, "old policy: store passwords in MD5")
        await storage.update_neuron(loser.with_status(NeuronStatus.SUPERSEDED))

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query(
            "password storage policy",
            include_status=frozenset({"active", "superseded"}),
        )
        assert "MD5" in result.context

    @pytest.mark.asyncio
    async def test_invalid_include_status_raises_at_engine_boundary(
        self, storage: InMemoryStorage
    ) -> None:
        await _encode(storage, "any content")
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="Invalid include_status"):
            await pipeline.query("anything", include_status=frozenset({"actve"}))

    @pytest.mark.asyncio
    async def test_legacy_superseded_flag_still_filtered(self, storage: InMemoryStorage) -> None:
        """Pre-Item-#2 brains used `_superseded=True` flag; status property maps it."""
        loser = await _encode(storage, "deprecated config option enabled by default")
        # Simulate a pre-Item-#2 state — only the boolean flag, no `_status`.
        await storage.update_neuron(loser.with_metadata(_superseded=True))

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("deprecated config")
        assert "deprecated config option enabled by default" not in result.context

    @pytest.mark.asyncio
    async def test_pin_as_reflex_revives_superseded_winner(self, storage: InMemoryStorage) -> None:
        """C1: re-pinning a formerly-superseded winner must revive it to ACTIVE."""
        n = await _encode(storage, "always validate user input")
        # Simulate a prior supersede cycle leaving stale flags.
        await storage.update_neuron(
            n.with_status(NeuronStatus.SUPERSEDED, superseded_by="winner-1")
        )
        await pin_as_reflex(n.id, storage, BrainConfig(max_reflexes=10))

        stored = await storage.get_neuron(n.id)
        assert stored is not None
        assert stored.status == NeuronStatus.ACTIVE
        assert "_superseded_by" not in stored.metadata


# ─────────────────── Item #3 — Validity window ───────────────────


class TestValidityE2E:
    @pytest.mark.asyncio
    async def test_expired_neuron_penalized_at_recall(self, storage: InMemoryStorage) -> None:
        n = await _encode(storage, "API key sk-q3-2025 expires September 30")
        past = (utcnow() - timedelta(days=30)).isoformat()
        await storage.update_neuron(n.with_metadata(_valid_until=past))

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("API key sk-q3-2025")

        # Penalty alone may keep high-activation neurons in result, but the
        # baseline activation is low here — should be penalized below threshold.
        assert "sk-q3-2025" not in result.context

    @pytest.mark.asyncio
    async def test_as_of_time_travel_keeps_then_valid_memory(
        self, storage: InMemoryStorage
    ) -> None:
        n = await _encode(storage, "Q1 2026 sprint goal: ship neural memory v5")
        until = datetime(2026, 3, 31, 23, 59, 59).isoformat()
        await storage.update_neuron(n.with_metadata(_valid_until=until))

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]

        # Query as if it's still Q1 — memory is in-window.
        result = await pipeline.query(
            "sprint goal",
            as_of=datetime(2026, 2, 15, 12, 0),
        )
        assert "v5" in result.context

    @pytest.mark.asyncio
    async def test_aware_iso_round_trip_does_not_crash(self, storage: InMemoryStorage) -> None:
        """Item #3 review C1: aware ISO with `+00:00` was crashing is_currently_valid."""
        n = await _encode(storage, "scheduled maintenance window")
        # Aware ISO is the natural form for international clients.
        await storage.update_neuron(n.with_metadata(_valid_until="2030-12-31T23:59:59+00:00"))

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("scheduled maintenance")
        # Pre-fix: TypeError on naive vs aware comparison. Post-fix: works.
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_offset_aware_datetime_object_normalized_to_utc(
        self, storage: InMemoryStorage
    ) -> None:
        """Item #3 review C2: `astimezone(None)` was producing local-time shift."""
        n = await _encode(storage, "boundary check fact")
        aware = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        await storage.update_neuron(n.with_validity(valid_until=aware))

        stored = await storage.get_neuron(n.id)
        assert stored is not None
        # No local-tz shift: stored value is the same UTC moment.
        assert stored.valid_until == datetime(2026, 6, 1, 12, 0)


# ─────────────────── Item #1 — BM25 lexical retrieval ───────────────────


class TestBM25E2E:
    @pytest.mark.asyncio
    async def test_bm25_keyword_query_surfaces_exact_match(
        self, storage_bm25: InMemoryStorage
    ) -> None:
        """A query whose keywords appear verbatim should hit BM25 even if
        semantic similarity is weak (rare token, no embedding match)."""
        await _encode(storage_bm25, "OpenClaw plugin uses ACP SDK v0.14.1 with 30s init timeout")
        await _encode(storage_bm25, "completely unrelated content about gardening")

        brain = await storage_bm25.get_brain(storage_bm25._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage_bm25, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("ACP SDK v0.14.1")

        assert "ACP SDK" in result.context

    @pytest.mark.asyncio
    async def test_bm25_off_does_not_change_default_behavior(
        self, storage: InMemoryStorage
    ) -> None:
        """`bm25_enabled=False` (default fixture) — pipeline runs without BM25 path."""
        await _encode(storage, "neural memory uses spreading activation")
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("activation")
        # Just verify no crash and we still get a result; the OFF path never
        # touches `_get_lexical_index`.
        assert result.latency_ms >= 0
        assert pipeline._lexical_index is None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_bm25_ghost_neuron_dropped_after_delete(
        self, storage_bm25: InMemoryStorage
    ) -> None:
        """Item #1 review C1: BM25 cache holds deleted IDs but recall must
        not surface them — `_bm25_anchors` validates via `get_neurons_batch`."""
        live = await _encode(storage_bm25, "alpha beta gamma keyword distinctive")
        ghost = await _encode(storage_bm25, "alpha beta gamma another keyword")

        brain = await storage_bm25.get_brain(storage_bm25._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage_bm25, brain.config)  # type: ignore[union-attr]
        # Build the index once.
        await pipeline.query("alpha beta gamma")

        # Delete the ghost without invalidating the BM25 cache.
        await storage_bm25.delete_neuron(ghost.id)

        # Subsequent query: the BM25 cache still has `ghost` but the
        # validation step drops it. The query MUST not crash on the
        # ghost ID and `live` must remain queryable.
        await pipeline.query("alpha beta gamma")
        assert live.id in {n.id for n in await storage_bm25.find_neurons(limit=100)}


# ─────────────────── Cross-feature combinations ───────────────────


class TestCrossFeature:
    @pytest.mark.asyncio
    async def test_superseded_with_validity_window_double_filtered(
        self, storage: InMemoryStorage
    ) -> None:
        """A neuron that is BOTH superseded AND past valid_until must not
        reach the activation pool — both filters must compose without crash."""
        n = await _encode(storage, "old policy with expiry")
        past = (utcnow() - timedelta(days=10)).isoformat()
        await storage.update_neuron(
            n.with_status(NeuronStatus.SUPERSEDED).with_metadata(_valid_until=past)
        )

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("old policy")
        assert "old policy with expiry" not in result.context

    @pytest.mark.asyncio
    async def test_bm25_hit_on_superseded_dropped_at_status_filter(
        self, storage_bm25: InMemoryStorage
    ) -> None:
        """BM25 surfaces a superseded ID; status filter must drop it."""
        n = await _encode(storage_bm25, "old credential xyz-deprecated-flag")
        await storage_bm25.update_neuron(n.with_status(NeuronStatus.SUPERSEDED))

        brain = await storage_bm25.get_brain(storage_bm25._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage_bm25, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("xyz-deprecated-flag")
        assert "xyz-deprecated-flag" not in result.context

    @pytest.mark.asyncio
    async def test_provenance_footer_renders_for_bm25_anchored_neuron(
        self, storage_bm25: InMemoryStorage
    ) -> None:
        await _encode(storage_bm25, "rare-keyword-zoo makes BM25 winner", source="auto-capture")
        brain = await storage_bm25.get_brain(storage_bm25._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage_bm25, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("rare-keyword-zoo")
        assert "[src=auto-capture" in result.context

    @pytest.mark.asyncio
    async def test_include_status_override_with_validity_still_penalizes(
        self, storage: InMemoryStorage
    ) -> None:
        """Caller asks for superseded ones explicitly, but expired ones
        within that set should still get the validity penalty applied."""
        n = await _encode(storage, "historical-record-zeta")
        past = (utcnow() - timedelta(days=10)).isoformat()
        await storage.update_neuron(
            n.with_status(NeuronStatus.SUPERSEDED).with_metadata(_valid_until=past)
        )

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        # Caller override: surface superseded — BUT validity penalty still
        # applies; with low baseline activation it falls below threshold.
        result = await pipeline.query(
            "historical record zeta",
            include_status=frozenset({"active", "superseded"}),
            as_of=utcnow(),
        )
        # Penalty drives below threshold → not in context.
        assert "historical-record-zeta" not in result.context

        # As-of inside the validity window: same neuron surfaces.
        past_when_valid = utcnow() - timedelta(days=15)
        result_history = await pipeline.query(
            "historical record zeta",
            include_status=frozenset({"active", "superseded"}),
            as_of=past_when_valid,
        )
        assert "historical-record-zeta" in result_history.context


# ─────────────────── Backward-compat smoke ───────────────────


class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_pre_tllr_brain_no_metadata_unaffected(self, storage: InMemoryStorage) -> None:
        """A neuron saved without any new metadata keys must recall normally."""
        await _encode(storage, "vanilla neuron with only legacy fields")
        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("vanilla")
        assert "vanilla" in result.context

    @pytest.mark.asyncio
    async def test_neuron_with_garbage_status_does_not_crash(
        self, storage: InMemoryStorage
    ) -> None:
        """L1: a corrupted `_status` value falls back to ACTIVE (with warning)."""
        n = await _encode(storage, "corrupted status content")
        await storage.update_neuron(n.with_metadata(_status="garbage-status"))

        brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
        pipeline = ReflexPipeline(storage, brain.config)  # type: ignore[union-attr]
        result = await pipeline.query("corrupted status")
        # Falls back to ACTIVE → still surfaces.
        assert "corrupted status content" in result.context
