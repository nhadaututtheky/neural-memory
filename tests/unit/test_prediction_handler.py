"""Tests for cognitive handler prediction and verify tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.cognitive_handler import CognitiveHandler
from neural_memory.utils.timeutils import utcnow

# Patch at engine level since MemoryEncoder is imported lazily inside methods
_ENCODER_PATCH = "neural_memory.engine.encoder.MemoryEncoder"


def _make_storage(brain_id: str | None = "test-brain") -> AsyncMock:
    """Build a mock NeuralStorage with explicit brain_id."""
    storage = AsyncMock()
    storage._current_brain_id = brain_id
    storage.brain_id = brain_id
    storage.current_brain_id = brain_id
    storage.get_brain = AsyncMock(return_value=MagicMock(config=MagicMock()))
    storage.get_cognitive_state = AsyncMock(return_value=None)
    storage.list_predictions = AsyncMock(return_value=[])
    storage.get_calibration_stats = AsyncMock(
        return_value={"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}
    )
    storage.get_synapses = AsyncMock(return_value=[])
    return storage


def _make_handler(storage: AsyncMock) -> CognitiveHandler:
    """Build a CognitiveHandler with mocked storage."""
    handler = CognitiveHandler()
    handler.get_storage = AsyncMock(return_value=storage)  # type: ignore[attr-defined]
    handler.config = MagicMock()  # type: ignore[attr-defined]
    return handler


@dataclass(frozen=True)
class FakeNeuron:
    id: str = "n-1"
    content: str = "test neuron"


@dataclass(frozen=True)
class FakeFiber:
    id: str = "f-1"
    anchor_neuron_id: str = "n-1"


@dataclass(frozen=True)
class FakeEncodeResult:
    fiber: FakeFiber = field(default_factory=FakeFiber)
    neurons_created: list[Any] = field(default_factory=list)
    synapses_created: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class FakeSynapse:
    source_id: str = ""
    target_id: str = ""
    type: Any = None
    weight: float = 0.5


# ──────────────────── nmem_predict tests ────────────────────


class TestPredictCreate:
    """Tests for _predict action=create."""

    @pytest.mark.asyncio
    async def test_create_basic(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)

        with patch(_ENCODER_PATCH) as mock_encoder_cls:
            mock_enc = AsyncMock()
            mock_enc.encode = AsyncMock(return_value=FakeEncodeResult())
            mock_encoder_cls.return_value = mock_enc

            result = await handler._predict(
                {
                    "action": "create",
                    "content": "Next release will fix the memory leak",
                    "confidence": 0.8,
                }
            )

        assert result["status"] == "created"
        assert result["prediction_id"] == "n-1"
        assert result["confidence"] == 0.8
        storage.upsert_cognitive_state.assert_called_once()
        call_kwargs = storage.upsert_cognitive_state.call_args
        assert call_kwargs[1]["status"] == "pending"
        assert call_kwargs[1]["predicted_at"] is not None

    @pytest.mark.asyncio
    async def test_create_no_content(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._predict({"action": "create", "content": ""})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_content_too_long(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._predict(
            {
                "action": "create",
                "content": "x" * 100_001,
            }
        )
        assert "error" in result
        assert "100,000" in result["error"]

    @pytest.mark.asyncio
    async def test_create_no_brain(self) -> None:
        storage = _make_storage(brain_id=None)
        handler = _make_handler(storage)
        result = await handler._predict({"action": "create", "content": "test"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_with_deadline(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        future = (utcnow() + timedelta(days=7)).isoformat()

        with patch(_ENCODER_PATCH) as mock_encoder_cls:
            mock_enc = AsyncMock()
            mock_enc.encode = AsyncMock(return_value=FakeEncodeResult())
            mock_encoder_cls.return_value = mock_enc

            result = await handler._predict(
                {
                    "action": "create",
                    "content": "Tests will pass by next week",
                    "deadline": future,
                }
            )

        assert result["status"] == "created"
        assert result["deadline"] is not None

    @pytest.mark.asyncio
    async def test_create_past_deadline_rejected(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        past = (utcnow() - timedelta(days=1)).isoformat()

        result = await handler._predict(
            {
                "action": "create",
                "content": "Past prediction",
                "deadline": past,
            }
        )
        assert "error" in result
        assert "future" in result["error"]

    @pytest.mark.asyncio
    async def test_create_with_hypothesis_link(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            side_effect=[
                # First call: check hypothesis exists
                {
                    "neuron_id": "hyp-1",
                    "confidence": 0.6,
                    "status": "active",
                    "evidence_for_count": 1,
                    "evidence_against_count": 0,
                    "predicted_at": None,
                    "created_at": "2026-01-01",
                },
                # Second call: check prediction doesn't exist yet
                None,
            ]
        )
        handler = _make_handler(storage)

        with patch(_ENCODER_PATCH) as mock_encoder_cls:
            mock_enc = AsyncMock()
            mock_enc.encode = AsyncMock(return_value=FakeEncodeResult())
            mock_encoder_cls.return_value = mock_enc

            result = await handler._predict(
                {
                    "action": "create",
                    "content": "Based on hypothesis, X will happen",
                    "hypothesis_id": "hyp-1",
                }
            )

        assert result["status"] == "created"
        assert result["linked_hypothesis_id"] == "hyp-1"
        storage.add_synapse.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_invalid_hypothesis(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(return_value=None)
        handler = _make_handler(storage)

        with patch(_ENCODER_PATCH) as mock_encoder_cls:
            mock_enc = AsyncMock()
            mock_enc.encode = AsyncMock(return_value=FakeEncodeResult())
            mock_encoder_cls.return_value = mock_enc

            result = await handler._predict(
                {
                    "action": "create",
                    "content": "Prediction",
                    "hypothesis_id": "nonexistent",
                }
            )

        assert "error" in result
        assert "Hypothesis not found" in result["error"]

    @pytest.mark.asyncio
    async def test_create_existing_prediction(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)

        # After encoding, get_cognitive_state returns existing
        storage.get_cognitive_state = AsyncMock(
            return_value={"confidence": 0.7, "status": "pending"}
        )

        with patch(_ENCODER_PATCH) as mock_encoder_cls:
            mock_enc = AsyncMock()
            mock_enc.encode = AsyncMock(return_value=FakeEncodeResult())
            mock_encoder_cls.return_value = mock_enc

            result = await handler._predict(
                {
                    "action": "create",
                    "content": "Already exists",
                }
            )

        assert result["status"] == "existing"

    @pytest.mark.asyncio
    async def test_create_invalid_action(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._predict({"action": "delete"})
        assert "error" in result


class TestPredictList:
    """Tests for _predict action=list."""

    @pytest.mark.asyncio
    async def test_list_empty(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._predict({"action": "list"})
        assert result["count"] == 0
        assert "calibration" in result
        assert result["calibration"]["score"] == 0.5  # Default when no data

    @pytest.mark.asyncio
    async def test_list_with_results(self) -> None:
        storage = _make_storage()
        storage.list_predictions = AsyncMock(
            return_value=[
                {
                    "neuron_id": "n-1",
                    "confidence": 0.7,
                    "status": "pending",
                    "predicted_at": "2026-04-01",
                    "resolved_at": None,
                    "created_at": "2026-03-01",
                },
            ]
        )
        storage.get_neuron = AsyncMock(return_value=FakeNeuron())
        handler = _make_handler(storage)

        result = await handler._predict({"action": "list"})
        assert result["count"] == 1
        assert result["predictions"][0]["prediction_id"] == "n-1"
        assert result["predictions"][0]["deadline"] == "2026-04-01"

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        await handler._predict({"action": "list", "status": "pending"})
        storage.list_predictions.assert_called_once_with(status="pending", limit=20)

    @pytest.mark.asyncio
    async def test_list_invalid_status(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._predict({"action": "list", "status": "bogus"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_bad_limit(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        await handler._predict({"action": "list", "limit": "abc"})
        # Should fallback to 20
        storage.list_predictions.assert_called_once_with(status=None, limit=20)


class TestPredictGet:
    """Tests for _predict action=get."""

    @pytest.mark.asyncio
    async def test_get_missing_id(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._predict({"action": "get"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_not_found(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._predict({"action": "get", "prediction_id": "nope"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_not_prediction(self) -> None:
        """Should reject cognitive states without predicted_at."""
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "n-1",
                "confidence": 0.5,
                "status": "active",
                "predicted_at": None,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "created_at": "2026-01-01",
            }
        )
        handler = _make_handler(storage)
        result = await handler._predict({"action": "get", "prediction_id": "n-1"})
        assert "error" in result
        assert "not a prediction" in result["error"]

    @pytest.mark.asyncio
    async def test_get_success(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "n-1",
                "confidence": 0.7,
                "status": "pending",
                "predicted_at": "2026-04-01",
                "resolved_at": None,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "created_at": "2026-03-01",
            }
        )
        storage.get_neuron = AsyncMock(return_value=FakeNeuron())
        handler = _make_handler(storage)

        result = await handler._predict({"action": "get", "prediction_id": "n-1"})
        assert result["prediction_id"] == "n-1"
        assert result["deadline"] == "2026-04-01"
        assert result["status"] == "pending"


# ──────────────────── nmem_verify tests ────────────────────


class TestVerify:
    """Tests for _verify."""

    @pytest.mark.asyncio
    async def test_verify_missing_id(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._verify({"outcome": "correct"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_verify_invalid_outcome(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._verify({"prediction_id": "n-1", "outcome": "maybe"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_verify_no_brain(self) -> None:
        storage = _make_storage(brain_id=None)
        handler = _make_handler(storage)
        result = await handler._verify({"prediction_id": "n-1", "outcome": "correct"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_verify_not_found(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._verify({"prediction_id": "nope", "outcome": "correct"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_verify_not_prediction(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "n-1",
                "confidence": 0.5,
                "status": "active",
                "predicted_at": None,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "created_at": "2026-01-01",
            }
        )
        handler = _make_handler(storage)
        result = await handler._verify({"prediction_id": "n-1", "outcome": "correct"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_verify_already_resolved(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "n-1",
                "confidence": 0.9,
                "status": "confirmed",
                "predicted_at": "2026-04-01",
                "resolved_at": "2026-03-15",
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "created_at": "2026-03-01",
            }
        )
        handler = _make_handler(storage)
        result = await handler._verify({"prediction_id": "n-1", "outcome": "correct"})
        assert "error" in result
        assert "already confirmed" in result["error"]

    @pytest.mark.asyncio
    async def test_verify_correct_no_content(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "n-1",
                "confidence": 0.7,
                "status": "pending",
                "predicted_at": "2026-04-01",
                "resolved_at": None,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "created_at": "2026-03-01",
            }
        )
        handler = _make_handler(storage)

        result = await handler._verify({"prediction_id": "n-1", "outcome": "correct"})
        assert result["status"] == "verified"
        assert result["outcome"] == "correct"
        assert result["prediction_status"] == "confirmed"
        assert "calibration_score" in result
        assert "observation_id" not in result

    @pytest.mark.asyncio
    async def test_verify_wrong_with_content(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "n-1",
                "confidence": 0.7,
                "status": "pending",
                "predicted_at": "2026-04-01",
                "resolved_at": None,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "created_at": "2026-03-01",
            }
        )
        handler = _make_handler(storage)

        with patch(_ENCODER_PATCH) as mock_encoder_cls:
            mock_enc = AsyncMock()
            mock_enc.encode = AsyncMock(return_value=FakeEncodeResult())
            mock_encoder_cls.return_value = mock_enc

            result = await handler._verify(
                {
                    "prediction_id": "n-1",
                    "outcome": "wrong",
                    "content": "Actually the opposite happened",
                }
            )

        assert result["status"] == "verified"
        assert result["outcome"] == "wrong"
        assert result["prediction_status"] == "refuted"
        assert result["observation_id"] == "n-1"  # anchor of encoded observation

    @pytest.mark.asyncio
    async def test_verify_propagates_to_hypothesis(self) -> None:
        """Correct prediction should propagate evidence-for to linked hypothesis."""
        from neural_memory.core.synapse import SynapseType

        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            side_effect=[
                # First call: get prediction state
                {
                    "neuron_id": "pred-1",
                    "confidence": 0.7,
                    "status": "pending",
                    "predicted_at": "2026-04-01",
                    "resolved_at": None,
                    "evidence_for_count": 0,
                    "evidence_against_count": 0,
                    "created_at": "2026-03-01",
                },
                # Second call: get linked hypothesis state
                {
                    "neuron_id": "hyp-1",
                    "confidence": 0.6,
                    "status": "active",
                    "predicted_at": None,
                    "resolved_at": None,
                    "evidence_for_count": 1,
                    "evidence_against_count": 0,
                    "created_at": "2026-02-01",
                },
            ]
        )
        # Return PREDICTED synapse linking prediction -> hypothesis
        storage.get_synapses = AsyncMock(
            return_value=[
                FakeSynapse(source_id="pred-1", target_id="hyp-1", type=SynapseType.PREDICTED),
            ]
        )
        handler = _make_handler(storage)

        result = await handler._verify({"prediction_id": "pred-1", "outcome": "correct"})
        assert result["status"] == "verified"
        assert "propagated_to_hypothesis" in result
        prop = result["propagated_to_hypothesis"]
        assert prop["hypothesis_id"] == "hyp-1"
        assert prop["evidence_type"] == "for"
        assert prop["confidence_after"] > 0.6  # Should increase

    @pytest.mark.asyncio
    async def test_verify_wrong_propagates_against(self) -> None:
        """Wrong prediction should propagate evidence-against to linked hypothesis."""
        from neural_memory.core.synapse import SynapseType

        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            side_effect=[
                # prediction
                {
                    "neuron_id": "pred-1",
                    "confidence": 0.7,
                    "status": "pending",
                    "predicted_at": "2026-04-01",
                    "resolved_at": None,
                    "evidence_for_count": 0,
                    "evidence_against_count": 0,
                    "created_at": "2026-03-01",
                },
                # hypothesis
                {
                    "neuron_id": "hyp-1",
                    "confidence": 0.6,
                    "status": "active",
                    "predicted_at": None,
                    "resolved_at": None,
                    "evidence_for_count": 1,
                    "evidence_against_count": 0,
                    "created_at": "2026-02-01",
                },
            ]
        )
        storage.get_synapses = AsyncMock(
            return_value=[
                FakeSynapse(source_id="pred-1", target_id="hyp-1", type=SynapseType.PREDICTED),
            ]
        )
        handler = _make_handler(storage)

        result = await handler._verify({"prediction_id": "pred-1", "outcome": "wrong"})
        assert result["propagated_to_hypothesis"]["evidence_type"] == "against"
        assert result["propagated_to_hypothesis"]["confidence_after"] < 0.6

    @pytest.mark.asyncio
    async def test_verify_calibration_updates(self) -> None:
        """Calibration stats should reflect the verified prediction."""
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "n-1",
                "confidence": 0.7,
                "status": "pending",
                "predicted_at": "2026-04-01",
                "resolved_at": None,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "created_at": "2026-03-01",
            }
        )
        storage.get_calibration_stats = AsyncMock(
            return_value={
                "correct_count": 3,
                "wrong_count": 1,
                "total_resolved": 4,
                "pending_count": 2,
            }
        )
        handler = _make_handler(storage)

        result = await handler._verify({"prediction_id": "n-1", "outcome": "correct"})
        assert result["calibration_score"] == 0.75  # 3/4
        assert result["calibration_stats"]["correct"] == 3


# ──────────────────── Storage mixin tests ────────────────────


class TestSQLiteCognitivePredictions:
    """Integration tests for prediction-related storage methods."""

    @pytest.mark.asyncio
    async def test_list_predictions_empty(self) -> None:
        import aiosqlite

        db = await aiosqlite.connect(":memory:")
        await db.execute("""CREATE TABLE cognitive_state (
            brain_id TEXT, neuron_id TEXT, confidence REAL,
            evidence_for_count INTEGER DEFAULT 0,
            evidence_against_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending', predicted_at TEXT,
            resolved_at TEXT, schema_version INTEGER DEFAULT 1,
            parent_schema_id TEXT, last_evidence_at TEXT, created_at TEXT,
            PRIMARY KEY (brain_id, neuron_id)
        )""")

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        mixin = SQLiteCognitiveMixin()
        mixin._ensure_read_conn = lambda: db  # type: ignore[assignment]
        mixin._get_brain_id = lambda: "b1"  # type: ignore[assignment]

        result = await mixin.list_predictions()
        assert result == []
        await db.close()

    @pytest.mark.asyncio
    async def test_list_predictions_filters_correctly(self) -> None:
        import aiosqlite

        db = await aiosqlite.connect(":memory:")
        await db.execute("""CREATE TABLE cognitive_state (
            brain_id TEXT, neuron_id TEXT, confidence REAL,
            evidence_for_count INTEGER DEFAULT 0,
            evidence_against_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending', predicted_at TEXT,
            resolved_at TEXT, schema_version INTEGER DEFAULT 1,
            parent_schema_id TEXT, last_evidence_at TEXT, created_at TEXT,
            PRIMARY KEY (brain_id, neuron_id)
        )""")
        # Insert a hypothesis (no predicted_at) and a prediction
        await db.execute(
            "INSERT INTO cognitive_state (brain_id, neuron_id, confidence, status, predicted_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("b1", "hyp-1", 0.5, "active", None, "2026-01-01"),
        )
        await db.execute(
            "INSERT INTO cognitive_state (brain_id, neuron_id, confidence, status, predicted_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("b1", "pred-1", 0.7, "pending", "2026-04-01", "2026-03-01"),
        )
        await db.commit()

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        mixin = SQLiteCognitiveMixin()
        mixin._ensure_read_conn = lambda: db  # type: ignore[assignment]
        mixin._get_brain_id = lambda: "b1"  # type: ignore[assignment]

        result = await mixin.list_predictions()
        assert len(result) == 1
        assert result[0]["neuron_id"] == "pred-1"
        await db.close()

    @pytest.mark.asyncio
    async def test_calibration_stats(self) -> None:
        import aiosqlite

        db = await aiosqlite.connect(":memory:")
        await db.execute("""CREATE TABLE cognitive_state (
            brain_id TEXT, neuron_id TEXT, confidence REAL,
            evidence_for_count INTEGER DEFAULT 0,
            evidence_against_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending', predicted_at TEXT,
            resolved_at TEXT, schema_version INTEGER DEFAULT 1,
            parent_schema_id TEXT, last_evidence_at TEXT, created_at TEXT,
            PRIMARY KEY (brain_id, neuron_id)
        )""")
        # 2 correct, 1 wrong, 1 pending
        for i, (status, pat) in enumerate(
            [
                ("confirmed", "2026-04-01"),
                ("confirmed", "2026-04-02"),
                ("refuted", "2026-04-03"),
                ("pending", "2026-05-01"),
            ]
        ):
            await db.execute(
                "INSERT INTO cognitive_state (brain_id, neuron_id, confidence, status, predicted_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("b1", f"p-{i}", 0.5, status, pat, "2026-03-01"),
            )
        await db.commit()

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        mixin = SQLiteCognitiveMixin()
        mixin._ensure_read_conn = lambda: db  # type: ignore[assignment]
        mixin._get_brain_id = lambda: "b1"  # type: ignore[assignment]

        stats = await mixin.get_calibration_stats()
        assert stats["correct_count"] == 2
        assert stats["wrong_count"] == 1
        assert stats["total_resolved"] == 3
        assert stats["pending_count"] == 1
        await db.close()
