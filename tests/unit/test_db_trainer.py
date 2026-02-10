"""Tests for db_trainer: DB-to-Brain training pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.neuron import NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.db_introspector import (
    ColumnInfo,
    ForeignKeyInfo,
    SchemaSnapshot,
    TableInfo,
)
from neural_memory.engine.db_knowledge import (
    KnowledgeEntity,
    KnowledgePattern,
    KnowledgeProperty,
    KnowledgeRelationship,
    SchemaKnowledge,
    SchemaPatternType,
)
from neural_memory.engine.db_trainer import (
    DBTrainer,
    DBTrainingConfig,
    DBTrainingResult,
)

# ── Frozen dataclass tests ──────────────────────────────────────


class TestFrozenDataclasses:
    """Training config and result dataclasses are immutable."""

    def test_config_frozen(self) -> None:
        cfg = DBTrainingConfig(connection_string="sqlite:///test.db")
        with pytest.raises(AttributeError):
            cfg.connection_string = "changed"  # type: ignore[misc]

    def test_result_frozen(self) -> None:
        result = DBTrainingResult(tables_processed=5)
        with pytest.raises(AttributeError):
            result.tables_processed = 10  # type: ignore[misc]

    def test_config_defaults(self) -> None:
        cfg = DBTrainingConfig(connection_string="sqlite:///test.db")
        assert cfg.domain_tag == ""
        assert cfg.brain_name == ""
        assert cfg.consolidate is True
        assert cfg.salience_ceiling == 0.5
        assert cfg.initial_stage == "episodic"
        assert cfg.include_patterns is True
        assert cfg.include_relationships is True
        assert cfg.max_tables == 100

    def test_result_defaults(self) -> None:
        result = DBTrainingResult(tables_processed=0)
        assert result.tables_skipped == 0
        assert result.columns_processed == 0
        assert result.relationships_mapped == 0
        assert result.patterns_detected == 0
        assert result.neurons_created == 0
        assert result.synapses_created == 0
        assert result.enrichment_synapses == 0
        assert result.schema_fingerprint == ""
        assert result.brain_name == "current"


# ── Helpers ─────────────────────────────────────────────────────


def _make_snapshot() -> SchemaSnapshot:
    """Minimal schema snapshot for testing."""
    return SchemaSnapshot(
        database_name="test.db",
        dialect="sqlite",
        tables=(
            TableInfo(
                name="users",
                schema=None,
                columns=(
                    ColumnInfo("id", "INTEGER", False, True),
                    ColumnInfo("name", "TEXT", False, False),
                    ColumnInfo("email", "TEXT", False, False),
                ),
                foreign_keys=(),
                indexes=(),
                row_count_estimate=10,
            ),
            TableInfo(
                name="posts",
                schema=None,
                columns=(
                    ColumnInfo("id", "INTEGER", False, True),
                    ColumnInfo("title", "TEXT", False, False),
                    ColumnInfo("user_id", "INTEGER", False, False),
                ),
                foreign_keys=(
                    ForeignKeyInfo("user_id", "users", "id", None, None),
                ),
                indexes=(),
                row_count_estimate=50,
            ),
        ),
        schema_fingerprint="abc123",
    )


def _make_knowledge() -> SchemaKnowledge:
    """Minimal knowledge for testing."""
    return SchemaKnowledge(
        entities=(
            KnowledgeEntity(
                table_name="users",
                description="Database table 'users' stores user records.",
                column_summary="id (INTEGER), name (TEXT), email (TEXT)",
                row_count_estimate=10,
                business_purpose="stores user records",
                confidence=0.75,
            ),
            KnowledgeEntity(
                table_name="posts",
                description="Database table 'posts' stores post records. Links to: users.",
                column_summary="id (INTEGER), title (TEXT), user_id (INTEGER)",
                row_count_estimate=50,
                business_purpose="stores post records",
                confidence=0.75,
            ),
        ),
        relationships=(
            KnowledgeRelationship(
                source_table="posts",
                source_column="user_id",
                target_table="users",
                target_column="id",
                synapse_type=SynapseType.INVOLVES,
                confidence=0.75,
            ),
        ),
        patterns=(
            KnowledgePattern(
                pattern_type=SchemaPatternType.AUDIT_TRAIL,
                table_name="users",
                evidence={"has_timestamps": True},
                description="Table 'users' uses audit trail pattern",
                confidence=0.70,
            ),
        ),
        properties=(
            KnowledgeProperty("users", "id", "INTEGER", ("PRIMARY KEY",), "primary identifier"),
            KnowledgeProperty("users", "name", "TEXT", ("NOT NULL",), "display name"),
        ),
    )


def _mock_encode_result(anchor_id: str = "neuron-1") -> MagicMock:
    """Create a mock EncodingResult."""
    result = MagicMock()
    result.neurons_created = [MagicMock()]
    result.synapses_created = [MagicMock()]
    result.fiber = MagicMock()
    result.fiber.anchor_neuron_id = anchor_id
    return result


def _build_mock_storage() -> AsyncMock:
    """Build a mock NeuralStorage."""
    storage = AsyncMock()
    storage._current_brain_id = "brain-1"
    storage.disable_auto_save = MagicMock()
    storage.enable_auto_save = MagicMock()
    storage.batch_save = AsyncMock()
    storage.add_neuron = AsyncMock()
    storage.add_synapse = AsyncMock()
    return storage


def _build_mock_config() -> MagicMock:
    """Build a mock BrainConfig."""
    config = MagicMock()
    config.name = "test-brain"
    return config


# ── DBTrainer batch pattern tests ───────────────────────────────


class TestDBTrainerBatchPattern:
    """Verifies disable_auto_save → process → batch_save → enable pattern."""

    @pytest.mark.asyncio
    async def test_batch_save_called(self) -> None:
        """batch_save is called after processing."""
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            trainer._extractor.extract = MagicMock(return_value=_make_knowledge())
            trainer._encoder.encode = AsyncMock(return_value=_mock_encode_result())

            tc = DBTrainingConfig(
                connection_string="sqlite:///test.db",
                consolidate=False,
            )
            await trainer.train(tc)

            storage.disable_auto_save.assert_called_once()
            storage.batch_save.assert_called_once()
            storage.enable_auto_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_save_restored_on_error(self) -> None:
        """enable_auto_save is called even when batch_save raises."""
        storage = _build_mock_storage()
        storage.batch_save = AsyncMock(side_effect=RuntimeError("disk full"))

        with patch.object(DBTrainer, "__init__", lambda self, *a, **kw: None):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = _build_mock_config()
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()
            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            empty_knowledge = SchemaKnowledge((), (), (), ())
            trainer._extractor.extract = MagicMock(return_value=empty_knowledge)

            tc = DBTrainingConfig(connection_string="sqlite:///test.db", consolidate=False)
            with pytest.raises(RuntimeError, match="disk full"):
                await trainer.train(tc)
            storage.enable_auto_save.assert_called_once()


# ── DBTrainer entity encoding ───────────────────────────────────


class TestDBTrainerEncoding:
    """Verifies entities and patterns are encoded correctly."""

    @pytest.mark.asyncio
    async def test_entities_encoded(self) -> None:
        """Each entity gets encoded via MemoryEncoder."""
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            trainer._extractor.extract = MagicMock(return_value=_make_knowledge())

            call_count = 0

            async def mock_encode(**kwargs):
                nonlocal call_count
                call_count += 1
                return _mock_encode_result(f"neuron-{call_count}")

            trainer._encoder.encode = mock_encode

            tc = DBTrainingConfig(
                connection_string="sqlite:///test.db",
                consolidate=False,
            )
            result = await trainer.train(tc)

            # 2 entities + 1 pattern = 3 encode calls
            assert call_count == 3
            assert result.tables_processed == 2
            assert result.patterns_detected == 1

    @pytest.mark.asyncio
    async def test_error_isolation_per_table(self) -> None:
        """One table failing doesn't abort others."""
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())

            # Knowledge with 2 entities, no patterns/relationships
            knowledge = SchemaKnowledge(
                entities=(
                    KnowledgeEntity("t1", "desc1", "col", 0, "purpose", 0.75),
                    KnowledgeEntity("t2", "desc2", "col", 0, "purpose", 0.75),
                ),
                relationships=(),
                patterns=(),
                properties=(),
            )
            trainer._extractor.extract = MagicMock(return_value=knowledge)

            call_count = 0

            async def mock_encode_with_failure(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("encoding failed")
                return _mock_encode_result("neuron-2")

            trainer._encoder.encode = mock_encode_with_failure

            tc = DBTrainingConfig(
                connection_string="sqlite:///test.db",
                consolidate=False,
                include_patterns=False,
            )
            result = await trainer.train(tc)

            # First table failed, second succeeded
            assert result.tables_processed == 1
            assert call_count == 2


# ── max_tables guard ────────────────────────────────────────────


class TestMaxTablesGuard:
    """max_tables limits how many tables are encoded."""

    @pytest.mark.asyncio
    async def test_max_tables_limits_processing(self) -> None:
        """With max_tables=1, only 1 of 2 entities is processed."""
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            trainer._extractor.extract = MagicMock(return_value=_make_knowledge())
            trainer._encoder.encode = AsyncMock(return_value=_mock_encode_result())

            tc = DBTrainingConfig(
                connection_string="sqlite:///test.db",
                consolidate=False,
                max_tables=1,
                include_patterns=False,
            )
            result = await trainer.train(tc)

            assert result.tables_processed == 1
            assert result.tables_skipped == 1


# ── Domain neuron ───────────────────────────────────────────────


class TestDomainNeuron:
    """Domain neuron created when domain_tag is set."""

    @pytest.mark.asyncio
    async def test_domain_neuron_created(self) -> None:
        """Domain tag → domain CONCEPT neuron created."""
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            trainer._extractor.extract = MagicMock(return_value=_make_knowledge())
            trainer._encoder.encode = AsyncMock(return_value=_mock_encode_result())

            tc = DBTrainingConfig(
                connection_string="sqlite:///test.db",
                domain_tag="ecommerce",
                consolidate=False,
            )
            await trainer.train(tc)

            # Domain neuron added via storage.add_neuron
            assert storage.add_neuron.call_count >= 1
            domain_call = storage.add_neuron.call_args_list[0]
            neuron = domain_call[0][0]
            assert neuron.type == NeuronType.CONCEPT
            assert "ecommerce" in neuron.content

    @pytest.mark.asyncio
    async def test_no_domain_neuron_without_tag(self) -> None:
        """Empty domain_tag → no domain neuron."""
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())

            # Empty knowledge = no entities to trigger add_neuron
            empty_knowledge = SchemaKnowledge((), (), (), ())
            trainer._extractor.extract = MagicMock(return_value=empty_knowledge)

            tc = DBTrainingConfig(
                connection_string="sqlite:///test.db",
                domain_tag="",
                consolidate=False,
            )
            await trainer.train(tc)

            # No domain neuron → add_neuron not called
            storage.add_neuron.assert_not_called()


# ── Relationship synapses ───────────────────────────────────────


class TestRelationshipSynapses:
    """FK relationships create direct synapses."""

    @pytest.mark.asyncio
    async def test_relationship_synapse_created(self) -> None:
        """FK relationship → Synapse.create with correct type and weight."""
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            trainer._extractor.extract = MagicMock(return_value=_make_knowledge())

            call_idx = 0

            async def mock_encode(**kwargs):
                nonlocal call_idx
                call_idx += 1
                return _mock_encode_result(f"anchor-{call_idx}")

            trainer._encoder.encode = mock_encode

            tc = DBTrainingConfig(
                connection_string="sqlite:///test.db",
                consolidate=False,
                include_patterns=False,
            )
            result = await trainer.train(tc)

            assert result.relationships_mapped == 1
            # Synapse added for the relationship
            synapse_calls = [
                c for c in storage.add_synapse.call_args_list
                if c[0][0].metadata.get("db_relationship")
            ]
            assert len(synapse_calls) == 1

    @pytest.mark.asyncio
    async def test_no_relationship_when_disabled(self) -> None:
        """include_relationships=False → no relationship synapses."""
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            trainer._extractor.extract = MagicMock(return_value=_make_knowledge())
            trainer._encoder.encode = AsyncMock(return_value=_mock_encode_result())

            tc = DBTrainingConfig(
                connection_string="sqlite:///test.db",
                consolidate=False,
                include_relationships=False,
                include_patterns=False,
            )
            result = await trainer.train(tc)

            assert result.relationships_mapped == 0


# ── Introspection failure ───────────────────────────────────────


class TestIntrospectionFailure:
    """Introspection failures raise ValueError."""

    @pytest.mark.asyncio
    async def test_introspection_error_raises_value_error(self) -> None:
        storage = _build_mock_storage()
        config = _build_mock_config()

        with (
            patch.object(
                DBTrainer, "__init__", lambda self, *a, **kw: None
            ),
        ):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = config
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()

            trainer._introspector.introspect = AsyncMock(
                side_effect=RuntimeError("connection refused")
            )

            tc = DBTrainingConfig(connection_string="sqlite:///bad.db")
            with pytest.raises(ValueError, match="Failed to introspect"):
                await trainer.train(tc)


# ── MCP handler tests ───────────────────────────────────────────


class TestDBTrainHandler:
    """Tests for mcp/db_train_handler.py validation."""

    def _make_handler(self) -> MagicMock:
        """Build a mock handler with _train_db methods and storage."""
        from neural_memory.mcp.db_train_handler import DBTrainHandler

        handler = MagicMock(spec=DBTrainHandler)
        handler._train_db = DBTrainHandler._train_db.__get__(handler)
        handler._train_db_schema = DBTrainHandler._train_db_schema.__get__(handler)
        handler._train_db_status = DBTrainHandler._train_db_status.__get__(handler)

        # Mock get_storage → returns mock storage with brain
        mock_brain = MagicMock()
        mock_brain.config = MagicMock()
        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "brain-1"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        handler.get_storage = AsyncMock(return_value=mock_storage)

        return handler

    @pytest.mark.asyncio
    async def test_missing_connection_string(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db_schema({"connection_string": ""})
        assert "error" in result
        assert "required" in result["error"]

    @pytest.mark.asyncio
    async def test_connection_string_too_long(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db_schema(
            {"connection_string": "sqlite:///" + "a" * 500}
        )
        assert "error" in result
        assert "too long" in result["error"]

    @pytest.mark.asyncio
    async def test_non_sqlite_rejected(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db_schema(
            {"connection_string": "postgresql://localhost/db"}
        )
        assert "error" in result
        assert "SQLite" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_max_tables(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db_schema(
            {"connection_string": "sqlite:///test.db", "max_tables": -1}
        )
        assert "error" in result
        assert "max_tables" in result["error"]

    @pytest.mark.asyncio
    async def test_domain_tag_too_long(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db_schema(
            {
                "connection_string": "sqlite:///test.db",
                "domain_tag": "x" * 101,
            }
        )
        assert "error" in result
        assert "domain_tag" in result["error"]

    @pytest.mark.asyncio
    async def test_brain_name_too_long(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db_schema(
            {
                "connection_string": "sqlite:///test.db",
                "brain_name": "x" * 65,
            }
        )
        assert "error" in result
        assert "brain_name" in result["error"]

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db({"action": "unknown"})
        assert "error" in result
        assert "Unknown" in result["error"]

    @pytest.mark.asyncio
    async def test_no_brain_configured(self) -> None:
        """No brain → returns error."""
        handler = self._make_handler()
        # Override: get_brain returns None
        storage = await handler.get_storage()
        storage.get_brain = AsyncMock(return_value=None)
        result = await handler._train_db_schema(
            {"connection_string": "sqlite:///test.db"}
        )
        assert "error" in result
        assert "No brain" in result["error"]


# ── Enrichment path tests ─────────────────────────────────────


class TestEnrichmentPath:
    """Tests for consolidation / enrichment path."""

    @pytest.mark.asyncio
    async def test_enrichment_called_when_consolidate_true(self) -> None:
        storage = _build_mock_storage()
        with patch.object(DBTrainer, "__init__", lambda self, *a, **kw: None):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = _build_mock_config()
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()
            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            trainer._extractor.extract = MagicMock(return_value=_make_knowledge())
            trainer._encoder.encode = AsyncMock(return_value=_mock_encode_result())
            trainer._run_enrichment = AsyncMock(return_value=5)

            tc = DBTrainingConfig(connection_string="sqlite:///test.db", consolidate=True)
            result = await trainer.train(tc)
            trainer._run_enrichment.assert_called_once()
            assert result.enrichment_synapses == 5

    @pytest.mark.asyncio
    async def test_enrichment_failure_returns_partial_result(self) -> None:
        storage = _build_mock_storage()
        with patch.object(DBTrainer, "__init__", lambda self, *a, **kw: None):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = _build_mock_config()
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()
            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            trainer._extractor.extract = MagicMock(return_value=_make_knowledge())
            trainer._encoder.encode = AsyncMock(return_value=_mock_encode_result())
            trainer._run_enrichment = AsyncMock(side_effect=RuntimeError("enrich failed"))

            tc = DBTrainingConfig(connection_string="sqlite:///test.db", consolidate=True)
            result = await trainer.train(tc)
            # Should succeed with partial result (enrichment_synapses=0)
            assert result.tables_processed == 2
            assert result.enrichment_synapses == 0


# ── Handler happy-path tests ──────────────────────────────────


class TestHandlerHappyPath:
    """Tests for successful training and status operations."""

    def _make_handler(self):
        from neural_memory.mcp.db_train_handler import DBTrainHandler

        handler = MagicMock(spec=DBTrainHandler)
        handler._train_db = DBTrainHandler._train_db.__get__(handler)
        handler._train_db_schema = DBTrainHandler._train_db_schema.__get__(handler)
        handler._train_db_status = DBTrainHandler._train_db_status.__get__(handler)
        mock_brain = MagicMock()
        mock_brain.config = MagicMock()
        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "brain-1"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        handler.get_storage = AsyncMock(return_value=mock_storage)
        return handler, mock_storage

    @pytest.mark.asyncio
    async def test_successful_train_returns_result(self) -> None:
        handler, _ = self._make_handler()
        mock_result = DBTrainingResult(
            tables_processed=3,
            relationships_mapped=2,
            patterns_detected=1,
            neurons_created=10,
            synapses_created=5,
            schema_fingerprint="abc123",
        )
        with patch("neural_memory.engine.db_trainer.DBTrainer") as mock_trainer_cls:
            mock_instance = AsyncMock()
            mock_instance.train = AsyncMock(return_value=mock_result)
            mock_trainer_cls.return_value = mock_instance
            result = await handler._train_db_schema({"connection_string": "sqlite:///test.db"})
        assert result["tables_processed"] == 3
        assert result["relationships_mapped"] == 2
        assert "message" in result

    @pytest.mark.asyncio
    async def test_status_returns_count(self) -> None:
        handler, mock_storage = self._make_handler()
        mock_neuron = MagicMock()
        mock_neuron.metadata = {"db_schema": True}
        mock_storage.find_neurons = AsyncMock(return_value=[mock_neuron, mock_neuron])
        result = await handler._train_db_status()
        assert result["trained_tables"] == 2
        assert result["has_training_data"] is True

    @pytest.mark.asyncio
    async def test_status_empty_brain(self) -> None:
        handler, mock_storage = self._make_handler()
        mock_storage.find_neurons = AsyncMock(return_value=[])
        result = await handler._train_db_status()
        assert result["trained_tables"] == 0
        assert result["has_training_data"] is False


# ── Handler error-path tests ──────────────────────────────────


class TestHandlerErrorPaths:
    """Tests for ValueError and generic Exception handling."""

    def _make_handler(self):
        from neural_memory.mcp.db_train_handler import DBTrainHandler

        handler = MagicMock(spec=DBTrainHandler)
        handler._train_db = DBTrainHandler._train_db.__get__(handler)
        handler._train_db_schema = DBTrainHandler._train_db_schema.__get__(handler)
        mock_brain = MagicMock()
        mock_brain.config = MagicMock()
        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "brain-1"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        handler.get_storage = AsyncMock(return_value=mock_storage)
        return handler

    @pytest.mark.asyncio
    async def test_value_error_returns_generic_message(self) -> None:
        handler = self._make_handler()
        with patch("neural_memory.engine.db_trainer.DBTrainer") as mock_trainer_cls:
            mock_instance = AsyncMock()
            mock_instance.train = AsyncMock(side_effect=ValueError("secret path /etc/shadow"))
            mock_trainer_cls.return_value = mock_instance
            result = await handler._train_db_schema({"connection_string": "sqlite:///test.db"})
        assert "error" in result
        assert "invalid configuration" in result["error"]
        assert "secret" not in result["error"]  # no info leakage

    @pytest.mark.asyncio
    async def test_generic_exception_returns_generic_message(self) -> None:
        handler = self._make_handler()
        with patch("neural_memory.engine.db_trainer.DBTrainer") as mock_trainer_cls:
            mock_instance = AsyncMock()
            mock_instance.train = AsyncMock(side_effect=RuntimeError("internal error details"))
            mock_trainer_cls.return_value = mock_instance
            result = await handler._train_db_schema({"connection_string": "sqlite:///test.db"})
        assert "error" in result
        assert "unexpectedly" in result["error"]
        assert "internal" not in result["error"]


# ── Dispatch routing tests ────────────────────────────────────


class TestDispatchRouting:
    """Tests for _train_db dispatcher routing."""

    def _make_handler(self):
        from neural_memory.mcp.db_train_handler import DBTrainHandler

        handler = MagicMock(spec=DBTrainHandler)
        handler._train_db = DBTrainHandler._train_db.__get__(handler)
        handler._train_db_schema = AsyncMock(return_value={"ok": True})
        handler._train_db_status = AsyncMock(return_value={"status": "ok"})
        return handler

    @pytest.mark.asyncio
    async def test_default_action_is_train(self) -> None:
        handler = self._make_handler()
        await handler._train_db({})
        handler._train_db_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_action_dispatches(self) -> None:
        handler = self._make_handler()
        await handler._train_db({"action": "train"})
        handler._train_db_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_status_action_dispatches(self) -> None:
        handler = self._make_handler()
        await handler._train_db({"action": "status"})
        handler._train_db_status.assert_called_once()


# ── Handler new validations tests ─────────────────────────────


class TestHandlerNewValidations:
    """Tests for consolidate and max_tables upper bound validations."""

    def _make_handler(self):
        from neural_memory.mcp.db_train_handler import DBTrainHandler

        handler = MagicMock(spec=DBTrainHandler)
        handler._train_db_schema = DBTrainHandler._train_db_schema.__get__(handler)
        mock_brain = MagicMock()
        mock_brain.config = MagicMock()
        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "brain-1"
        mock_storage.get_brain = AsyncMock(return_value=mock_brain)
        handler.get_storage = AsyncMock(return_value=mock_storage)
        return handler

    @pytest.mark.asyncio
    async def test_max_tables_over_500_rejected(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db_schema(
            {"connection_string": "sqlite:///test.db", "max_tables": 501}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_consolidate_non_boolean_rejected(self) -> None:
        handler = self._make_handler()
        result = await handler._train_db_schema(
            {"connection_string": "sqlite:///test.db", "consolidate": "yes"}
        )
        assert "error" in result
        assert "boolean" in result["error"]


# ── Pattern skip for unencoded tables ─────────────────────────


class TestPatternSkipForUnencoded:
    """Patterns are skipped when their table was not encoded."""

    @pytest.mark.asyncio
    async def test_pattern_skipped_if_table_not_encoded(self) -> None:
        storage = _build_mock_storage()
        with patch.object(DBTrainer, "__init__", lambda self, *a, **kw: None):
            trainer = DBTrainer.__new__(DBTrainer)
            trainer._storage = storage
            trainer._config = _build_mock_config()
            trainer._encoder = MagicMock()
            trainer._introspector = MagicMock()
            trainer._extractor = MagicMock()
            trainer._introspector.introspect = AsyncMock(return_value=_make_snapshot())
            # Knowledge has pattern for table "unknown" which is not in entities
            knowledge = SchemaKnowledge(
                entities=(),
                relationships=(),
                patterns=(
                    KnowledgePattern(
                        SchemaPatternType.AUDIT_TRAIL, "unknown", {}, "desc", 0.7
                    ),
                ),
                properties=(),
            )
            trainer._extractor.extract = MagicMock(return_value=knowledge)
            tc = DBTrainingConfig(connection_string="sqlite:///test.db", consolidate=False)
            result = await trainer.train(tc)
            assert result.patterns_detected == 0
