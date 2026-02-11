"""Database schema training handler for MCP server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage

logger = logging.getLogger(__name__)

# Validation limits
_MAX_CONNECTION_STRING_LEN = 500
_MAX_DOMAIN_TAG_LEN = 100
_MAX_BRAIN_NAME_LEN = 64


class DBTrainHandler:
    """Mixin: database schema training tool handlers.

    Exposes nmem_train_db with actions: train, status.
    Validates inputs and delegates to DBTrainer.
    """

    async def get_storage(self) -> SQLiteStorage:
        raise NotImplementedError

    async def _train_db(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle database schema training actions."""
        action = args.get("action", "train")

        if action == "train":
            return await self._train_db_schema(args)
        if action == "status":
            return await self._train_db_status()
        return {"error": f"Unknown train_db action: {action}"}

    async def _train_db_schema(self, args: dict[str, Any]) -> dict[str, Any]:
        """Train a brain from database schema knowledge."""
        from neural_memory.engine.db_trainer import DBTrainer, DBTrainingConfig

        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id or "")
        if not brain:
            return {"error": "No brain configured"}

        # Validate connection_string
        connection_string = args.get("connection_string", "")
        if not connection_string:
            return {"error": "connection_string is required"}
        if len(connection_string) > _MAX_CONNECTION_STRING_LEN:
            return {"error": f"connection_string too long (max {_MAX_CONNECTION_STRING_LEN} chars)"}
        # v1: SQLite only
        if not connection_string.lower().startswith("sqlite:///"):
            return {"error": "Only SQLite is supported in v1. Use: sqlite:///path/to/db"}

        # Validate domain_tag
        domain_tag = args.get("domain_tag", "")
        if len(domain_tag) > _MAX_DOMAIN_TAG_LEN:
            return {"error": f"domain_tag too long (max {_MAX_DOMAIN_TAG_LEN} chars)"}

        # Validate brain_name
        brain_name = args.get("brain_name", "")
        if len(brain_name) > _MAX_BRAIN_NAME_LEN:
            return {"error": f"brain_name too long (max {_MAX_BRAIN_NAME_LEN} chars)"}

        # Validate max_tables
        max_tables = args.get("max_tables", 100)
        if not isinstance(max_tables, int) or max_tables < 1 or max_tables > 500:
            return {"error": "max_tables must be an integer between 1 and 500"}

        # Validate consolidate
        consolidate = args.get("consolidate", True)
        if not isinstance(consolidate, bool):
            return {"error": "consolidate must be a boolean"}

        # Build training config
        tc = DBTrainingConfig(
            connection_string=connection_string,
            domain_tag=domain_tag,
            brain_name=brain_name,
            consolidate=consolidate,
            max_tables=max_tables,
        )

        try:
            trainer = DBTrainer(storage, brain.config)
            result = await trainer.train(tc)
        except ValueError as exc:
            logger.error("DB training validation error: %s", exc)
            return {"error": "Training failed: invalid configuration"}
        except Exception:
            logger.error("DB training failed", exc_info=True)
            return {"error": "Database training failed unexpectedly"}

        return {
            "tables_processed": result.tables_processed,
            "tables_skipped": result.tables_skipped,
            "columns_processed": result.columns_processed,
            "relationships_mapped": result.relationships_mapped,
            "patterns_detected": result.patterns_detected,
            "neurons_created": result.neurons_created,
            "synapses_created": result.synapses_created,
            "enrichment_synapses": result.enrichment_synapses,
            "schema_fingerprint": result.schema_fingerprint,
            "brain_name": result.brain_name,
            "message": (
                f"Trained schema: {result.tables_processed} tables, "
                f"{result.relationships_mapped} relationships, "
                f"{result.patterns_detected} patterns detected"
            ),
        }

    async def _train_db_status(self) -> dict[str, Any]:
        """Show database schema training status."""
        storage = await self.get_storage()

        from neural_memory.core.neuron import NeuronType

        # Count schema-trained neurons
        schema_neurons = await storage.find_neurons(
            type=NeuronType.CONCEPT,
            limit=1000,
        )
        trained_count = sum(1 for n in schema_neurons if n.metadata.get("db_schema"))

        return {
            "trained_tables": trained_count,
            "has_training_data": trained_count > 0,
            "message": (
                f"Brain has {trained_count} trained schema entities"
                if trained_count > 0
                else "No schema training data. Use nmem_train_db(action='train', connection_string='sqlite:///path') to train."
            ),
        }
