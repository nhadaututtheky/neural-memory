"""Index and import handlers for MCP server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage

logger = logging.getLogger(__name__)


class IndexHandler:
    """Mixin: codebase indexing + external import tool handlers."""

    async def get_storage(self) -> SQLiteStorage:
        raise NotImplementedError

    async def _index(self, args: dict[str, Any]) -> dict[str, Any]:
        """Index codebase into neural memory."""
        action = args.get("action", "status")
        storage = await self.get_storage()

        if action == "scan":
            return await self._index_scan(args, storage)
        elif action == "status":
            return await self._index_status(storage)
        return {"error": f"Unknown index action: {action}"}

    async def _index_scan(self, args: dict[str, Any], storage: Any) -> dict[str, Any]:
        """Scan and index a directory."""
        from pathlib import Path

        from neural_memory.engine.codebase_encoder import CodebaseEncoder

        brain = await storage.get_brain(storage._current_brain_id or "")
        if not brain:
            return {"error": "No brain configured"}

        cwd = Path(".").resolve()
        path = Path(args.get("path", ".")).resolve()
        if not path.is_dir():
            return {"error": f"Not a directory: {path}"}
        if not path.is_relative_to(cwd):
            return {"error": f"Path must be within working directory: {cwd}"}

        extensions = set(args.get("extensions", [".py"]))
        encoder = CodebaseEncoder(storage, brain.config)
        storage.disable_auto_save()
        try:
            results = await encoder.index_directory(path, extensions=extensions)
            await storage.batch_save()
        finally:
            storage.enable_auto_save()

        total_neurons = sum(len(r.neurons_created) for r in results)
        total_synapses = sum(len(r.synapses_created) for r in results)

        return {
            "files_indexed": len(results),
            "neurons_created": total_neurons,
            "synapses_created": total_synapses,
            "path": str(path),
            "message": f"Indexed {len(results)} files → {total_neurons} neurons, {total_synapses} synapses",
        }

    @staticmethod
    async def _index_status(storage: Any) -> dict[str, Any]:
        """Get index status."""
        from neural_memory.core.neuron import NeuronType

        indexed_files = await storage.find_neurons(type=NeuronType.SPATIAL, limit=1000)
        code_files = [n for n in indexed_files if n.metadata.get("indexed")]

        return {
            "indexed_files": len(code_files),
            "file_list": [n.content for n in code_files[:20]],
            "message": f"{len(code_files)} files indexed"
            if code_files
            else "No codebase indexed yet. Use scan action.",
        }

    async def _import(self, args: dict[str, Any]) -> dict[str, Any]:
        """Import memories from an external source."""
        from neural_memory.integration.adapters import get_adapter
        from neural_memory.integration.sync_engine import SyncEngine

        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id or "")
        if not brain:
            return {"error": "No brain configured"}

        source = args.get("source", "")
        if not source:
            return {"error": "Source system name required"}

        adapter_kwargs = _build_adapter_kwargs(source, args)

        try:
            adapter = get_adapter(source, **adapter_kwargs)
        except ValueError:
            return {"error": f"Unsupported or misconfigured source: {source}"}

        engine = SyncEngine(storage, brain.config)
        storage.disable_auto_save()

        try:
            raw_limit = args.get("limit")
            capped_limit = min(int(raw_limit), 10000) if raw_limit is not None else None
            result, _sync_state = await engine.sync(
                adapter=adapter,
                collection=args.get("collection"),
                limit=capped_limit,
            )
            await storage.batch_save()
        except Exception:
            logger.warning("Import from %s failed", source, exc_info=True)
            return {"error": f"Import from '{source}' failed unexpectedly"}
        finally:
            storage.enable_auto_save()

        return {
            "success": True,
            "source": result.source_system,
            "collection": result.source_collection,
            "records_fetched": result.records_fetched,
            "records_imported": result.records_imported,
            "records_skipped": result.records_skipped,
            "records_failed": result.records_failed,
            "duration_seconds": result.duration_seconds,
            "errors": list(result.errors)[:5],
            "message": (
                f"Imported {result.records_imported} memories from "
                f"{result.source_system}/{result.source_collection}"
            ),
        }


def _validate_local_path(path_str: str) -> str | None:
    """Validate a local filesystem path for adapter connections.

    Returns resolved path string if valid, None if invalid.
    """
    from pathlib import Path

    resolved = Path(path_str).resolve()
    if not resolved.exists():
        return None
    return str(resolved)


def _build_adapter_kwargs(source: str, args: dict[str, Any]) -> dict[str, Any]:
    """Build adapter-specific kwargs from tool args.

    API keys are read from environment variables (never from tool parameters)
    to prevent accidental logging/exposure:
      - MEM0_API_KEY for Mem0
      - COGNEE_API_KEY for Cognee
    """
    import os

    kwargs: dict[str, Any] = {}
    connection = args.get("connection")

    if source == "chromadb" and connection:
        validated = _validate_local_path(connection)
        if validated:
            kwargs["path"] = validated
    elif source == "mem0":
        api_key = os.environ.get("MEM0_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key
        if args.get("user_id"):
            kwargs["user_id"] = args["user_id"]
    elif source == "awf" and connection:
        validated = _validate_local_path(connection)
        if validated:
            kwargs["brain_dir"] = validated
    elif source == "cognee":
        api_key = os.environ.get("COGNEE_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key
    elif source == "graphiti":
        if connection:
            kwargs["uri"] = connection
        if args.get("group_id"):
            kwargs["group_id"] = args["group_id"]
    elif source == "llamaindex" and connection:
        validated = _validate_local_path(connection)
        if validated:
            kwargs["persist_dir"] = validated

    return kwargs
