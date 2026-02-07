"""Index and import handlers for MCP server."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class IndexHandler:
    """Mixin: codebase indexing + external import tool handlers."""

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

        brain = await storage.get_brain(storage._current_brain_id)
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
        results = await encoder.index_directory(path, extensions=extensions)
        await storage.batch_save()

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
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        source = args.get("source", "")
        if not source:
            return {"error": "Source system name required"}

        adapter_kwargs = _build_adapter_kwargs(source, args)

        try:
            adapter = get_adapter(source, **adapter_kwargs)
        except ValueError as e:
            return {"error": str(e)}

        engine = SyncEngine(storage, brain.config)
        storage.disable_auto_save()

        try:
            result, _sync_state = await engine.sync(
                adapter=adapter,
                collection=args.get("collection"),
                limit=args.get("limit"),
            )
            await storage.batch_save()
        except Exception as e:
            logger.warning("Import from %s failed: %s", source, e)
            return {"error": f"Import failed: {e}"}

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


def _build_adapter_kwargs(source: str, args: dict[str, Any]) -> dict[str, Any]:
    """Build adapter-specific kwargs from tool args."""
    kwargs: dict[str, Any] = {}
    connection = args.get("connection")

    if source == "chromadb" and connection:
        kwargs["path"] = connection
    elif source == "mem0":
        if connection:
            kwargs["api_key"] = connection
        if args.get("user_id"):
            kwargs["user_id"] = args["user_id"]
    elif source == "awf" and connection:
        kwargs["brain_dir"] = connection
    elif source == "cognee" and connection:
        kwargs["api_key"] = connection
    elif source == "graphiti":
        if connection:
            kwargs["uri"] = connection
        if args.get("group_id"):
            kwargs["group_id"] = args["group_id"]
    elif source == "llamaindex" and connection:
        kwargs["persist_dir"] = connection

    return kwargs
