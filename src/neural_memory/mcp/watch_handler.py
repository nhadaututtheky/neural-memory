"""MCP handler mixin for file watcher — nmem_watch tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class WatchHandler:
    """Mixin providing nmem_watch tool handler for MCPServer."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    _file_watcher: Any = None

    async def _watch(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_watch tool calls.

        Actions:
        - scan: One-shot scan and ingest a directory
        - start: Start background file watching
        - stop: Stop background file watching
        - status: Show watcher status and stats
        - list: List tracked files
        """
        action = args.get("action", "status")
        storage = await self.get_storage()
        brain_id = _require_brain_id(storage)
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        try:
            if action == "scan":
                return await self._watch_scan(storage, args)
            elif action == "start":
                return await self._watch_start(storage, args)
            elif action == "stop":
                return self._watch_stop()
            elif action == "status":
                return await self._watch_status(storage)
            elif action == "list":
                return await self._watch_list(storage, args)
            else:
                return {
                    "error": f"Unknown watch action: {action}. Use: scan, start, stop, status, list"
                }
        except ImportError:
            return {
                "error": "watchdog not installed. Run: pip install neural-memory[watch]",
            }
        except Exception:
            logger.error("Watch handler failed for action '%s'", action, exc_info=True)
            return {"error": "File watcher operation failed"}

    async def _watch_scan(
        self,
        storage: Any,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """One-shot scan and ingest a directory."""
        directory = args.get("directory", "")
        if not directory:
            return {"error": "directory parameter is required for scan action"}

        path = Path(directory).resolve()

        watcher = await self._get_or_create_watcher(storage)
        error = watcher.validate_path(path)
        if error:
            return {"error": error}

        results = await watcher.process_path(path)

        processed = [r for r in results if r.success and not r.skipped]
        skipped = [r for r in results if r.skipped]
        failed = [r for r in results if not r.success]

        return {
            "action": "scan",
            "directory": str(path),
            "processed": len(processed),
            "skipped": len(skipped),
            "failed": len(failed),
            "total_neurons": sum(r.neurons_created for r in processed),
            "details": [
                {
                    "path": r.path,
                    "neurons": r.neurons_created,
                    "status": "skipped" if r.skipped else ("ok" if r.success else "error"),
                    **({"error": r.error} if r.error else {}),
                }
                for r in results[:20]  # Cap details
            ],
        }

    async def _watch_start(
        self,
        storage: Any,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Start background file watching."""
        directories = args.get("directories", [])
        if not directories:
            return {"error": "directories parameter is required (list of paths to watch)"}

        if len(directories) > 10:
            return {"error": "Maximum 10 directories allowed"}

        # Validate all paths first
        valid_paths: list[str] = []
        for d in directories:
            path = Path(d).resolve()

            watcher = await self._get_or_create_watcher(storage)
            error = watcher.validate_path(path)
            if error:
                return {"error": f"Invalid directory: {error}"}
            valid_paths.append(str(path))

        # Create watcher with paths
        from neural_memory.engine.file_watcher import WatchConfig

        config = WatchConfig(watch_paths=tuple(valid_paths))
        watcher = await self._get_or_create_watcher(storage, config)

        try:
            watcher.start()
        except ImportError:
            return {"error": "watchdog not installed. Run: pip install neural-memory[watch]"}

        return {
            "action": "start",
            "watching": valid_paths,
            "status": "running",
        }

    def _watch_stop(self) -> dict[str, Any]:
        """Stop background file watching."""
        if self._file_watcher and self._file_watcher.is_running:
            self._file_watcher.stop()
            return {"action": "stop", "status": "stopped"}
        return {"action": "stop", "status": "not_running"}

    async def _watch_status(self, storage: Any) -> dict[str, Any]:
        """Show watcher status and statistics."""
        watcher = await self._get_or_create_watcher(storage)
        stats = await watcher._state.get_stats()

        return {
            "action": "status",
            "running": watcher.is_running,
            **stats,
            "recent_results": [
                {
                    "path": r.path,
                    "success": r.success,
                    "neurons": r.neurons_created,
                    "skipped": r.skipped,
                }
                for r in watcher.get_recent_results(10)
            ],
        }

    async def _watch_list(
        self,
        storage: Any,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """List tracked files."""
        status_filter = args.get("status")
        limit = min(args.get("limit", 50), 200)

        watcher = await self._get_or_create_watcher(storage)
        files = await watcher._state.list_watched_files(status=status_filter)

        return {
            "action": "list",
            "total": len(files),
            "files": [
                {
                    "path": f.file_path,
                    "neurons": f.neuron_count,
                    "last_ingested": f.last_ingested,
                    "status": f.status,
                }
                for f in files[:limit]
            ],
        }

    async def _get_or_create_watcher(
        self,
        storage: Any,
        config: Any = None,
    ) -> Any:
        """Get or create the FileWatcher instance."""
        if self._file_watcher and config is None:
            return self._file_watcher

        from neural_memory.engine.doc_trainer import DocTrainer
        from neural_memory.engine.file_watcher import FileWatcher, WatchConfig
        from neural_memory.engine.watch_state import WatchStateTracker

        db = getattr(storage, "_db", None)
        if db is None or not hasattr(db, "execute"):
            raise RuntimeError("File watcher requires SQLite storage backend")
        state_tracker = WatchStateTracker(db)
        brain_config = await storage.get_brain_config()
        trainer = DocTrainer(storage, brain_config)

        watcher_config = config or WatchConfig()
        self._file_watcher = FileWatcher(trainer, state_tracker, watcher_config)

        return self._file_watcher
