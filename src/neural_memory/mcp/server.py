"""MCP server implementation for NeuralMemory.

Exposes NeuralMemory as tools via Model Context Protocol (MCP),
allowing Claude Code, Cursor, AntiGravity and other MCP clients to
store and recall memories.

All tools share the same SQLite database at ~/.neuralmemory/brains/<brain>.db
This enables seamless memory sharing between different AI tools.

Usage:
    # Run directly
    python -m neural_memory.mcp

    # Or in Claude Code's mcp_servers.json:
    {
        "neural-memory": {
            "command": "python",
            "args": ["-m", "neural_memory.mcp"]
        }
    }

    # Or set NEURALMEMORY_BRAIN to use a specific brain:
    NEURALMEMORY_BRAIN=myproject python -m neural_memory.mcp
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import TYPE_CHECKING, Any

from neural_memory import __version__
from neural_memory.engine.hooks import HookRegistry
from neural_memory.mcp.auto_handler import AutoHandler
from neural_memory.mcp.conflict_handler import ConflictHandler
from neural_memory.mcp.db_train_handler import DBTrainHandler
from neural_memory.mcp.eternal_handler import EternalHandler
from neural_memory.mcp.expiry_cleanup_handler import ExpiryCleanupHandler
from neural_memory.mcp.index_handler import IndexHandler
from neural_memory.mcp.maintenance_handler import MaintenanceHandler
from neural_memory.mcp.mem0_sync_handler import Mem0SyncHandler
from neural_memory.mcp.onboarding_handler import OnboardingHandler
from neural_memory.mcp.prompt import get_system_prompt
from neural_memory.mcp.scheduled_consolidation_handler import ScheduledConsolidationHandler
from neural_memory.mcp.session_handler import SessionHandler
from neural_memory.mcp.tool_handlers import ToolHandler
from neural_memory.mcp.tool_schemas import get_tool_schemas
from neural_memory.mcp.train_handler import TrainHandler
from neural_memory.unified_config import get_config, get_shared_storage

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


def _sanitize_surrogates(obj: Any) -> Any:
    """Remove lone surrogate characters from strings in tool arguments.

    On Windows, stdio pipes can introduce surrogate characters (U+D800-U+DFFF)
    that cause UnicodeEncodeError when passed to UTF-8 encoders or SQLite.
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_surrogates(item) for item in obj]
    return obj


class MCPServer(
    ToolHandler,
    SessionHandler,
    EternalHandler,
    AutoHandler,
    IndexHandler,
    ConflictHandler,
    TrainHandler,
    DBTrainHandler,
    MaintenanceHandler,
    Mem0SyncHandler,
    OnboardingHandler,
    ExpiryCleanupHandler,
    ScheduledConsolidationHandler,
):
    """MCP server that exposes NeuralMemory tools.

    Uses shared SQLite storage for cross-tool memory sharing.
    Configuration from ~/.neuralmemory/config.toml

    Handler mixins:
        SessionHandler      — _session, _get_active_session
        EternalHandler      — _eternal, _recap, _fire_eternal_trigger
        AutoHandler         — _auto, _passive_capture, _save_detected_memories
        IndexHandler        — _index, _import
        ConflictHandler     — _conflicts (list, resolve, check)
        TrainHandler        — _train (train docs into brain, status)
        DBTrainHandler      — _train_db (train DB schema into brain, status)
        MaintenanceHandler  — _check_maintenance, health pulse
        Mem0SyncHandler     — maybe_start_mem0_sync, background auto-sync
        OnboardingHandler   — _check_onboarding, fresh-brain guidance
        ExpiryCleanupHandler — _maybe_run_expiry_cleanup, auto-delete expired
        ScheduledConsolidationHandler — periodic background consolidation
    """

    def __init__(self) -> None:
        self.config: UnifiedConfig = get_config()
        self._storage: SQLiteStorage | None = None
        self._eternal_ctx = None
        self.hooks: HookRegistry = HookRegistry()

    async def get_storage(self) -> SQLiteStorage:
        """Get or create shared SQLite storage instance.

        Re-reads ``current_brain`` from disk on each call so that
        brain switches made by the CLI are picked up without
        restarting the MCP server.
        """
        # get_shared_storage() handles brain-change detection internally
        # and returns the correct (possibly cached) storage instance.
        self._storage = await get_shared_storage()
        return self._storage

    def get_resources(self) -> list[dict[str, Any]]:
        """Return list of available MCP resources."""
        return [
            {
                "uri": "neuralmemory://prompt/system",
                "name": "NeuralMemory System Prompt",
                "description": "Instructions for AI on when/how to use NeuralMemory",
                "mimeType": "text/plain",
            },
            {
                "uri": "neuralmemory://prompt/compact",
                "name": "NeuralMemory Compact Prompt",
                "description": "Short version of system prompt for limited context",
                "mimeType": "text/plain",
            },
        ]

    def get_resource_content(self, uri: str) -> str | None:
        """Get content for a specific resource URI."""
        if uri == "neuralmemory://prompt/system":
            return get_system_prompt(compact=False)
        elif uri == "neuralmemory://prompt/compact":
            return get_system_prompt(compact=True)
        return None

    def get_tools(self) -> list[dict[str, Any]]:
        """Return list of available MCP tools."""
        return get_tool_schemas()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate handler."""
        dispatch = {
            "nmem_remember": self._remember,
            "nmem_recall": self._recall,
            "nmem_context": self._context,
            "nmem_todo": self._todo,
            "nmem_stats": self._stats,
            "nmem_auto": self._auto,
            "nmem_suggest": self._suggest,
            "nmem_session": self._session,
            "nmem_index": self._index,
            "nmem_import": self._import,
            "nmem_eternal": self._eternal,
            "nmem_recap": self._recap,
            "nmem_health": self._health,
            "nmem_evolution": self._evolution,
            "nmem_habits": self._habits,
            "nmem_version": self._version,
            "nmem_transplant": self._transplant,
            "nmem_conflicts": self._conflicts,
            "nmem_train": self._train,
            "nmem_train_db": self._train_db,
        }
        handler = dispatch.get(name)
        if handler:
            return await handler(arguments)
        return {"error": f"Unknown tool: {name}"}


# ──────────────────── Module-level functions ────────────────────


def create_mcp_server() -> MCPServer:
    """Create an MCP server instance."""
    return MCPServer()


async def handle_message(server: MCPServer, message: dict[str, Any]) -> dict[str, Any]:
    """Handle a single MCP message."""
    method = message.get("method", "")
    msg_id = message.get("id")
    params = message.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "neural-memory", "version": __version__},
                "capabilities": {"tools": {}, "resources": {}},
            },
        }

    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": server.get_tools()}}

    elif method == "resources/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": server.get_resources()}}

    elif method == "resources/read":
        uri = params.get("uri", "")
        content = server.get_resource_content(uri)
        if content is None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32002, "message": f"Resource not found: {uri}"},
            }
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"contents": [{"uri": uri, "mimeType": "text/plain", "text": content}]},
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = _sanitize_surrogates(params.get("arguments", {}))

        try:
            result = await asyncio.wait_for(
                server.call_tool(tool_name, tool_args),
                timeout=_TOOL_CALL_TIMEOUT,
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result)}]},
            }
        except TimeoutError:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32000,
                    "message": f"Tool '{tool_name}' timed out after {_TOOL_CALL_TIMEOUT}s",
                },
            }
        except Exception:
            logger.error("Tool '%s' raised an exception", tool_name, exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32000, "message": f"Tool '{tool_name}' failed unexpectedly"},
            }

    elif method == "notifications/initialized":
        return None  # type: ignore[return-value]

    else:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


_TOOL_CALL_TIMEOUT = 30.0  # seconds
_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


async def run_mcp_server() -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server()

    # Start background Mem0 auto-sync if configured
    try:
        await server.maybe_start_mem0_sync()
    except Exception:
        logger.debug("Mem0 auto-sync startup failed (non-critical)", exc_info=True)

    # Start scheduled consolidation loop if configured
    try:
        await server.maybe_start_scheduled_consolidation()
    except Exception:
        logger.debug("Scheduled consolidation startup failed (non-critical)", exc_info=True)

    try:
        while True:
            try:
                line = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                if len(line) > _MAX_MESSAGE_SIZE:
                    error_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32000, "message": "Message too large"},
                    }
                    print(json.dumps(error_resp), flush=True)
                    continue

                message = json.loads(line)
                response = await handle_message(server, message)

                if response is not None:
                    print(json.dumps(response), flush=True)

            except json.JSONDecodeError:
                continue
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    finally:
        # Cancel background tasks
        server.cancel_mem0_sync()
        server.cancel_expiry_cleanup()
        server.cancel_scheduled_consolidation()

        # Close aiosqlite connection before event loop exits to prevent
        # "Event loop is closed" noise from the background thread.
        if server._storage is not None:
            await server._storage.close()


def main() -> None:
    """Entry point for the MCP server."""
    asyncio.run(run_mcp_server())
