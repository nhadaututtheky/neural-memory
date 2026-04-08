"""MCP server implementation for NeuralMemory.

Exposes NeuralMemory as tools via Model Context Protocol (MCP),
allowing Claude Code, Cursor, AntiGravity and other MCP clients to
store and recall memories.

All tools share the same SQLite database at ~/.neuralmemory/brains/<brain>.db
This enables seamless memory sharing between different AI tools.

Usage:
    # Run directly
    python -m neural_memory.mcp

    # Or add to Claude Code via CLI:
    claude mcp add --scope user neural-memory -- nmem-mcp

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
from neural_memory.mcp.alert_handler import AlertHandler
from neural_memory.mcp.auto_handler import AutoHandler
from neural_memory.mcp.cognitive_handler import CognitiveHandler
from neural_memory.mcp.conflict_handler import ConflictHandler
from neural_memory.mcp.connection_handler import ConnectionHandler
from neural_memory.mcp.db_train_handler import DBTrainHandler
from neural_memory.mcp.drift_handler import DriftHandler
from neural_memory.mcp.goal_handler import GoalHandler
from neural_memory.mcp.eternal_handler import EternalHandler
from neural_memory.mcp.expiry_cleanup_handler import ExpiryCleanupHandler
from neural_memory.mcp.index_handler import IndexHandler
from neural_memory.mcp.maintenance_handler import MaintenanceHandler
from neural_memory.mcp.mem0_sync_handler import Mem0SyncHandler
from neural_memory.mcp.milestone_handler import MilestoneHandler
from neural_memory.mcp.narrative_handler import NarrativeHandler
from neural_memory.mcp.onboarding_handler import OnboardingHandler
from neural_memory.mcp.prompt import get_mcp_instructions, get_system_prompt
from neural_memory.mcp.review_handler import ReviewHandler
from neural_memory.mcp.scheduled_consolidation_handler import ScheduledConsolidationHandler
from neural_memory.mcp.session_handler import SessionHandler
from neural_memory.mcp.surface_handler import SurfaceHandler
from neural_memory.mcp.sync_handler import SyncToolHandler
from neural_memory.mcp.telegram_handler import TelegramHandler
from neural_memory.mcp.tool_handlers import ToolHandler
from neural_memory.mcp.tool_schemas import get_tool_schemas_for_tier
from neural_memory.mcp.train_handler import TrainHandler
from neural_memory.mcp.version_check_handler import VersionCheckHandler
from neural_memory.mcp.visualize_handler import VisualizeHandler
from neural_memory.mcp.watch_handler import WatchHandler
from neural_memory.unified_config import get_config, get_shared_storage

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
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
    AlertHandler,
    ReviewHandler,
    NarrativeHandler,
    VisualizeHandler,
    WatchHandler,
    ConnectionHandler,
    CognitiveHandler,
    Mem0SyncHandler,
    OnboardingHandler,
    ExpiryCleanupHandler,
    ScheduledConsolidationHandler,
    VersionCheckHandler,
    SurfaceHandler,
    SyncToolHandler,
    TelegramHandler,
    DriftHandler,
    MilestoneHandler,
    GoalHandler,
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
        AlertHandler        — _alerts, persistent alert lifecycle
        ReviewHandler       — _review, spaced repetition queue/mark/schedule/stats
        NarrativeHandler    — _narrative, timeline/topic/causal narratives
        ConnectionHandler   — _explain, shortest-path connection explanation
        CognitiveHandler    — _hypothesize, _evidence, _predict, _verify, _cognitive, _gaps, _schema
        Mem0SyncHandler     — maybe_start_mem0_sync, background auto-sync
        OnboardingHandler   — _check_onboarding, fresh-brain guidance
        ExpiryCleanupHandler — _maybe_run_expiry_cleanup, auto-delete expired
        ScheduledConsolidationHandler — periodic background consolidation
        VersionCheckHandler  — background PyPI version check + update hints
        SyncToolHandler      — _sync, _sync_status, _sync_config (multi-device sync)
        TelegramHandler      — _telegram_backup (send brain to Telegram)
        SurfaceHandler       — _surface (knowledge surface generate/show)
        DriftHandler         — _drift (semantic drift detection + resolution)
        MilestoneHandler     — _milestone (growth milestones + reports)
    """

    def __init__(self) -> None:
        self.config: UnifiedConfig = get_config()
        self._storage: NeuralStorage | None = None
        self._eternal_ctx = None
        self.hooks: HookRegistry = HookRegistry()
        self._surface_text: str = ""
        self._surface_brain: str = ""
        self._agent_id: str = ""

    async def get_storage(self) -> NeuralStorage:
        """Get or create shared storage instance.

        Re-reads ``current_brain`` from disk on each call so that
        brain switches made by the CLI are picked up without
        restarting the MCP server.
        """
        # get_shared_storage() handles brain-change detection internally
        # and returns the correct (possibly cached) storage instance.
        self._storage = await get_shared_storage()

        # Reload surface if brain changed
        current_brain = self.config.current_brain or "default"
        if current_brain != self._surface_brain:
            self.load_surface(current_brain)

        return self._storage

    def load_surface(self, brain_name: str = "") -> str:
        """Load the Knowledge Surface for the current brain.

        Caches the result — only re-reads from disk when brain changes.

        Args:
            brain_name: Brain to load surface for. If empty, uses current brain.

        Returns:
            Surface text (empty string if not found).
        """
        if not brain_name:
            brain_name = self.config.current_brain or "default"

        # Cache hit: same brain, already loaded
        if brain_name == self._surface_brain and self._surface_text:
            return self._surface_text

        from neural_memory.surface.resolver import load_surface_text

        text = load_surface_text(brain_name)
        self._surface_text = text or ""
        self._surface_brain = brain_name
        return self._surface_text

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
        """Return list of available MCP tools, filtered by tier + plugin tools."""
        from neural_memory.plugins import get_plugin_tools

        tier = self.config.tool_tier.tier
        tools = get_tool_schemas_for_tier(tier)
        tools.extend(get_plugin_tools())
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate handler."""
        dispatch = {
            "nmem_remember": self._remember,
            "nmem_remember_batch": self._remember_batch,
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
            "nmem_pin": self._pin,
            "nmem_alerts": self._alerts,
            "nmem_review": self._review,
            "nmem_narrative": self._narrative,
            "nmem_visualize": self._visualize,
            "nmem_watch": self._watch,
            "nmem_sync": self._sync,
            "nmem_sync_status": self._sync_status,
            "nmem_sync_config": self._sync_config,
            "nmem_telegram_backup": self._telegram_backup,
            "nmem_explain": self._explain,
            "nmem_hypothesize": self._hypothesize,
            "nmem_evidence": self._evidence,
            "nmem_predict": self._predict,
            "nmem_verify": self._verify,
            "nmem_cognitive": self._cognitive,
            "nmem_gaps": self._gaps,
            "nmem_schema": self._schema,
            "nmem_show": self._show,
            "nmem_source": self._source,
            "nmem_provenance": self._provenance,
            "nmem_edit": self._edit,
            "nmem_forget": self._forget,
            "nmem_consolidate": self._consolidate,
            "nmem_drift": self._drift,
            "nmem_surface": self._surface,
            "nmem_tool_stats": self._tool_stats,
            "nmem_lifecycle": self._lifecycle,
            "nmem_refine": self._refine,
            "nmem_report_outcome": self._report_outcome,
            "nmem_budget": self._budget,
            "nmem_tier": self._tier,
            "nmem_boundaries": self._boundaries,
            "nmem_milestone": self._milestone,
            "nmem_store": self._store,
            "nmem_goal": self._goal,
        }
        handler = dispatch.get(name)
        if handler:
            return await handler(arguments)

        # Check plugin-provided tools
        from neural_memory.plugins import get_plugin_tool_handler

        plugin_handler = get_plugin_tool_handler(name)
        if plugin_handler:
            return await plugin_handler(self, arguments)  # type: ignore[no-any-return]

        return {"error": f"Unknown tool: {name}"}


# ──────────────────── Tool event recording ────────────────────

# Tools that should NOT be recorded (meta/analytics tools → avoid recursion)
_SKIP_EVENT_TOOLS = frozenset({"nmem_tool_stats", "nmem_version", "nmem_stats"})


async def _record_tool_event(
    server: MCPServer,
    tool_name: str,
    tool_args: dict[str, Any],
    start_time: Any,
    *,
    success: bool,
) -> None:
    """Record a tool call event for analytics (fire-and-forget)."""
    if tool_name in _SKIP_EVENT_TOOLS:
        return

    from neural_memory.utils.timeutils import utcnow

    duration_ms = int((utcnow() - start_time).total_seconds() * 1000)

    # Build a short args summary (first 200 chars of key params)
    summary_parts: list[str] = []
    for key in ("query", "content", "action", "strategy", "topic"):
        val = tool_args.get(key)
        if val is not None:
            summary_parts.append(f"{key}={str(val)[:60]}")
    args_summary = ", ".join(summary_parts)[:200]

    try:
        storage = await server.get_storage()
        brain_name = server.config.current_brain
        brain = await storage.get_brain(brain_name)
        if not brain:
            return

        await storage.insert_tool_events(  # type: ignore[attr-defined]
            brain.id,
            [
                {
                    "tool_name": tool_name,
                    "server_name": "neural-memory",
                    "args_summary": args_summary,
                    "success": success,
                    "duration_ms": duration_ms,
                    "session_id": "",
                    "task_context": "",
                    "created_at": utcnow().isoformat(),
                }
            ],
        )
    except Exception:
        logger.debug("Failed to insert tool event for %s", tool_name, exc_info=True)


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
        # Capture agent identity from MCP clientInfo
        client_info = params.get("clientInfo", {})
        server._agent_id = str(client_info.get("name", "")) or ""

        instructions = get_mcp_instructions()

        # Inject Knowledge Surface if available
        surface_text = server.load_surface()
        if surface_text:
            instructions = f"{instructions}\n\n## Knowledge Surface\n{surface_text}"

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "neural-memory", "version": __version__},
                "capabilities": {"tools": {}, "resources": {}},
                "instructions": instructions,
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
        raw_args = params.get("arguments", {})
        # Some MCP clients (e.g. OpenClaw) pass arguments as a JSON string
        # instead of a parsed dict. Parse it gracefully.
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                raw_args = {"content": raw_args}
        tool_args = _sanitize_surrogates(raw_args)

        # Check compact mode before passing args to tool (pops 'compact' key)
        from neural_memory.mcp.response_compactor import (
            apply_token_budget,
            compact_response,
            needs_auto_compact,
            should_compact,
        )

        use_compact = should_compact(tool_args=tool_args, config=server.config.response)
        token_budget = tool_args.pop("token_budget", None)

        from neural_memory.utils.timeutils import utcnow

        t0 = utcnow()
        success = True
        try:
            result = await asyncio.wait_for(
                server.call_tool(tool_name, tool_args),
                timeout=_TOOL_CALL_TIMEOUT,
            )

            # Apply compact mode: explicit, global config, or auto-detect
            resp_config = server.config.response
            if isinstance(result, dict):
                if use_compact:
                    result = compact_response(result, resp_config)
                elif needs_auto_compact(result, resp_config.auto_compact_threshold):
                    logger.debug("Auto-compact: %s exceeded threshold", tool_name)
                    result = compact_response(result, resp_config)

                # Apply token budget if requested
                if token_budget is not None:
                    result = apply_token_budget(result, int(token_budget))

                # Check if tool returned an error
                if result.get("error"):
                    success = False

            result_text = json.dumps(result)

            # Post-tool passive capture (fire-and-forget, never blocks response)
            try:
                await server._post_tool_capture(tool_name, tool_args, result_text)
            except Exception:
                logger.debug("Post-tool passive capture failed", exc_info=True)

            # Record tool event for analytics (fire-and-forget)
            try:
                await _record_tool_event(server, tool_name, tool_args, t0, success=success)
            except Exception:
                logger.debug("Tool event recording failed", exc_info=True)

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": result_text}]},
            }
        except TimeoutError:
            try:
                await _record_tool_event(server, tool_name, tool_args, t0, success=False)
            except Exception:
                logger.debug("Tool event recording failed", exc_info=True)
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
            try:
                await _record_tool_event(server, tool_name, tool_args, t0, success=False)
            except Exception:
                logger.debug("Tool event recording failed", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32000, "message": f"Tool '{tool_name}' failed unexpectedly"},
            }

    elif method == "notifications/initialized" or (method and method.startswith("notifications/")):
        return None  # type: ignore[return-value]

    else:
        if msg_id is None:
            return None  # type: ignore[return-value]
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


_TOOL_CALL_TIMEOUT = 30.0  # seconds
_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def _lazy_init() -> None:
    """Run first-time setup if NeuralMemory has never been initialized.

    Safe to call on every MCP start — no-ops if config already exists.
    Only touches config/brain/hooks; never writes to stdout (reserved for JSON-RPC).
    """
    from neural_memory.unified_config import get_neuralmemory_dir

    data_dir = get_neuralmemory_dir()
    config_path = data_dir / "config.toml"
    if config_path.exists():
        return  # Already initialized — fast path, no heavy imports

    # Only import cli.setup when first-time init is actually needed
    from neural_memory.cli.setup import setup_brain, setup_config, setup_hooks_claude

    try:
        setup_config(data_dir)
        setup_brain(data_dir)
        hook_status = setup_hooks_claude()
        logger.info("NeuralMemory: first-time auto-init complete (hook: %s)", hook_status)
    except Exception:
        logger.debug("NeuralMemory: auto-init failed (non-critical)", exc_info=True)


async def run_mcp_server() -> None:
    """Run the MCP server over stdio."""
    _lazy_init()

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

    # Start background version check if configured
    try:
        await server.maybe_start_version_check()
    except Exception:
        logger.debug("Version check startup failed (non-critical)", exc_info=True)

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
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                print(json.dumps(error_resp), flush=True)
                continue
            except EOFError:
                break
            except KeyboardInterrupt:
                break
    finally:
        # Run session-end consolidation before shutdown (MATURE + INFER + ENRICH)
        try:
            await server.run_session_end_consolidation()
        except Exception:
            logger.debug("Session-end consolidation skipped", exc_info=True)

        # Cancel background tasks
        server.cancel_mem0_sync()
        server.cancel_expiry_cleanup()
        server.cancel_scheduled_consolidation()
        server.cancel_version_check()

        # Close aiosqlite connection before event loop exits to prevent
        # "Event loop is closed" noise from the background thread.
        if server._storage is not None:
            await server._storage.close()


def main() -> None:
    """Entry point for the MCP server.

    Supports two transports:
        nmem-mcp              → stdio (default, 1 process per client)
        nmem-mcp --http       → HTTP (single shared server, multi-client)
        nmem-mcp --http 9000  → HTTP on custom port
    """
    import argparse

    parser = argparse.ArgumentParser(description="NeuralMemory MCP server")
    parser.add_argument(
        "--http",
        nargs="?",
        const=8765,
        type=int,
        metavar="PORT",
        help="Run HTTP transport on PORT (default: 8765) instead of stdio",
    )
    args = parser.parse_args()

    if args.http is not None:
        from neural_memory.mcp.http_transport import run_http_server

        asyncio.run(run_http_server(port=args.http))
    else:
        asyncio.run(run_mcp_server())
