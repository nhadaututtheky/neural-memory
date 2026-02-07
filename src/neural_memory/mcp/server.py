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
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory import __version__
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory, suggest_memory_type
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.mcp.auto_handler import AutoHandler
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH
from neural_memory.mcp.eternal_handler import EternalHandler
from neural_memory.mcp.index_handler import IndexHandler
from neural_memory.mcp.prompt import get_system_prompt
from neural_memory.mcp.session_handler import SessionHandler
from neural_memory.mcp.tool_schemas import get_tool_schemas
from neural_memory.unified_config import get_config, get_shared_storage

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class MCPServer(SessionHandler, EternalHandler, AutoHandler, IndexHandler):
    """MCP server that exposes NeuralMemory tools.

    Uses shared SQLite storage for cross-tool memory sharing.
    Configuration from ~/.neuralmemory/config.toml

    Handler mixins:
        SessionHandler  — _session, _get_active_session
        EternalHandler  — _eternal, _recap, _fire_eternal_trigger
        AutoHandler     — _auto, _passive_capture, _save_detected_memories
        IndexHandler    — _index, _import
    """

    def __init__(self) -> None:
        self.config: UnifiedConfig = get_config()
        self._storage: SQLiteStorage | None = None
        self._eternal_ctx = None

    async def get_storage(self) -> SQLiteStorage:
        """Get or create shared SQLite storage instance."""
        if self._storage is None:
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
        }
        handler = dispatch.get(name)
        if handler:
            return await handler(arguments)
        return {"error": f"Unknown tool: {name}"}

    # ──────────────────── Core tool handlers ────────────────────

    async def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store a memory in the neural graph."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        content = args["content"]
        if len(content) > MAX_CONTENT_LENGTH:
            return {"error": f"Content too long ({len(content)} chars). Max: {MAX_CONTENT_LENGTH}."}

        # Check for sensitive content (high severity only)
        from neural_memory.safety.sensitive import check_sensitive_content

        sensitive_matches = check_sensitive_content(content, min_severity=2)
        if sensitive_matches:
            types_found = sorted({m.type.value for m in sensitive_matches})
            return {
                "error": "Sensitive content detected",
                "sensitive_types": types_found,
                "message": "Content contains potentially sensitive information. "
                "Remove secrets before storing.",
            }

        # Determine memory type
        if "type" in args:
            try:
                mem_type = MemoryType(args["type"])
            except ValueError:
                return {"error": f"Invalid memory type: {args['type']}"}
        else:
            mem_type = suggest_memory_type(content)

        priority = Priority.from_int(args.get("priority", 5))

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        tags = set(args.get("tags", []))
        result = await encoder.encode(
            content=content, timestamp=datetime.now(), tags=tags if tags else None
        )

        typed_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=mem_type,
            priority=priority,
            source="mcp_tool",
            expires_in_days=args.get("expires_days"),
            tags=tags if tags else None,
        )
        await storage.add_typed_memory(typed_mem)
        await storage.batch_save()

        self._fire_eternal_trigger(content)

        return {
            "success": True,
            "fiber_id": result.fiber.id,
            "memory_type": mem_type.value,
            "neurons_created": len(result.neurons_created),
            "message": f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
        }

    async def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query memories via spreading activation."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        query = args["query"]
        try:
            depth = DepthLevel(args.get("depth", 1))
        except ValueError:
            return {"error": f"Invalid depth level: {args.get('depth')}. Must be 0-3."}
        max_tokens = args.get("max_tokens", 500)
        min_confidence = args.get("min_confidence", 0.0)

        # Inject session context for richer recall on vague queries
        effective_query = query
        try:
            session = await self._get_active_session(storage)
            if session and isinstance(session, dict):
                session_terms: list[str] = []
                feature = session.get("feature", "")
                task = session.get("task", "")
                if isinstance(feature, str) and feature:
                    session_terms.append(feature)
                if isinstance(task, str) and task:
                    session_terms.append(task)
                if session_terms and len(query.split()) < 8:
                    effective_query = f"{query} [context: {', '.join(session_terms)}]"
        except Exception:
            logger.debug("Session context injection failed", exc_info=True)

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=effective_query, depth=depth, max_tokens=max_tokens, reference_time=datetime.now()
        )

        # Passive auto-capture on long queries
        if self.config.auto.enabled and len(query) >= 50:
            await self._passive_capture(query)

        self._fire_eternal_trigger(query)

        if result.confidence < min_confidence:
            return {
                "answer": None,
                "message": f"No memories found with confidence >= {min_confidence}",
                "confidence": result.confidence,
            }

        return {
            "answer": result.context or "No relevant memories found.",
            "confidence": result.confidence,
            "neurons_activated": result.neurons_activated,
            "fibers_matched": result.fibers_matched,
            "depth_used": result.depth_used.value,
            "tokens_used": result.tokens_used,
        }

    async def _context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get recent context."""
        storage = await self.get_storage()

        limit = args.get("limit", 10)
        fresh_only = args.get("fresh_only", False)

        fibers = await storage.get_fibers(limit=limit * 2 if fresh_only else limit)
        if not fibers:
            return {"context": "No memories stored yet.", "count": 0}

        if fresh_only:
            from neural_memory.safety.freshness import FreshnessLevel, evaluate_freshness

            now = datetime.now()
            fresh_fibers = [
                f
                for f in fibers
                if evaluate_freshness(f.created_at, now).level
                in (FreshnessLevel.FRESH, FreshnessLevel.RECENT)
            ]
            fibers = fresh_fibers[:limit]

        context_parts = []
        for fiber in fibers:
            content = fiber.summary
            if not content and fiber.anchor_neuron_id:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    content = anchor.content
            if content:
                context_parts.append(f"- {content}")

        context_text = "\n".join(context_parts) if context_parts else "No context available."
        return {
            "context": context_text,
            "count": len(context_parts),
            "tokens_used": len(context_text.split()),
        }

    async def _todo(self, args: dict[str, Any]) -> dict[str, Any]:
        """Add a TODO."""
        return await self._remember(
            {
                "content": args["task"],
                "type": "todo",
                "priority": args.get("priority", 5),
                "expires_days": 30,
            }
        )

    async def _stats(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get brain statistics."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        stats = await storage.get_enhanced_stats(brain.id)
        return {
            "brain": brain.name,
            "neuron_count": stats["neuron_count"],
            "synapse_count": stats["synapse_count"],
            "fiber_count": stats["fiber_count"],
            "db_size_bytes": stats.get("db_size_bytes", 0),
            "today_fibers_count": stats.get("today_fibers_count", 0),
            "hot_neurons": stats.get("hot_neurons", []),
            "newest_memory": stats.get("newest_memory"),
        }

    async def _suggest(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get prefix-based autocomplete suggestions."""
        storage = await self.get_storage()
        prefix = args.get("prefix", "")
        if not prefix.strip():
            return {"suggestions": [], "count": 0}

        limit = args.get("limit", 5)
        type_filter = None
        if "type_filter" in args:
            from neural_memory.core.neuron import NeuronType

            try:
                type_filter = NeuronType(args["type_filter"])
            except ValueError:
                return {"error": f"Invalid type_filter: {args['type_filter']}"}

        suggestions = await storage.suggest_neurons(
            prefix=prefix, type_filter=type_filter, limit=limit
        )
        formatted = [
            {
                "content": s["content"],
                "type": s["type"],
                "neuron_id": s["neuron_id"],
                "score": s["score"],
            }
            for s in suggestions
        ]
        return {
            "suggestions": formatted,
            "count": len(formatted),
            "tokens_used": sum(len(s["content"].split()) for s in formatted),
        }


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
        tool_args = params.get("arguments", {})

        try:
            result = await server.call_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except Exception as e:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": str(e)}}

    elif method == "notifications/initialized":
        return None  # type: ignore

    else:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


async def run_mcp_server() -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server()

    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
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


def main() -> None:
    """Entry point for the MCP server."""
    asyncio.run(run_mcp_server())
