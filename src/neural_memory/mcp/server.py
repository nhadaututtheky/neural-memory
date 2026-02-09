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
from neural_memory.core.memory_types import (
    MemoryType,
    Priority,
    TypedMemory,
    get_decay_rate,
    suggest_memory_type,
)
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
from neural_memory.utils.timeutils import utcnow

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
            "nmem_health": self._health,
            "nmem_habits": self._habits,
            "nmem_version": self._version,
            "nmem_transplant": self._transplant,
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

        try:
            tags = set(args.get("tags", []))
            result = await encoder.encode(
                content=content, timestamp=utcnow(), tags=tags if tags else None
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

            # Set type-specific decay rate on neuron states
            type_decay_rate = get_decay_rate(mem_type.value)
            for neuron in result.neurons_created:
                state = await storage.get_neuron_state(neuron.id)
                if state and state.decay_rate != type_decay_rate:
                    from neural_memory.core.neuron import NeuronState

                    updated_state = NeuronState(
                        neuron_id=state.neuron_id,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                        decay_rate=type_decay_rate,
                        created_at=state.created_at,
                    )
                    await storage.update_neuron_state(updated_state)

            await storage.batch_save()
        finally:
            storage.enable_auto_save()

        self._fire_eternal_trigger(content)

        await self._record_tool_action("remember", content[:100])

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
        max_tokens = min(args.get("max_tokens", 500), 10_000)
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

        # Parse optional temporal filter
        valid_at = None
        if "valid_at" in args:
            try:
                valid_at = datetime.fromisoformat(args["valid_at"])
            except (ValueError, TypeError):
                return {"error": f"Invalid valid_at datetime: {args['valid_at']}"}

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=effective_query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=utcnow(),
            valid_at=valid_at,
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

        response: dict[str, Any] = {
            "answer": result.context or "No relevant memories found.",
            "confidence": result.confidence,
            "neurons_activated": result.neurons_activated,
            "fibers_matched": result.fibers_matched,
            "depth_used": result.depth_used.value,
            "tokens_used": result.tokens_used,
        }

        if result.score_breakdown is not None:
            response["score_breakdown"] = {
                "base_activation": round(result.score_breakdown.base_activation, 4),
                "intersection_boost": round(result.score_breakdown.intersection_boost, 4),
                "freshness_boost": round(result.score_breakdown.freshness_boost, 4),
                "frequency_boost": round(result.score_breakdown.frequency_boost, 4),
            }

        await self._record_tool_action("recall", query[:100])

        return response

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

            now = utcnow()
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

        await self._record_tool_action("context")

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

    async def _health(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run brain health diagnostics."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        from neural_memory.engine.diagnostics import DiagnosticsEngine

        engine = DiagnosticsEngine(storage)
        report = await engine.analyze(brain.id)

        return {
            "brain": brain.name,
            "grade": report.grade,
            "purity_score": report.purity_score,
            "connectivity": report.connectivity,
            "diversity": report.diversity,
            "freshness": report.freshness,
            "consolidation_ratio": report.consolidation_ratio,
            "orphan_rate": report.orphan_rate,
            "activation_efficiency": report.activation_efficiency,
            "recall_confidence": report.recall_confidence,
            "neuron_count": report.neuron_count,
            "synapse_count": report.synapse_count,
            "fiber_count": report.fiber_count,
            "warnings": [
                {"severity": w.severity.value, "code": w.code, "message": w.message}
                for w in report.warnings
            ],
            "recommendations": list(report.recommendations),
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

    async def _habits(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage learned workflow habits."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        action = args.get("action", "list")

        if action == "suggest":
            current_action = args.get("current_action", "")
            if not current_action:
                return {"error": "current_action is required for suggest"}

            from neural_memory.engine.workflow_suggest import suggest_next_action

            suggestions = await suggest_next_action(storage, current_action, brain.config)
            return {
                "suggestions": [
                    {
                        "action": s.action_type,
                        "confidence": round(s.confidence, 4),
                        "source_habit": s.source_habit,
                        "sequential_count": s.sequential_count,
                    }
                    for s in suggestions
                ],
                "count": len(suggestions),
            }

        elif action == "list":
            fibers = await storage.get_fibers(limit=10000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern")]
            return {
                "habits": [
                    {
                        "name": h.summary or "unnamed",
                        "steps": h.metadata.get("_workflow_actions", []),
                        "frequency": h.metadata.get("_habit_frequency", 0),
                        "confidence": h.metadata.get("_habit_confidence", 0.0),
                        "fiber_id": h.id,
                    }
                    for h in habits
                ],
                "count": len(habits),
            }

        elif action == "clear":
            fibers = await storage.get_fibers(limit=10000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern")]
            cleared = 0
            for h in habits:
                await storage.delete_fiber(h.id)
                cleared += 1
            return {"cleared": cleared, "message": f"Cleared {cleared} learned habits"}

        return {"error": f"Unknown action: {action}"}

    async def _version(self, args: dict[str, Any]) -> dict[str, Any]:
        """Brain version control operations."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        from neural_memory.engine.brain_versioning import VersioningEngine

        engine = VersioningEngine(storage)
        action = args.get("action", "list")

        if action == "create":
            name = args.get("name")
            if not name:
                return {"error": "Version name is required for create"}
            description = args.get("description", "")
            try:
                version = await engine.create_version(brain.id, name, description)
            except ValueError as e:
                return {"error": str(e)}
            return {
                "success": True,
                "version_id": version.id,
                "version_name": version.version_name,
                "version_number": version.version_number,
                "neuron_count": version.neuron_count,
                "synapse_count": version.synapse_count,
                "fiber_count": version.fiber_count,
                "message": f"Created version '{name}' (#{version.version_number})",
            }

        elif action == "list":
            limit = args.get("limit", 20)
            versions = await engine.list_versions(brain.id, limit=limit)
            return {
                "versions": [
                    {
                        "id": v.id,
                        "name": v.version_name,
                        "number": v.version_number,
                        "description": v.description,
                        "neuron_count": v.neuron_count,
                        "synapse_count": v.synapse_count,
                        "fiber_count": v.fiber_count,
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in versions
                ],
                "count": len(versions),
            }

        elif action == "rollback":
            version_id = args.get("version_id")
            if not version_id:
                return {"error": "version_id is required for rollback"}
            try:
                rollback_v = await engine.rollback(brain.id, version_id)
            except ValueError as e:
                return {"error": str(e)}
            return {
                "success": True,
                "rollback_version_id": rollback_v.id,
                "rollback_version_name": rollback_v.version_name,
                "neuron_count": rollback_v.neuron_count,
                "synapse_count": rollback_v.synapse_count,
                "fiber_count": rollback_v.fiber_count,
                "message": f"Rolled back to '{rollback_v.version_name}'",
            }

        elif action == "diff":
            from_id = args.get("from_version")
            to_id = args.get("to_version")
            if not from_id or not to_id:
                return {"error": "from_version and to_version are required for diff"}
            try:
                diff = await engine.diff(brain.id, from_id, to_id)
            except ValueError as e:
                return {"error": str(e)}
            return {
                "summary": diff.summary,
                "neurons_added": len(diff.neurons_added),
                "neurons_removed": len(diff.neurons_removed),
                "neurons_modified": len(diff.neurons_modified),
                "synapses_added": len(diff.synapses_added),
                "synapses_removed": len(diff.synapses_removed),
                "synapses_weight_changed": len(diff.synapses_weight_changed),
                "fibers_added": len(diff.fibers_added),
                "fibers_removed": len(diff.fibers_removed),
            }

        return {"error": f"Unknown action: {action}"}

    async def _transplant(self, args: dict[str, Any]) -> dict[str, Any]:
        """Transplant memories from another brain."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        source_brain_name = args.get("source_brain")
        if not source_brain_name:
            return {"error": "source_brain is required"}

        # Find source brain
        source_brain = await storage.find_brain_by_name(source_brain_name)
        if source_brain is None:
            return {"error": f"Source brain '{source_brain_name}' not found"}

        if source_brain.id == brain.id:
            return {
                "error": "Source brain and target brain are the same. "
                "Transplanting a brain into itself is a destructive no-op."
            }

        from neural_memory.engine.brain_transplant import TransplantFilter, transplant
        from neural_memory.engine.merge import ConflictStrategy

        tags = args.get("tags")
        memory_types = args.get("memory_types")
        strategy_str = args.get("strategy", "prefer_local")

        try:
            strategy = ConflictStrategy(strategy_str)
        except ValueError:
            return {"error": f"Invalid strategy: {strategy_str}"}

        filt = TransplantFilter(
            tags=frozenset(tags) if tags else None,
            memory_types=frozenset(memory_types) if memory_types else None,
        )

        old_brain_id = storage._current_brain_id
        try:
            storage.set_brain(source_brain.id)
            result = await transplant(
                source_storage=storage,
                target_storage=storage,
                source_brain_id=source_brain.id,
                target_brain_id=brain.id,
                filt=filt,
                strategy=strategy,
            )
        finally:
            storage.set_brain(old_brain_id)

        return {
            "success": True,
            "fibers_transplanted": result.fibers_transplanted,
            "neurons_transplanted": result.neurons_transplanted,
            "synapses_transplanted": result.synapses_transplanted,
            "merge_summary": result.merge_report.summary(),
            "message": f"Transplanted from '{source_brain_name}': {result.fibers_transplanted} fibers",
        }

    async def _record_tool_action(self, action_type: str, context: str = "") -> None:
        """Record an action event for habit learning (fire-and-forget)."""
        try:
            import os

            source = os.environ.get("NEURALMEMORY_SOURCE", "mcp")
            storage = await self.get_storage()
            await storage.record_action(
                action_type=action_type,
                action_context=context[:200] if context else "",
                session_id=f"{source}-{id(self)}",
            )
        except Exception:
            logger.debug("Action recording failed (non-critical)", exc_info=True)


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
            result = await asyncio.wait_for(
                server.call_tool(tool_name, tool_args),
                timeout=30.0,
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            }
        except TimeoutError:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32000, "message": f"Tool '{tool_name}' timed out after 30s"},
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
        # Close aiosqlite connection before event loop exits to prevent
        # "Event loop is closed" noise from the background thread.
        if server._storage is not None:
            await server._storage.close()


def main() -> None:
    """Entry point for the MCP server."""
    asyncio.run(run_mcp_server())
