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
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory import __version__
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory, suggest_memory_type
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.mcp.auto_capture import analyze_text_for_memories
from neural_memory.mcp.prompt import get_system_prompt
from neural_memory.mcp.tool_schemas import get_tool_schemas
from neural_memory.unified_config import get_config, get_shared_storage

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage
    from neural_memory.unified_config import UnifiedConfig


class MCPServer:
    """MCP server that exposes NeuralMemory tools.

    Uses shared SQLite storage for cross-tool memory sharing.
    Configuration from ~/.neuralmemory/config.toml
    """

    def __init__(self) -> None:
        self.config: UnifiedConfig = get_config()
        self._storage: SQLiteStorage | None = None

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
        """Get content of a resource by URI."""
        if uri == "neuralmemory://prompt/system":
            return get_system_prompt(compact=False)
        elif uri == "neuralmemory://prompt/compact":
            return get_system_prompt(compact=True)
        return None

    def get_tools(self) -> list[dict[str, Any]]:
        """Return list of available MCP tools."""
        return get_tool_schemas()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP tool call."""
        if name == "nmem_remember":
            return await self._remember(arguments)
        elif name == "nmem_recall":
            return await self._recall(arguments)
        elif name == "nmem_context":
            return await self._context(arguments)
        elif name == "nmem_todo":
            return await self._todo(arguments)
        elif name == "nmem_stats":
            return await self._stats(arguments)
        elif name == "nmem_auto":
            return await self._auto(arguments)
        elif name == "nmem_suggest":
            return await self._suggest(arguments)
        elif name == "nmem_session":
            return await self._session(arguments)
        else:
            return {"error": f"Unknown tool: {name}"}

    async def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store a memory."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        content = args["content"]

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
            mem_type = MemoryType(args["type"])
        else:
            mem_type = suggest_memory_type(content)

        # Determine priority
        priority = Priority.from_int(args.get("priority", 5))

        # Encode memory
        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        tags = set(args.get("tags", []))
        result = await encoder.encode(
            content=content,
            timestamp=datetime.now(),
            tags=tags if tags else None,
        )

        # Create typed memory
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

        return {
            "success": True,
            "fiber_id": result.fiber.id,
            "memory_type": mem_type.value,
            "neurons_created": len(result.neurons_created),
            "message": f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
        }

    async def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query memories."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        query = args["query"]
        depth = DepthLevel(args.get("depth", 1))
        max_tokens = args.get("max_tokens", 500)
        min_confidence = args.get("min_confidence", 0.0)

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=datetime.now(),
        )

        # Passive auto-capture: analyze query text for capturable content
        if self.config.auto.enabled and len(query) >= 50:
            await self._passive_capture(query)

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

    async def _passive_capture(self, text: str) -> None:
        """Silently analyze text and capture high-confidence memories.

        Fire-and-forget: errors are swallowed to avoid disrupting
        the primary tool call.
        """
        try:
            detected = analyze_text_for_memories(
                text,
                capture_decisions=self.config.auto.capture_decisions,
                capture_errors=self.config.auto.capture_errors,
                capture_todos=self.config.auto.capture_todos,
                capture_facts=self.config.auto.capture_facts,
                capture_insights=self.config.auto.capture_insights,
            )
            if detected:
                # Higher threshold for passive capture (user didn't ask to save)
                passive_threshold = max(self.config.auto.min_confidence, 0.8)
                high_confidence = [
                    item for item in detected if item["confidence"] >= passive_threshold
                ]
                if high_confidence:
                    await self._save_detected_memories(high_confidence)
        except Exception:
            pass

    async def _context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get recent context."""
        storage = await self.get_storage()

        limit = args.get("limit", 10)
        fresh_only = args.get("fresh_only", False)

        fibers = await storage.get_fibers(limit=limit * 2 if fresh_only else limit)

        if not fibers:
            return {"context": "No memories stored yet.", "count": 0}

        # Filter by freshness if requested
        if fresh_only:
            from neural_memory.safety.freshness import FreshnessLevel, evaluate_freshness

            now = datetime.now()
            fresh_fibers = []
            for fiber in fibers:
                freshness = evaluate_freshness(fiber.created_at, now)
                if freshness.level in (FreshnessLevel.FRESH, FreshnessLevel.RECENT):
                    fresh_fibers.append(fiber)
            fibers = fresh_fibers[:limit]

        # Build context
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
            "synapse_stats": stats.get("synapse_stats", {}),
            "neuron_type_breakdown": stats.get("neuron_type_breakdown", {}),
            "oldest_memory": stats.get("oldest_memory"),
            "newest_memory": stats.get("newest_memory"),
        }

    async def _auto(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle auto-capture settings and analysis."""
        action = args.get("action", "status")

        if action == "status":
            return {
                "enabled": self.config.auto.enabled,
                "capture_decisions": self.config.auto.capture_decisions,
                "capture_errors": self.config.auto.capture_errors,
                "capture_todos": self.config.auto.capture_todos,
                "capture_facts": self.config.auto.capture_facts,
                "capture_insights": self.config.auto.capture_insights,
                "min_confidence": self.config.auto.min_confidence,
            }

        elif action == "enable":
            from dataclasses import replace

            self.config = replace(self.config, auto=replace(self.config.auto, enabled=True))
            self.config.save()
            return {"enabled": True, "message": "Auto-capture enabled"}

        elif action == "disable":
            from dataclasses import replace

            self.config = replace(self.config, auto=replace(self.config.auto, enabled=False))
            self.config.save()
            return {"enabled": False, "message": "Auto-capture disabled"}

        elif action == "analyze":
            text = args.get("text", "")
            if not text:
                return {"error": "Text required for analyze action"}

            detected = analyze_text_for_memories(
                text,
                capture_decisions=self.config.auto.capture_decisions,
                capture_errors=self.config.auto.capture_errors,
                capture_todos=self.config.auto.capture_todos,
                capture_facts=self.config.auto.capture_facts,
                capture_insights=self.config.auto.capture_insights,
            )

            if not detected:
                return {"detected": [], "message": "No memorable content detected"}

            should_save = args.get("save", False)
            if should_save:
                saved = await self._save_detected_memories(detected)
                return {
                    "detected": detected,
                    "saved": saved,
                    "message": f"Analyzed and saved {len(saved)} memories",
                }

            return {
                "detected": detected,
                "message": f"Detected {len(detected)} potential memories (not saved)",
            }

        elif action == "process":
            if not self.config.auto.enabled:
                return {
                    "saved": 0,
                    "message": "Auto-capture is disabled. Use nmem_auto(action='enable') to enable.",
                }

            text = args.get("text", "")
            if not text:
                return {"error": "Text required for process action"}

            detected = analyze_text_for_memories(
                text,
                capture_decisions=self.config.auto.capture_decisions,
                capture_errors=self.config.auto.capture_errors,
                capture_todos=self.config.auto.capture_todos,
                capture_facts=self.config.auto.capture_facts,
                capture_insights=self.config.auto.capture_insights,
            )

            if not detected:
                return {"saved": 0, "message": "No memorable content detected"}

            saved = await self._save_detected_memories(detected)

            return {
                "saved": len(saved),
                "memories": saved,
                "message": f"Auto-captured {len(saved)} memories"
                if saved
                else "No memories met confidence threshold",
            }

        return {"error": f"Unknown action: {action}"}

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

            type_filter = NeuronType(args["type_filter"])

        suggestions = await storage.suggest_neurons(
            prefix=prefix,
            type_filter=type_filter,
            limit=limit,
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

    async def _session(self, args: dict[str, Any]) -> dict[str, Any]:
        """Track current working session state."""
        action = args.get("action", "get")
        storage = await self.get_storage()

        async def _find_current_session() -> TypedMemory | None:
            sessions = await storage.find_typed_memories(
                memory_type=MemoryType.CONTEXT,
                tags={"session_state"},
                limit=1,
            )
            return sessions[0] if sessions else None

        if action == "get":
            session = await _find_current_session()
            if not session:
                return {"active": False, "message": "No active session"}

            return {
                "active": True,
                "feature": session.metadata.get("feature", ""),
                "task": session.metadata.get("task", ""),
                "progress": session.metadata.get("progress", 0.0),
                "started_at": session.metadata.get("started_at", ""),
                "notes": session.metadata.get("notes", ""),
            }

        elif action == "set":
            # Build metadata from args
            now = datetime.now()
            existing = await _find_current_session()

            metadata: dict[str, Any] = {
                "feature": args.get(
                    "feature", existing.metadata.get("feature", "") if existing else ""
                ),
                "task": args.get("task", existing.metadata.get("task", "") if existing else ""),
                "progress": args.get(
                    "progress", existing.metadata.get("progress", 0.0) if existing else 0.0
                ),
                "notes": args.get("notes", existing.metadata.get("notes", "") if existing else ""),
                "started_at": existing.metadata.get("started_at", now.isoformat())
                if existing
                else now.isoformat(),
                "updated_at": now.isoformat(),
            }

            # Build content summary
            content = f"Session: {metadata['feature']}"
            if metadata["task"]:
                content += f" — {metadata['task']}"
            if metadata["progress"]:
                content += f" ({int(metadata['progress'] * 100)}%)"

            # Encode as a new CONTEXT memory (immutable pattern)
            brain = await storage.get_brain(storage._current_brain_id)
            if not brain:
                return {"error": "No brain configured"}

            encoder = MemoryEncoder(storage, brain.config)
            storage.disable_auto_save()

            result = await encoder.encode(
                content=content,
                timestamp=now,
                tags={"session_state"},
            )

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=MemoryType.CONTEXT,
                priority=Priority.from_int(7),
                source="mcp_session",
                expires_in_days=1,
                tags={"session_state"},
                metadata=metadata,
            )
            await storage.add_typed_memory(typed_mem)
            await storage.batch_save()

            return {
                "active": True,
                "feature": metadata["feature"],
                "task": metadata["task"],
                "progress": metadata["progress"],
                "started_at": metadata["started_at"],
                "notes": metadata["notes"],
                "message": "Session state updated",
            }

        elif action == "end":
            existing = await _find_current_session()
            if not existing:
                return {"active": False, "message": "No active session to end"}

            feature = existing.metadata.get("feature", "unknown")
            task = existing.metadata.get("task", "")
            progress = existing.metadata.get("progress", 0.0)

            # Create a summary memory
            summary = f"Session ended: worked on {feature}"
            if task:
                summary += f", task: {task}"
            summary += f", progress: {int(progress * 100)}%"

            brain = await storage.get_brain(storage._current_brain_id)
            if not brain:
                return {"error": "No brain configured"}

            encoder = MemoryEncoder(storage, brain.config)
            storage.disable_auto_save()

            result = await encoder.encode(
                content=summary,
                timestamp=datetime.now(),
                tags={"session_summary"},
            )

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=MemoryType.CONTEXT,
                priority=Priority.from_int(5),
                source="mcp_session",
                expires_in_days=7,
                tags={"session_summary"},
            )
            await storage.add_typed_memory(typed_mem)
            await storage.batch_save()

            return {
                "active": False,
                "summary": summary,
                "message": "Session ended and summary saved",
            }

        return {"error": f"Unknown session action: {action}"}

    async def _save_detected_memories(self, detected: list[dict[str, Any]]) -> list[str]:
        """Save detected memories that meet confidence threshold."""
        saved = []
        for item in detected:
            if item["confidence"] >= self.config.auto.min_confidence:
                result = await self._remember(
                    {
                        "content": item["content"],
                        "type": item["type"],
                        "priority": item.get("priority", 5),
                    }
                )
                if "error" not in result:
                    saved.append(item["content"][:50])
        return saved


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
        # No response needed for notifications
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

    # Read from stdin, write to stdout
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
    """Main entry point for MCP server."""
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()
