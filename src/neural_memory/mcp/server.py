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

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory, suggest_memory_type
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.mcp.prompt import get_prompt_for_mcp, get_system_prompt
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
        return [
            {
                "name": "nmem_remember",
                "description": "Store a memory in NeuralMemory. Use this to remember facts, decisions, insights, todos, errors, and other information that should persist across sessions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The content to remember"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "fact",
                                "decision",
                                "preference",
                                "todo",
                                "insight",
                                "context",
                                "instruction",
                                "error",
                                "workflow",
                                "reference",
                            ],
                            "description": "Memory type (auto-detected if not specified)",
                        },
                        "priority": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                            "description": "Priority 0-10 (5=normal, 10=critical)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization",
                        },
                        "expires_days": {
                            "type": "integer",
                            "description": "Days until memory expires",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "nmem_recall",
                "description": "Query memories from NeuralMemory. Use this to recall past information, decisions, patterns, or context relevant to the current task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to search memories"},
                        "depth": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 3,
                            "description": "Search depth: 0=instant, 1=context, 2=habit, 3=deep",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens in response (default: 500)",
                        },
                        "min_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Minimum confidence threshold",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "nmem_context",
                "description": "Get recent context from NeuralMemory. Use this at the start of tasks to inject relevant recent memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of recent memories (default: 10)",
                        },
                        "fresh_only": {
                            "type": "boolean",
                            "description": "Only include memories < 30 days old",
                        },
                    },
                },
            },
            {
                "name": "nmem_todo",
                "description": "Quick shortcut to add a TODO memory with 30-day expiry.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "The task to remember"},
                        "priority": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                            "description": "Priority 0-10 (default: 5)",
                        },
                    },
                    "required": ["task"],
                },
            },
            {
                "name": "nmem_stats",
                "description": "Get brain statistics including memory counts and freshness.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "nmem_auto",
                "description": "Control auto-capture settings and analyze text for auto-save. When enabled, NeuralMemory automatically detects and saves decisions, errors, todos, and important facts from conversations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "enable", "disable", "analyze"],
                            "description": "Action to perform",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to analyze for auto-capture (required for 'analyze' action)",
                        },
                        "save": {
                            "type": "boolean",
                            "description": "If true, save detected memories (default: false for analyze)",
                        },
                    },
                    "required": ["action"],
                },
            },
        ]

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
        else:
            return {"error": f"Unknown tool: {name}"}

    async def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store a memory."""
        storage = await self.get_storage()
        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        content = args["content"]

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
        }

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

        return {
            "context": "\n".join(context_parts) if context_parts else "No context available.",
            "count": len(context_parts),
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

        stats = await storage.get_stats(brain.id)

        return {
            "brain": brain.name,
            "neuron_count": stats["neuron_count"],
            "synapse_count": stats["synapse_count"],
            "fiber_count": stats["fiber_count"],
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
                "min_confidence": self.config.auto.min_confidence,
            }

        elif action == "enable":
            self.config.auto.enabled = True
            self.config.save()
            return {"enabled": True, "message": "Auto-capture enabled"}

        elif action == "disable":
            self.config.auto.enabled = False
            self.config.save()
            return {"enabled": False, "message": "Auto-capture disabled"}

        elif action == "analyze":
            text = args.get("text", "")
            if not text:
                return {"error": "Text required for analyze action"}

            # Analyze text for potential memories
            detected = self._analyze_text_for_memories(text)

            if not detected:
                return {"detected": [], "message": "No memorable content detected"}

            # Optionally save detected memories
            if args.get("save", False) and self.config.auto.enabled:
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
                return {
                    "detected": detected,
                    "saved": saved,
                    "message": f"Analyzed and saved {len(saved)} memories",
                }

            return {
                "detected": detected,
                "message": f"Detected {len(detected)} potential memories (not saved)",
            }

        return {"error": f"Unknown action: {action}"}

    def _analyze_text_for_memories(self, text: str) -> list[dict[str, Any]]:
        """Analyze text and detect potential memories.

        Returns list of detected memories with type, content, and confidence.
        """
        import re

        detected: list[dict[str, Any]] = []
        text_lower = text.lower()

        # Decision patterns
        decision_patterns = [
            r"(?:we |I )(?:decided|chose|selected|picked|opted)(?: to)?[:\s]+(.+?)(?:\.|$)",
            r"(?:the )?decision(?: is)?[:\s]+(.+?)(?:\.|$)",
            r"(?:we\'re |I\'m )going (?:to|with)[:\s]+(.+?)(?:\.|$)",
            r"let\'s (?:go with|use|choose)[:\s]+(.+?)(?:\.|$)",
        ]
        if self.config.auto.capture_decisions:
            for pattern in decision_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if len(match) > 10:
                        detected.append(
                            {
                                "type": "decision",
                                "content": f"Decision: {match.strip()}",
                                "confidence": 0.8,
                                "priority": 6,
                            }
                        )

        # Error patterns
        error_patterns = [
            r"error[:\s]+(.+?)(?:\.|$)",
            r"failed[:\s]+(.+?)(?:\.|$)",
            r"bug[:\s]+(.+?)(?:\.|$)",
            r"(?:the )?issue (?:is|was)[:\s]+(.+?)(?:\.|$)",
            r"problem[:\s]+(.+?)(?:\.|$)",
        ]
        if self.config.auto.capture_errors:
            for pattern in error_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if len(match) > 10:
                        detected.append(
                            {
                                "type": "error",
                                "content": f"Error: {match.strip()}",
                                "confidence": 0.85,
                                "priority": 7,
                            }
                        )

        # TODO patterns
        todo_patterns = [
            r"(?:TODO|FIXME|HACK|XXX)[:\s]+(.+?)(?:\.|$)",
            r"(?:we |I )?(?:need to|should|must|have to)[:\s]+(.+?)(?:\.|$)",
            r"(?:remember to|don\'t forget to)[:\s]+(.+?)(?:\.|$)",
            r"(?:later|next)[:\s]+(.+?)(?:\.|$)",
        ]
        if self.config.auto.capture_todos:
            for pattern in todo_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match) > 5:
                        detected.append(
                            {
                                "type": "todo",
                                "content": f"TODO: {match.strip()}",
                                "confidence": 0.75,
                                "priority": 5,
                            }
                        )

        # Fact patterns (more conservative)
        fact_patterns = [
            r"(?:the |a )?(?:answer|solution|fix) (?:is|was)[:\s]+(.+?)(?:\.|$)",
            r"(?:it |this )(?:works|worked) because[:\s]+(.+?)(?:\.|$)",
            r"(?:the )?(?:key|important|note)[:\s]+(.+?)(?:\.|$)",
            r"(?:learned|discovered|found out)[:\s]+(.+?)(?:\.|$)",
        ]
        if self.config.auto.capture_facts:
            for pattern in fact_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if len(match) > 15:
                        detected.append(
                            {
                                "type": "fact",
                                "content": match.strip(),
                                "confidence": 0.7,
                                "priority": 5,
                            }
                        )

        # Remove duplicates
        seen = set()
        unique_detected = []
        for item in detected:
            content_key = item["content"][:50].lower()
            if content_key not in seen:
                seen.add(content_key)
                unique_detected.append(item)

        return unique_detected


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
                "serverInfo": {"name": "neural-memory", "version": "0.1.0"},
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
