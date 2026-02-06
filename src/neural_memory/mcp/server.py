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
from neural_memory.core.brain_persistence import BrainPersistence
from neural_memory.core.eternal_context import EternalContext
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory, suggest_memory_type
from neural_memory.core.trigger_engine import check_triggers
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.git_context import detect_git_context
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
        self._eternal_ctx: EternalContext | None = None

    async def get_storage(self) -> SQLiteStorage:
        """Get or create shared SQLite storage instance."""
        if self._storage is None:
            self._storage = await get_shared_storage()
        return self._storage

    def get_eternal_context(self) -> EternalContext:
        """Get or create the eternal context manager."""
        if self._eternal_ctx is None:
            brain_id = self.config.current_brain
            persistence = BrainPersistence(brain_id)
            self._eternal_ctx = EternalContext(brain_id, persistence)
            self._eternal_ctx.load()
        return self._eternal_ctx

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
        elif name == "nmem_index":
            return await self._index(arguments)
        elif name == "nmem_import":
            return await self._import(arguments)
        elif name == "nmem_eternal":
            return await self._eternal(arguments)
        elif name == "nmem_recap":
            return await self._recap(arguments)
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

        # Fire-and-forget: check eternal context triggers
        self._fire_eternal_trigger(content)

        return {
            "success": True,
            "fiber_id": result.fiber.id,
            "memory_type": mem_type.value,
            "neurons_created": len(result.neurons_created),
            "message": f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
        }

    async def _get_active_session(self, storage: SQLiteStorage) -> dict[str, Any] | None:
        """Get active session metadata, or None if no active session."""
        try:
            sessions = await storage.find_typed_memories(
                memory_type=MemoryType.CONTEXT,
                tags={"session_state"},
                limit=1,
            )
            if sessions and sessions[0].metadata.get("active", True):
                return sessions[0].metadata
        except Exception:
            pass
        return None

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

        # Inject session context for richer recall on vague queries
        # Fire-and-forget: never let session logic break core recall
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
                # Only inject if query is vague (short, few specific terms)
                if session_terms and len(query.split()) < 8:
                    effective_query = f"{query} [context: {', '.join(session_terms)}]"
        except Exception:
            pass

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=effective_query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=datetime.now(),
        )

        # Passive auto-capture: analyze query text for capturable content
        if self.config.auto.enabled and len(query) >= 50:
            await self._passive_capture(query)

        # Fire-and-forget: track query in eternal context
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
                # Type-aware thresholds: errors/decisions are high-value
                type_thresholds = {"error": 0.7, "decision": 0.75, "insight": 0.75}
                high_confidence = [
                    item
                    for item in detected
                    if item["confidence"]
                    >= max(
                        self.config.auto.min_confidence,
                        type_thresholds.get(item["type"], 0.8),
                    )
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
            if not session or not session.metadata.get("active", True):
                return {"active": False, "message": "No active session"}

            return {
                "active": True,
                "feature": session.metadata.get("feature", ""),
                "task": session.metadata.get("task", ""),
                "progress": session.metadata.get("progress", 0.0),
                "started_at": session.metadata.get("started_at", ""),
                "notes": session.metadata.get("notes", ""),
                "branch": session.metadata.get("branch", ""),
                "commit": session.metadata.get("commit", ""),
                "repo": session.metadata.get("repo", ""),
            }

        elif action == "set":
            # Build metadata from args
            now = datetime.now()
            existing = await _find_current_session()

            # Auto-detect git context
            git_ctx = detect_git_context()

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

            # Add git context if available
            if git_ctx:
                metadata["branch"] = git_ctx.branch
                metadata["commit"] = git_ctx.commit
                metadata["repo"] = git_ctx.repo_name

            # Build content summary
            content = f"Session: {metadata['feature']}"
            if metadata["task"]:
                content += f" — {metadata['task']}"
            if metadata["progress"]:
                content += f" ({int(metadata['progress'] * 100)}%)"

            # Include branch in tags for filtering
            session_tags: set[str] = {"session_state"}
            if git_ctx:
                session_tags.add(f"branch:{git_ctx.branch}")

            # Encode as a new CONTEXT memory (immutable pattern)
            brain = await storage.get_brain(storage._current_brain_id)
            if not brain:
                return {"error": "No brain configured"}

            encoder = MemoryEncoder(storage, brain.config)
            storage.disable_auto_save()

            result = await encoder.encode(
                content=content,
                timestamp=now,
                tags=session_tags,
            )

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=MemoryType.CONTEXT,
                priority=Priority.from_int(7),
                source="mcp_session",
                expires_in_days=1,
                tags=session_tags,
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
                "branch": metadata.get("branch", ""),
                "commit": metadata.get("commit", ""),
                "repo": metadata.get("repo", ""),
                "message": "Session state updated",
            }

        elif action == "end":
            existing = await _find_current_session()
            if not existing or not existing.metadata.get("active", True):
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

            # Write tombstone session_state so GET returns inactive
            now = datetime.now()
            tombstone_result = await encoder.encode(
                content=summary,
                timestamp=now,
                tags={"session_state"},
            )
            tombstone_mem = TypedMemory.create(
                fiber_id=tombstone_result.fiber.id,
                memory_type=MemoryType.CONTEXT,
                priority=Priority.from_int(7),
                source="mcp_session",
                expires_in_days=1,
                tags={"session_state"},
                metadata={"active": False, "ended_at": now.isoformat()},
            )
            await storage.add_typed_memory(tombstone_mem)

            # Also save a longer-lived summary for future recall
            summary_result = await encoder.encode(
                content=summary,
                timestamp=now,
                tags={"session_summary"},
            )
            summary_mem = TypedMemory.create(
                fiber_id=summary_result.fiber.id,
                memory_type=MemoryType.CONTEXT,
                priority=Priority.from_int(5),
                source="mcp_session",
                expires_in_days=7,
                tags={"session_summary"},
            )
            await storage.add_typed_memory(summary_mem)
            await storage.batch_save()

            # Auto-save eternal context on session end
            try:
                if self.config.eternal.enabled:
                    ctx = self.get_eternal_context()
                    ctx.add_summary(summary)
                    ctx.save_with_snapshot()
                    ctx.log(f"Session ended: {summary[:80]}")
            except Exception:
                pass

            return {
                "active": False,
                "summary": summary,
                "message": "Session ended and summary saved",
            }

        return {"error": f"Unknown session action: {action}"}

    async def _index(self, args: dict[str, Any]) -> dict[str, Any]:
        """Index codebase into neural memory."""
        from pathlib import Path

        from neural_memory.core.neuron import NeuronType
        from neural_memory.engine.codebase_encoder import CodebaseEncoder

        action = args.get("action", "status")
        storage = await self.get_storage()

        if action == "scan":
            brain = await storage.get_brain(storage._current_brain_id)
            if not brain:
                return {"error": "No brain configured"}

            path = Path(args.get("path", ".")).resolve()
            if not path.is_dir():
                return {"error": f"Not a directory: {path}"}

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

        elif action == "status":
            indexed_files = await storage.find_neurons(
                type=NeuronType.SPATIAL,
                limit=1000,
            )
            code_files = [n for n in indexed_files if n.metadata.get("indexed")]

            return {
                "indexed_files": len(code_files),
                "file_list": [n.content for n in code_files[:20]],
                "message": f"{len(code_files)} files indexed"
                if code_files
                else "No codebase indexed yet. Use scan action.",
            }

        return {"error": f"Unknown index action: {action}"}

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

        adapter_kwargs: dict[str, Any] = {}
        connection = args.get("connection")

        if source == "chromadb":
            if connection:
                adapter_kwargs["path"] = connection
        elif source == "mem0":
            if connection:
                adapter_kwargs["api_key"] = connection
            if args.get("user_id"):
                adapter_kwargs["user_id"] = args["user_id"]
        elif source == "awf":
            if connection:
                adapter_kwargs["brain_dir"] = connection
        elif source == "cognee":
            if connection:
                adapter_kwargs["api_key"] = connection
        elif source == "graphiti":
            if connection:
                adapter_kwargs["uri"] = connection
            if args.get("group_id"):
                adapter_kwargs["group_id"] = args["group_id"]
        elif source == "llamaindex":
            if connection:
                adapter_kwargs["persist_dir"] = connection

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

    async def _eternal(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage eternal context persistence."""
        action = args.get("action", "status")
        ctx = self.get_eternal_context()

        if action == "status":
            return {
                "enabled": self.config.eternal.enabled,
                "loaded": ctx.is_loaded,
                "brain": {
                    "project_name": ctx.brain.project_name,
                    "tech_stack": list(ctx.brain.tech_stack),
                    "decisions_count": len(ctx.brain.key_decisions),
                    "instructions_count": len(ctx.brain.instructions),
                },
                "session": {
                    "feature": ctx.session.feature,
                    "task": ctx.session.task,
                    "progress": ctx.session.progress,
                    "errors_count": len(ctx.session.errors_history),
                    "pending_tasks_count": len(ctx.session.pending_tasks),
                    "branch": ctx.session.branch,
                },
                "context": {
                    "message_count": ctx.context.message_count,
                    "summaries_count": len(ctx.context.conversation_summary),
                    "recent_files_count": len(ctx.context.recent_files),
                    "token_estimate": ctx.context.token_estimate,
                },
                "context_usage": round(
                    ctx.estimate_context_usage(self.config.eternal.max_context_tokens), 3
                ),
            }

        elif action == "save":
            # Apply optional updates before saving
            if "project_name" in args:
                ctx.update_brain(project_name=args["project_name"])
            if "tech_stack" in args:
                ctx.update_brain(tech_stack=args["tech_stack"])
            if "decision" in args:
                ctx.add_decision(args["decision"], args.get("reason", ""))
            if "instruction" in args:
                instructions = (*ctx.brain.instructions, args["instruction"])
                ctx.update_brain(instructions=instructions)

            ctx.save_with_snapshot()
            ctx.log("Manual save triggered")
            return {
                "saved": True,
                "message": "Eternal context saved with snapshot.",
            }

        elif action == "load":
            ctx.load()
            return {
                "loaded": True,
                "project_name": ctx.brain.project_name,
                "feature": ctx.session.feature,
                "task": ctx.session.task,
                "message": "Eternal context reloaded from files.",
            }

        elif action == "compact":
            summary = ctx.compact()
            ctx.save(tiers=(2, 3))
            ctx.log(f"Compacted: {summary[:100]}")
            return {
                "compacted": True,
                "summary": summary,
                "message": "Tier 3 compacted into Tier 2.",
            }

        return {"error": f"Unknown eternal action: {action}"}

    async def _recap(self, args: dict[str, Any]) -> dict[str, Any]:
        """Load saved context for session resumption."""
        ctx = self.get_eternal_context()
        if not ctx.is_loaded:
            ctx.load()

        topic = args.get("topic")

        # Topic-based search: query recall + format
        if topic:
            storage = await self.get_storage()
            brain = await storage.get_brain(storage._current_brain_id)
            if brain:
                pipeline = ReflexPipeline(storage, brain.config)
                result = await pipeline.query(
                    query=topic,
                    depth=DepthLevel(1),
                    max_tokens=500,
                    reference_time=datetime.now(),
                )
                # Combine recall results with eternal context
                context_text = ctx.get_injection(level=1)
                if result.context:
                    context_text += f"\n\n## Topic: {topic}\n{result.context}"
                return {
                    "context": context_text,
                    "topic": topic,
                    "confidence": result.confidence,
                    "level": 1,
                    "message": f"Recap for topic: {topic}",
                }

        level = args.get("level", 1)
        level = max(1, min(3, level))

        context_text = ctx.get_injection(level=level)
        token_est = int(len(context_text.split()) * 1.3)

        return {
            "context": context_text,
            "level": level,
            "tokens_used": token_est,
            "message": f"Level {level} recap loaded."
            + (" Welcome back!" if ctx.session.feature else ""),
        }

    def _fire_eternal_trigger(self, text: str) -> None:
        """Fire-and-forget: check auto-save triggers and save if needed.

        Never blocks or raises. Swallows all errors.
        """
        if not self.config.eternal.enabled:
            return
        try:
            ctx = self.get_eternal_context()
            msg_count = ctx.increment_message_count()

            result = check_triggers(
                text=text,
                message_count=msg_count,
                token_estimate=ctx.context.token_estimate,
                max_tokens=self.config.eternal.max_context_tokens,
                checkpoint_interval=self.config.eternal.auto_save_interval,
                warning_threshold=self.config.eternal.context_warning_threshold,
            )

            if result.triggered:
                ctx.save(tiers=result.save_tiers)
                ctx.log(f"Auto-save [{result.trigger_type}]: {result.message}")
        except Exception:
            pass

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
