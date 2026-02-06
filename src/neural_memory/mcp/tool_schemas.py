"""MCP tool schema definitions for NeuralMemory."""

from __future__ import annotations

from typing import Any


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return list of available MCP tool schemas."""
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
                        "description": "Search depth: 0=instant (direct lookup, 1 hop), 1=context (spreading activation, 3 hops), 2=habit (cross-time patterns, 4 hops), 3=deep (full graph traversal). Auto-detected if unset.",
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
            "description": "Auto-capture memories from text. Use 'process' to analyze and save in one call. Call this after important conversations to capture decisions, errors, todos, and facts automatically.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "enable", "disable", "analyze", "process"],
                        "description": "Action: 'process' analyzes and saves, 'analyze' only detects",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to analyze (required for 'analyze' and 'process')",
                    },
                    "save": {
                        "type": "boolean",
                        "description": "Force save even if auto-capture disabled (for 'analyze')",
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "nmem_suggest",
            "description": "Get autocomplete suggestions from brain neurons. Returns matches ranked by relevance and usage frequency.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "The prefix text to autocomplete",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Max suggestions (default: 5)",
                    },
                    "type_filter": {
                        "type": "string",
                        "enum": [
                            "time",
                            "spatial",
                            "entity",
                            "action",
                            "state",
                            "concept",
                            "sensory",
                            "intent",
                        ],
                        "description": "Filter by neuron type",
                    },
                },
                "required": ["prefix"],
            },
        },
        {
            "name": "nmem_session",
            "description": "Track current working session state (task, feature, progress). Use at session start to resume context, and during work to track progress.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "set", "end"],
                        "description": "get=load current session, set=update session state, end=close session",
                    },
                    "feature": {
                        "type": "string",
                        "description": "Current feature being worked on",
                    },
                    "task": {
                        "type": "string",
                        "description": "Current specific task",
                    },
                    "progress": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Progress 0.0 to 1.0",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional context notes",
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "nmem_index",
            "description": "Index codebase into neural memory for code-aware recall. Scans Python files and creates neurons for functions, classes, imports. Enables 'where is X implemented?' queries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["scan", "status"],
                        "description": "scan=index codebase, status=show what's indexed",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to index (default: current working directory)",
                    },
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'File extensions to index (default: [".py"])',
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "nmem_import",
            "description": "Import memories from external systems (ChromaDB, Mem0, AWF, Cognee, Graphiti, LlamaIndex) into NeuralMemory. Supports full and incremental sync.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["chromadb", "mem0", "awf", "cognee", "graphiti", "llamaindex"],
                        "description": "Source system to import from",
                    },
                    "connection": {
                        "type": "string",
                        "description": "Connection string/path (e.g., '/path/to/chroma', API key, graph URI, or index dir path)",
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection/namespace to import from",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum records to import",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User ID filter (for Mem0)",
                    },
                },
                "required": ["source"],
            },
        },
        {
            "name": "nmem_eternal",
            "description": "Manage eternal context persistence. Auto-saves project context, decisions, and session state across sessions so nothing is lost.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "save", "load", "compact"],
                        "description": "status=view state, save=force save+snapshot, load=reload from files, compact=summarize context into session",
                    },
                    "project_name": {
                        "type": "string",
                        "description": "Set project name (Tier 1)",
                    },
                    "tech_stack": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Set tech stack (Tier 1)",
                    },
                    "decision": {
                        "type": "string",
                        "description": "Add a key decision (Tier 1)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the decision",
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Add a user instruction (Tier 1)",
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "nmem_recap",
            "description": "Load saved context at session start. Returns project state, decisions, progress. Use at the beginning of every session to resume where you left off.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "description": "Detail level: 1=quick (~500 tokens), 2=detailed (~1300 tokens), 3=full (~3300 tokens). Default: 1",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Search for a specific topic in context (e.g., 'auth', 'database')",
                    },
                },
            },
        },
    ]
