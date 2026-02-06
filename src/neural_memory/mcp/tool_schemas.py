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
    ]
