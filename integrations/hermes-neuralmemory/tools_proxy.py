"""Tool schemas + normalization — port of integrations/neuralmemory/src/tools.ts.

Faithful port: same STRIP_KEYS set, same normalization algorithm,
same 5 fallback tools and 2 compat shims with identical schemas.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

# ── Schema normalization ───────────────────────────

# Keywords that some LLM providers reject in function schemas.
STRIP_KEYS = frozenset([
    "minimum",
    "maximum",
    "maxLength",
    "minLength",
    "maxItems",
    "minItems",
    "exclusiveMinimum",
    "exclusiveMaximum",
])


def normalize_schema(node: Any) -> Any:
    """Recursively normalize a JSON Schema node for provider compatibility.

    Port of normalizeSchema(): strip constraint keywords, replace integer
    with number (Gemini compat), add additionalProperties:false to objects
    (OpenAI strict mode), ensure every object has properties (Anthropic SDK).
    """
    if node is None or not isinstance(node, (dict, list)):
        return node

    if isinstance(node, list):
        return [normalize_schema(item) for item in node]

    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in STRIP_KEYS:
            continue
        if key == "type" and value == "integer":
            result[key] = "number"
        elif key == "properties" and isinstance(value, dict):
            result[key] = {prop_name: normalize_schema(prop_schema)
                           for prop_name, prop_schema in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = normalize_schema(value)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            result[key] = [normalize_schema(item) for item in value]
        else:
            result[key] = value

    # Ensure objects have `properties` and `additionalProperties`
    if result.get("type") == "object":
        if result.get("properties") is None:
            result["properties"] = {}
        if "additionalProperties" not in result:
            result["additionalProperties"] = False

    return result


def to_safe_schema(input_schema: dict | None) -> dict:
    """Convert an MCP inputSchema into a provider-safe schema. Port of toSafeSchema()."""
    if not input_schema or not isinstance(input_schema, dict):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    normalized = normalize_schema(input_schema)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": normalized.get("properties") or {},
        "additionalProperties": False,
    }
    required = normalized.get("required")
    if isinstance(required, list) and required:
        schema["required"] = required
    return schema


# ── Tool factory ────────────────────────────────

def make_call_fn(mcp) -> Callable[[str, dict], str]:
    """Create a tool-call helper that auto-reconnects to MCP. Port of makeCallFn().

    Returns the raw tool text (Hermes tool handlers return JSON strings,
    so the handler wrapper JSON-encodes the outcome).
    """

    def call(tool_name: str, args: dict) -> str:
        if not mcp.connected:
            try:
                mcp.ensure_connected()
            except Exception as err:  # noqa: BLE001
                return json.dumps({
                    "error": True,
                    "message": f"NeuralMemory auto-connect failed: {err}",
                })
        try:
            raw = mcp.call_tool(tool_name, args)
            try:
                return json.dumps(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                return json.dumps({"text": raw})
        except Exception as err:  # noqa: BLE001
            return json.dumps({
                "error": True,
                "message": f"Tool {tool_name} failed: {err}",
            })

    return call


# ── Fallback tools (registered synchronously at plugin load) ──────

def create_fallback_tools(mcp) -> list[dict]:
    """Port of createFallbackTools() — 5 core tools, identical schemas."""
    call = make_call_fn(mcp)

    tools = [
        {
            "name": "nmem_remember",
            "description": (
                "Store a memory in NeuralMemory. Use this to remember facts, decisions, "
                "insights, todos, errors, and other information that should persist across sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The content to remember"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "fact", "decision", "preference", "todo", "insight",
                            "context", "instruction", "error", "workflow", "reference",
                        ],
                        "description": "Memory type (auto-detected if not specified)",
                    },
                    "priority": {"type": "number", "description": "Priority 0-10 (5=normal, 10=critical)"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            "handler": lambda params, **kw: call("nmem_remember", params),
        },
        {
            "name": "nmem_recall",
            "description": (
                "Query memories from NeuralMemory. Use this to recall past information, "
                "decisions, patterns, or context relevant to the current task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to search memories"},
                    "depth": {"type": "number", "description": "Search depth: 0=instant, 1=context, 2=habit, 3=deep"},
                    "max_tokens": {"type": "number", "description": "Maximum tokens in response (default: 500)"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "handler": lambda params, **kw: call("nmem_recall", params),
        },
        {
            "name": "nmem_context",
            "description": "Get recent context from NeuralMemory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "number", "description": "Number of recent memories (default: 10)"},
                },
                "additionalProperties": False,
            },
            "handler": lambda params, **kw: call("nmem_context", params),
        },
        {
            "name": "nmem_stats",
            "description": "Get brain statistics including memory counts and freshness.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "handler": lambda params, **kw: call("nmem_stats", params),
        },
        {
            "name": "nmem_health",
            "description": "Get brain health diagnostics including grade and recommendations.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "handler": lambda params, **kw: call("nmem_health", params),
        },
    ]
    return tools


def create_compatibility_tools(mcp) -> list[dict]:
    """Port of createCompatibilityTools() — legacy memory-core aliases."""
    call = make_call_fn(mcp)

    return [
        {
            "name": "memory_search",
            "description": (
                "Search memories (legacy alias for nmem_recall). "
                "Prefer nmem_recall for full NeuralMemory features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "handler": lambda params, **kw: call("nmem_recall", {"query": params.get("query"), "depth": 1}),
        },
        {
            "name": "memory_get",
            "description": (
                "Get a memory by ID (legacy alias for nmem_recall). "
                "Prefer nmem_recall for full NeuralMemory features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Memory identifier or query"},
                },
                "required": ["id"],
                "additionalProperties": False,
            },
            "handler": lambda params, **kw: call("nmem_recall", {"query": str(params.get("id")), "depth": 0}),
        },
    ]


def create_tools_from_mcp(mcp) -> list[dict]:
    """Port of createToolsFromMcp() — dynamic discovery from tools/list.

    Used for diagnostics/logging at startup (Hermes freezes the tool list
    after register(), so dynamically-discovered tools beyond the fallback
    set are logged rather than registered — documented adaptation).
    """
    mcp_tools = mcp.list_tools()
    call = make_call_fn(mcp)
    tools = []
    for t in mcp_tools:
        tool_name = t.get("name", "")

        def _make_handler(name: str):
            return lambda params, **kw: call(name, params)

        tools.append({
            "name": tool_name,
            "description": t.get("description") or f"NeuralMemory tool: {tool_name}",
            "parameters": to_safe_schema(t.get("inputSchema")),
            "handler": _make_handler(tool_name),
        })
    return tools
