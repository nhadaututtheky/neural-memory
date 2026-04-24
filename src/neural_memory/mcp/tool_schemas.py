"""MCP tool schema definitions for NeuralMemory."""

from __future__ import annotations

from typing import Any

# Tool tier definitions — controls which tools are exposed via tools/list.
# Hidden tools remain callable via dispatch (safety net).
TOOL_TIERS: dict[str, frozenset[str]] = {
    "minimal": frozenset(
        {
            "nmem_remember",
            "nmem_recall",
            "nmem_context",
            "nmem_recap",
        }
    ),
    "standard": frozenset(
        {
            "nmem_remember",
            "nmem_remember_batch",
            "nmem_recall",
            "nmem_context",
            "nmem_recap",
            "nmem_todo",
            "nmem_session",
            "nmem_auto",
            "nmem_eternal",
        }
    ),
    # "full" = all tools, no filtering
}


_COMPACT_PROPERTY: dict[str, Any] = {
    "type": "boolean",
    "description": "Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens.",
}

_TOKEN_BUDGET_PROPERTY: dict[str, Any] = {
    "type": "integer",
    "minimum": 50,
    "description": "Max tokens for response. Progressively strips content to fit budget.",
}

# Parameters injected into every tool schema.
_INJECTED_PARAMS: dict[str, dict[str, Any]] = {
    "compact": _COMPACT_PROPERTY,
    "token_budget": _TOKEN_BUDGET_PROPERTY,
}


def _with_parameters_alias(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``parameters`` as an alias for ``inputSchema`` on each tool.

    MCP clients read ``inputSchema``, but OpenAI-compatible bridges
    (Cursor, LiteLLM, etc.) read ``parameters``.  Including both keys
    prevents HTTP 400 errors when tools are forwarded to OpenAI API.

    Also injects ``compact`` and ``token_budget`` parameters into every tool schema.
    """
    out: list[dict[str, Any]] = []
    for tool in schemas:
        t = {**tool}
        # Inject compact + token_budget parameters into inputSchema
        if "inputSchema" in t:
            schema = {**t["inputSchema"]}
            props = {**schema.get("properties", {})}
            for param_name, param_schema in _INJECTED_PARAMS.items():
                if param_name not in props:
                    props[param_name] = param_schema
            schema["properties"] = props
            t["inputSchema"] = schema
        if "inputSchema" in t and "parameters" not in t:
            t["parameters"] = t["inputSchema"]
        out.append(t)
    return out


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return list of all MCP tool schemas (unfiltered)."""
    return _with_parameters_alias(_ALL_TOOL_SCHEMAS)


def get_tool_schemas_for_tier(tier: str) -> list[dict[str, Any]]:
    """Return tool schemas filtered by tier.

    Args:
        tier: One of "minimal", "standard", "full".
              Unknown values default to "full".

    Returns:
        List of tool schema dicts for the requested tier.
    """
    allowed = TOOL_TIERS.get(tier)
    if allowed is None:
        # "full" or unknown → return all
        return _with_parameters_alias(_ALL_TOOL_SCHEMAS)
    return _with_parameters_alias([t for t in _ALL_TOOL_SCHEMAS if t["name"] in allowed])


_ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "nmem_remember",
        "description": "Store a memory. Auto-detects type, auto-resolves contradicted errors (RESOLVED_BY synapse). "
        "Use after completing a task, fixing a bug, or making a decision. "
        "Don't use for temporary notes (use ephemeral=true) or project context (use nmem_eternal).",
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
                        "boundary",
                    ],
                    "description": "Memory type (auto-detected if not specified)",
                },
                "tier": {
                    "type": "string",
                    "enum": ["hot", "warm", "cold"],
                    "description": "Memory tier: hot (always in context, slow decay), "
                    "warm (default, semantic match), cold (explicit recall only, fast decay). "
                    "Boundary type auto-promotes to hot.",
                },
                "domain": {
                    "type": "string",
                    "maxLength": 50,
                    "description": "Domain scope for boundary memories (e.g. 'financial', 'security', 'code-review'). "
                    "Adds a domain:{value} tag. Boundaries without domain are global (apply everywhere). "
                    "Only meaningful for type=boundary; ignored for other types.",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Priority 0-10 (5=normal, 10=critical)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 100},
                    # maxItems: 50 for storage (remember) vs 20 for filtering (recall)
                    # — storing supports richer tagging, filtering caps for query perf
                    "maxItems": 50,
                    "description": "Tags for categorization",
                },
                "expires_days": {
                    "type": "integer",
                    "description": "Days until memory expires",
                },
                "encrypted": {
                    "type": "boolean",
                    "description": "Force encrypt this memory's neuron content (default: false). When true, content is encrypted with the brain's Fernet key regardless of sensitive content detection.",
                },
                "event_at": {
                    "type": "string",
                    "description": "ISO datetime of when the event originally occurred "
                    "(e.g. '2026-03-02T08:00:00'). Defaults to current time if not provided. "
                    "Useful for batch-importing past events with correct timestamps.",
                },
                "trust_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Trust level 0.0-1.0. Capped by source ceiling "
                    "(user_input max 0.9, ai_inference max 0.7). NULL = unscored.",
                },
                "source_id": {
                    "type": "string",
                    "description": "Link this memory to a registered source. "
                    "Creates a SOURCE_OF synapse for provenance tracking.",
                },
                "context": {
                    "type": "object",
                    "description": "Structured context dict merged into content "
                    "server-side using type-specific templates. Keys like "
                    "'reason', 'alternatives', 'cause', 'fix', 'steps' are "
                    "auto-expanded. For type='decision': 'chosen', 'alternatives'/"
                    "'rejected', 'confidence' enable decision intelligence "
                    "(overlap detection, evolution tracking). Any agent can send "
                    "structured data instead of crafting perfect prose.",
                    "additionalProperties": True,
                },
                "ephemeral": {
                    "type": "boolean",
                    "description": "Session-scoped memory: auto-expires after TTL (default 24h), "
                    "never synced to cloud, excluded from consolidation. "
                    "Use for scratch notes, debugging context, temporary reasoning.",
                },
                "compact": {
                    "type": "boolean",
                    "description": "Compact response: return only success + fiber_id + memory_type, skip verbose metadata. Saves 200-400 tokens. Default: true. Set false for full response.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "nmem_remember_batch",
        "description": "Store multiple memories at once (max 20). Use when saving 3+ memories together. "
        "Partial success — one bad item won't block the rest.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
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
                            "encrypted": {
                                "type": "boolean",
                                "description": "Force encrypt this memory",
                            },
                            "event_at": {
                                "type": "string",
                                "description": "ISO datetime of when the event originally occurred",
                            },
                            "trust_score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Trust level 0.0-1.0",
                            },
                            "source_id": {
                                "type": "string",
                                "description": "Link to a registered source",
                            },
                            "ephemeral": {
                                "type": "boolean",
                                "description": "Session-scoped memory (auto-expires, never synced)",
                            },
                        },
                        "required": ["content"],
                    },
                    "description": "Array of memories to store (max 20)",
                    "maxItems": 20,
                },
            },
            "required": ["memories"],
        },
    },
    {
        "name": "nmem_recall",
        "description": "Query memories via spreading activation. Use when you need past context, decisions, or knowledge. "
        "Depth: 0=instant lookup, 1=context (default), 2=cross-time patterns, 3=deep graph. "
        "Add tags for precision. Use nmem_context instead for broad recent context.",
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
                    "minimum": 1,
                    "maximum": 10000,
                    "description": "Maximum tokens in response (default: 500)",
                },
                "min_confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Minimum confidence threshold",
                },
                "valid_at": {
                    "type": "string",
                    "description": "ISO datetime string to filter memories valid at that point in time (e.g. '2026-02-01T12:00:00')",
                },
                "include_conflicts": {
                    "type": "boolean",
                    "description": "Include full conflict details in response (default: false). When false, only has_conflicts flag and conflict_count are returned.",
                },
                "warn_expiry_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 90,
                    "description": "If set, warn about memories expiring within this many days. Adds expiry_warnings to response.",
                },
                "brains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of brain names to query across (max 5). When provided, runs parallel recall across all specified brains and merges results.",
                },
                "min_trust": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Filter: only return memories with trust_score >= this value. Unscored memories (NULL) are always included.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 100},
                    "maxItems": 20,
                    "description": "Filter by tags. Checks tags, auto_tags, and agent_tags columns.",
                },
                "tag_mode": {
                    "type": "string",
                    "enum": ["and", "or"],
                    "description": "Tag matching mode: 'and' (default, all tags must match) or 'or' (any tag matches).",
                },
                "mode": {
                    "type": "string",
                    "enum": ["associative", "exact"],
                    "description": "Recall mode: 'associative' (default) returns formatted context, 'exact' returns raw neuron contents verbatim without truncation or summarization.",
                },
                "include_citations": {
                    "type": "boolean",
                    "description": "Include citation and audit trail in exact recall results (default: true).",
                },
                "recall_token_budget": {
                    "type": "integer",
                    "minimum": 50,
                    "maximum": 100000,
                    "description": "When set, activates budget-aware fiber selection: ranks fibers by value-per-token "
                    "and selects the most efficient ones to fit within this budget. "
                    "Adds budget_stats to the response. Default: not set (uses standard sequential truncation).",
                },
                "permanent_only": {
                    "type": "boolean",
                    "description": "Exclude ephemeral (session-scoped) memories from results. Default: false (include all).",
                },
                "clean_for_prompt": {
                    "type": "boolean",
                    "description": "Return clean bullet-point text without section headers or neuron-type tags. Default: true.",
                },
                "compact": {
                    "type": "boolean",
                    "description": "Compact mode: return only core answer + confidence, skip all optional metadata (thought_chains, sources, cognitive_chunks, etc). Saves 200-800 tokens. Default: true. Set false for full metadata.",
                },
                "tier": {
                    "type": "string",
                    "enum": ["hot", "warm", "cold"],
                    "description": "Filter results by memory tier. Only return memories matching this tier.",
                },
                "domain": {
                    "type": "string",
                    "maxLength": 50,
                    "description": "Domain scope filter. When set, HOT context injection only includes boundaries "
                    "tagged with this domain (plus unscoped global boundaries). "
                    "Example: domain='financial' filters out security boundaries from context.",
                },
                "as_of": {
                    "type": "string",
                    "description": "ISO datetime for time-travel recall. Returns only memories that existed at that point "
                    "in time (created_at <= as_of) and reconstructs their maturation stage. "
                    "Example: '2026-03-01T00:00:00' recalls memory state as of March 1st.",
                },
                "simhash_threshold": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 64,
                    "description": "SimHash pre-filter Hamming distance cutoff. Neurons with content_hash farther than "
                    "this threshold from the query hash are excluded before spreading activation. "
                    "0 = disabled (default). Lower values = stricter filtering. Overrides brain config for this query.",
                },
                "min_arousal": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Filter: only return memories with arousal (emotional intensity) >= this value. "
                    "Arousal is detected at encoding time (0.0=neutral, 1.0=maximum intensity). "
                    "Use to find emotionally significant memories (e.g. incidents, breakthroughs).",
                },
                "valence": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "Filter: only return memories with this emotional valence. "
                    "Valence is detected at encoding via sentiment analysis. "
                    "Use to find e.g. only frustrations (negative) or breakthroughs (positive).",
                },
                "layer": {
                    "type": "string",
                    "enum": ["auto", "project", "global"],
                    "description": "Layer scope: 'auto' (default) merges project + global brains, "
                    "'project' restricts to current brain only, "
                    "'global' queries only the global brain.",
                },
                "exclude_reflexes": {
                    "type": "boolean",
                    "description": "Exclude reflex (always-on) neurons from this recall. Default: false.",
                },
                "include_paths": {
                    "type": "boolean",
                    "description": "Include activation paths (thought chains) showing how each neuron was reached. "
                    "Returns top-5 paths with neuron content and hop distance. Default: false.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "nmem_show",
        "description": "Get full verbatim content + metadata + synapses for a memory by ID. "
        "Use after recall when you need exact content, not the summarized version.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The fiber_id or neuron_id of the memory to retrieve",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "nmem_provenance",
        "description": "Trace or audit a memory's origin chain. Use when verifying where a fact came from "
        "or adding verification/approval stamps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["trace", "verify", "approve"],
                    "description": "Action: trace (view chain), verify (mark verified), approve (mark approved).",
                },
                "neuron_id": {
                    "type": "string",
                    "description": "Neuron ID to trace/verify/approve.",
                },
                "actor": {
                    "type": "string",
                    "description": "Who is performing the verification/approval (default: mcp_agent).",
                },
            },
            "required": ["action", "neuron_id"],
        },
    },
    {
        "name": "nmem_source",
        "description": "Register external sources (docs, laws, APIs) for provenance tracking. "
        "Use before nmem_train to link trained memories to their origin.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["register", "list", "get", "update", "delete"],
                    "description": "Action to perform on sources.",
                },
                "source_id": {
                    "type": "string",
                    "description": "Source ID (required for get/update/delete).",
                },
                "name": {
                    "type": "string",
                    "description": "Source name (required for register).",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "law",
                        "contract",
                        "ledger",
                        "document",
                        "api",
                        "manual",
                        "website",
                        "book",
                        "research",
                    ],
                    "description": "Type of source (default: document).",
                },
                "version": {
                    "type": "string",
                    "description": "Version string (e.g. '2024-01', 'v2.0').",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "superseded", "repealed", "draft"],
                    "description": "Source lifecycle status.",
                },
                "file_hash": {
                    "type": "string",
                    "description": "File hash for integrity checking.",
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata.",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_context",
        "description": "Get recent memories as auto-injected context. Use for broad task context. "
        "For specific queries use nmem_recall. For project-level context use nmem_recap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Number of recent memories (default: 10)",
                },
                "fresh_only": {
                    "type": "boolean",
                    "description": "Only include memories < 30 days old",
                },
                "warn_expiry_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 90,
                    "description": "If set, warn about memories expiring within this many days. Adds expiry_warnings to response.",
                },
                "include_ghosts": {
                    "type": "boolean",
                    "description": "Include faded ghost memories at bottom of context with recall keys (default: true). Set false to suppress.",
                },
            },
        },
    },
    {
        "name": "nmem_todo",
        "description": "Quick TODO memory (auto-expires in 30 days). Use nmem_forget to close when done.",
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
        "description": "Quick brain stats: counts and freshness. For quality assessment use nmem_health instead.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "nmem_auto",
        "description": "Auto-extract memories from text. 'process'=analyze+save, 'flush'=emergency capture before compaction. "
        "Use at session end or when processing large text blocks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "enable", "disable", "analyze", "process", "flush"],
                    "description": "Action: 'process' analyzes and saves, 'analyze' only detects, 'flush' emergency capture before compaction (skips dedup, lower threshold)",
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
        "description": "Autocomplete from brain neurons. No prefix = idle/neglected neurons needing reinforcement.",
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
        },
    },
    {
        "name": "nmem_session",
        "description": "Track current session state (task, feature, progress). Single-session only. "
        "For cross-session persistence use nmem_eternal.",
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
        "description": "Index codebase for code-aware recall. Extracts symbols, imports, and relationships. "
        "Run once per project, re-scan after major changes.",
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
                    "description": 'File extensions to index (default: [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cc"])',
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_import",
        "description": "Import memories from external systems (ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex). "
        "One-time migration tool.",
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
                    "description": "Connection string/path (e.g., '/path/to/chroma', graph URI, or index dir path). For API keys, prefer env vars: MEM0_API_KEY, COGNEE_API_KEY.",
                },
                "collection": {
                    "type": "string",
                    "description": "Collection/namespace to import from",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
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
        "description": "SAVE project context, decisions, instructions that persist across sessions. "
        "Pair with nmem_recap to LOAD. Use for project-level facts, not task-specific memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "save"],
                    "description": "status=view memory counts and session state, save=store project context/decisions/instructions",
                },
                "project_name": {
                    "type": "string",
                    "description": "Set project name (saved as FACT)",
                },
                "tech_stack": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Set tech stack (saved as FACT)",
                },
                "decision": {
                    "type": "string",
                    "description": "Add a key decision (saved as DECISION)",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the decision",
                },
                "instruction": {
                    "type": "string",
                    "description": "Add a persistent instruction (saved as INSTRUCTION)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_recap",
        "description": "LOAD project context saved by nmem_eternal. Call at SESSION START to restore cross-session state. "
        "Level 1=quick (~500 tokens), 2=detailed, 3=full.",
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
    {
        "name": "nmem_health",
        "description": "Primary health check — purity score, grade, warnings. Call FIRST, then fix top penalty. "
        "For specific alerts use nmem_alerts. For trends use nmem_evolution.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "nmem_evolution",
        "description": "Long-term brain growth trends: maturation, plasticity, coherence. "
        "Use for trend analysis, not immediate health (use nmem_health for that).",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "nmem_habits",
        "description": "Learned workflow habits from tool usage patterns. Suggest next action, list habits, or clear.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["suggest", "list", "clear"],
                    "description": "suggest=get next action suggestions, list=show learned habits, clear=remove all habits",
                },
                "current_action": {
                    "type": "string",
                    "description": "Current action type for suggestions (required for suggest action)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_version",
        "description": "Brain version control: snapshot current state, rollback, or diff between versions. "
        "Use before risky consolidation or major changes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "rollback", "diff"],
                    "description": "create=snapshot current state, list=show versions, rollback=restore version, diff=compare versions",
                },
                "name": {
                    "type": "string",
                    "description": "Version name (required for create)",
                },
                "description": {
                    "type": "string",
                    "description": "Version description (optional for create)",
                },
                "version_id": {
                    "type": "string",
                    "description": "Version ID (required for rollback)",
                },
                "from_version": {
                    "type": "string",
                    "description": "Source version ID (required for diff)",
                },
                "to_version": {
                    "type": "string",
                    "description": "Target version ID (required for diff)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Max versions to list (default: 20)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_transplant",
        "description": "Copy memories from another brain by tags/types. Use for sharing knowledge between project brains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_brain": {
                    "type": "string",
                    "description": "Name of the source brain to extract from",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to filter — fibers matching ANY tag will be included",
                },
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory types to filter (fact, decision, etc.)",
                },
                "strategy": {
                    "type": "string",
                    "enum": [
                        "prefer_local",
                        "prefer_remote",
                        "prefer_recent",
                        "prefer_stronger",
                    ],
                    "description": "Conflict resolution strategy (default: prefer_local)",
                },
            },
            "required": ["source_brain"],
        },
    },
    {
        "name": "nmem_conflicts",
        "description": "Detect and resolve conflicting memories. Pre-check new content for contradictions before saving.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "resolve", "check"],
                    "description": "list=view active conflicts, resolve=manually resolve a conflict, check=pre-check content for conflicts",
                },
                "neuron_id": {
                    "type": "string",
                    "description": "Neuron ID of the disputed memory (required for resolve)",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["keep_existing", "keep_new", "keep_both"],
                    "description": "How to resolve: keep_existing=undo dispute, keep_new=supersede old, keep_both=accept both",
                },
                "content": {
                    "type": "string",
                    "description": "Content to pre-check for conflicts (required for check)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for more accurate conflict checking",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Max conflicts to list (default: 50)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_train",
        "description": "Train brain from docs (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV). "
        "Pinned by default as permanent KB. Requires: pip install neural-memory[extract].",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["train", "status"],
                    "description": "train=process docs into brain, status=show training stats",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to train from (default: current directory)",
                },
                "domain_tag": {
                    "type": "string",
                    "maxLength": 100,
                    "description": "Domain tag for all chunks (e.g., 'react', 'kubernetes')",
                },
                "brain_name": {
                    "type": "string",
                    "maxLength": 64,
                    "description": "Target brain name (default: current brain)",
                },
                "extensions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            ".md",
                            ".mdx",
                            ".txt",
                            ".rst",
                            ".pdf",
                            ".docx",
                            ".pptx",
                            ".html",
                            ".htm",
                            ".json",
                            ".xlsx",
                            ".csv",
                        ],
                    },
                    "description": "File extensions to include (default: ['.md']). "
                    "Rich formats (PDF, DOCX, PPTX, HTML, XLSX) require: pip install neural-memory[extract]",
                },
                "consolidate": {
                    "type": "boolean",
                    "description": "Run ENRICH consolidation after encoding (default: true)",
                },
                "pinned": {
                    "type": "boolean",
                    "description": "Pin trained memories as permanent KB — skip decay/prune/compress (default: true)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_pin",
        "description": "Pin memories as permanent KB (skip decay/pruning/compression). Use for critical knowledge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["pin", "unpin", "list"],
                    "description": "Action: pin (default), unpin, or list pinned memories",
                },
                "fiber_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fiber IDs to pin or unpin (required for pin/unpin, ignored for list)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results for list action (default: 50, max: 200)",
                    "minimum": 1,
                    "maximum": 200,
                },
            },
        },
    },
    {
        "name": "nmem_reflex",
        "description": "Pin/unpin neurons as reflexes (always-on in every recall). Reflexes bypass spreading activation and appear first in context. "
        "Use for critical rules, preferences, or constraints that must always be present.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["pin", "unpin", "list"],
                    "description": "Action: pin neuron as reflex, unpin to remove, or list all reflexes",
                },
                "neuron_id": {
                    "type": "string",
                    "description": "Neuron ID to pin or unpin (required for pin/unpin, ignored for list)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results for list action (default: 20, max: 50)",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
        },
    },
    {
        "name": "nmem_train_db",
        "description": "Train brain from database schema (tables, columns, relationships). SQLite supported.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["train", "status"],
                    "description": "train=extract schema into brain, status=show training stats",
                },
                "connection_string": {
                    "type": "string",
                    "maxLength": 500,
                    "description": "Database connection string (v1: sqlite:///path/to/db.db)",
                },
                "domain_tag": {
                    "type": "string",
                    "maxLength": 100,
                    "description": "Domain tag for schema knowledge (e.g., 'ecommerce', 'analytics')",
                },
                "brain_name": {
                    "type": "string",
                    "maxLength": 64,
                    "description": "Target brain name (default: current brain)",
                },
                "consolidate": {
                    "type": "boolean",
                    "description": "Run ENRICH consolidation after encoding (default: true)",
                },
                "max_tables": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Maximum tables to process (default: 100)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_alerts",
        "description": "Actionable health alerts. Call after nmem_health to see specific issues. "
        "Acknowledge alerts after fixing them.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "acknowledge"],
                    "description": "list=view active/seen alerts, acknowledge=mark alert as handled",
                },
                "alert_id": {
                    "type": "string",
                    "description": "Alert ID to acknowledge (required for acknowledge action)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Max alerts to list (default: 50)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_narrative",
        "description": "Generate memory narratives: timeline (date range), topic (spreading activation), or causal chain. "
        "Use to understand how knowledge connects.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["timeline", "topic", "causal"],
                    "description": "timeline=date-range narrative, topic=SA-driven topic narrative, causal=causal chain narrative",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic to explore (required for topic and causal actions)",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in ISO format (required for timeline, e.g., '2026-02-01')",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in ISO format (required for timeline, e.g., '2026-02-18')",
                },
                "max_fibers": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Max fibers in narrative (default: 20)",
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Max causal chain depth (default: 5, for causal action only)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_visualize",
        "description": "Generate charts from memory data (Vega-Lite/markdown/ASCII). "
        "Use for financial metrics, trends, or any structured data in memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to visualize (e.g., 'ROE trend across quarters', 'revenue by product')",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "pie", "scatter", "table", "timeline"],
                    "description": "Chart type (auto-detected if omitted based on data shape)",
                },
                "format": {
                    "type": "string",
                    "enum": ["vega_lite", "markdown_table", "ascii", "all"],
                    "description": "Output format (default: vega_lite)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Max data points (default: 20)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "nmem_watch",
        "description": "Watch directories for file changes, auto-ingest into memory. "
        "Scan for one-shot, start/stop for continuous monitoring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["scan", "start", "stop", "status", "list"],
                    "description": "scan=one-shot ingest directory, start=background watch, stop=stop watching, status=show stats, list=tracked files",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory path to scan (for scan action)",
                },
                "directories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of directory paths to watch (for start action, max 10)",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "deleted"],
                    "description": "Filter files by status (for list action)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Max files to return (default: 50, for list action)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_review",
        "description": "Spaced repetition reviews (Leitner 5-box system). Queue due reviews, mark success/fail, view stats.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["queue", "mark", "schedule", "stats"],
                    "description": "queue=get due reviews, mark=record review result, schedule=manually schedule a fiber, stats=review statistics",
                },
                "fiber_id": {
                    "type": "string",
                    "description": "Fiber ID (required for mark and schedule actions)",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether recall was successful (for mark action, default: true)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Max items in queue (default: 20)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_sync",
        "description": "Manual sync with cloud hub. Push local changes, pull remote, or full bidirectional sync.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["push", "pull", "full", "seed"],
                    "description": "push=send local changes, pull=get remote changes, full=bidirectional sync, seed=populate change log from existing data for initial sync",
                },
                "hub_url": {
                    "type": "string",
                    "description": "Hub server URL (overrides config). Must be http:// or https://",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["prefer_recent", "prefer_local", "prefer_remote", "prefer_stronger"],
                    "description": "Conflict resolution strategy (default: from config)",
                },
                "api_key": {
                    "type": "string",
                    "description": "API key override (default: from config)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_sync_status",
        "description": "View sync status: pending changes, connected devices, last sync time.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "nmem_sync_config",
        "description": "Configure sync: setup (onboarding), activate (license key), get/set settings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "setup", "activate"],
                    "description": "get=view config, set=update config, setup=guided onboarding, activate=activate purchased license key",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Enable/disable sync",
                },
                "hub_url": {
                    "type": "string",
                    "description": "Hub server URL (default: cloud hub)",
                },
                "api_key": {
                    "type": "string",
                    "description": "API key for cloud hub (starts with nmk_)",
                },
                "auto_sync": {
                    "type": "boolean",
                    "description": "Enable/disable auto-sync",
                },
                "sync_interval_seconds": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 86400,
                    "description": "Sync interval in seconds",
                },
                "conflict_strategy": {
                    "type": "string",
                    "enum": ["prefer_recent", "prefer_local", "prefer_remote", "prefer_stronger"],
                    "description": "Default conflict strategy",
                },
                "license_key": {
                    "type": "string",
                    "description": "NM license key to activate (for action='activate', starts with nm_)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_telegram_backup",
        "description": "Backup brain database to Telegram. Requires Telegram bot config.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "brain_name": {
                    "type": "string",
                    "description": "Brain name to backup (default: active brain)",
                },
            },
        },
    },
    {
        "name": "nmem_hypothesize",
        "description": "Create or inspect hypotheses (Bayesian confidence). "
        "Cognitive workflow: hypothesize -> evidence -> predict -> verify -> cognitive (dashboard). "
        "Auto-resolves at >=0.9 (confirmed) or <=0.1 (refuted) with 3+ evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "get"],
                    "description": "create=new hypothesis, list=show all, get=detail view",
                },
                "content": {
                    "type": "string",
                    "description": "Hypothesis statement (required for create)",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.99,
                    "description": "Initial confidence level (default: 0.5)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Priority 0-10 (default: 6)",
                },
                "hypothesis_id": {
                    "type": "string",
                    "description": "Hypothesis neuron ID (required for get)",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "confirmed", "refuted", "superseded", "pending", "expired"],
                    "description": "Filter by status (for list action)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Max results for list (default: 20)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_evidence",
        "description": "Add evidence for/against a hypothesis. Bayesian confidence update with auto-resolve. "
        "Requires an existing hypothesis_id from nmem_hypothesize.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hypothesis_id": {
                    "type": "string",
                    "description": "Target hypothesis neuron ID",
                },
                "content": {
                    "type": "string",
                    "description": "Evidence content — what was observed/discovered",
                },
                "type": {
                    "type": "string",
                    "enum": ["for", "against"],
                    "description": "Evidence direction: 'for' supports, 'against' weakens",
                },
                "weight": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1.0,
                    "description": "Evidence strength (default: 0.5). Higher = stronger evidence",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the evidence memory",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Priority 0-10 (default: 5)",
                },
            },
            "required": ["hypothesis_id", "content", "type"],
        },
    },
    {
        "name": "nmem_predict",
        "description": "Create falsifiable predictions linked to hypotheses. "
        "Use nmem_verify to record outcomes (propagates evidence back to hypothesis).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "get"],
                    "description": "create=new prediction, list=show all, get=detail view",
                },
                "content": {
                    "type": "string",
                    "description": "Prediction statement (required for create)",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.99,
                    "description": "How confident you are in this prediction (default: 0.7)",
                },
                "deadline": {
                    "type": "string",
                    "description": "ISO datetime deadline for verification (e.g. '2026-04-01T00:00:00')",
                },
                "hypothesis_id": {
                    "type": "string",
                    "description": "Link prediction to a hypothesis (creates PREDICTED synapse)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Priority 0-10 (default: 5)",
                },
                "prediction_id": {
                    "type": "string",
                    "description": "Prediction neuron ID (required for get)",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "confirmed", "refuted", "superseded", "pending", "expired"],
                    "description": "Filter by status (for list action)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Max results for list (default: 20)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_verify",
        "description": "Record prediction outcome (correct/wrong). Propagates evidence to linked hypothesis. "
        "Use after observing the predicted event.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prediction_id": {
                    "type": "string",
                    "description": "Target prediction neuron ID",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["correct", "wrong"],
                    "description": "Whether the prediction was correct or wrong",
                },
                "content": {
                    "type": "string",
                    "description": "Observation content — what actually happened (optional)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the observation memory",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Priority 0-10 (default: 5)",
                },
            },
            "required": ["prediction_id", "outcome"],
        },
    },
    {
        "name": "nmem_cognitive",
        "description": "Cognitive dashboard — instant O(1) summary of hypotheses, predictions, calibration, and gaps. "
        "Use as overview after cognitive workflow steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["summary", "refresh"],
                    "description": "summary=get current hot index, refresh=recompute scores",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Max hot items to return (default: 10, for summary)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_gaps",
        "description": "Track what the brain doesn't know. Detect gaps from contradictions, low-confidence, "
        "or recall misses. Resolve when new info fills them.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["detect", "list", "resolve", "get"],
                    "description": "detect=flag new gap, list=show gaps, resolve=mark filled, get=detail",
                },
                "topic": {
                    "type": "string",
                    "description": "What knowledge is missing (required for detect)",
                },
                "source": {
                    "type": "string",
                    "enum": [
                        "contradicting_evidence",
                        "low_confidence_hypothesis",
                        "user_flagged",
                        "recall_miss",
                        "stale_schema",
                    ],
                    "description": "How the gap was detected (default: user_flagged)",
                },
                "priority": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Gap priority (auto-set from source if not provided)",
                },
                "related_neuron_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Neuron IDs related to this gap (max 10)",
                },
                "gap_id": {
                    "type": "string",
                    "description": "Gap ID (required for resolve and get)",
                },
                "resolved_by_neuron_id": {
                    "type": "string",
                    "description": "Neuron that resolved the gap (optional for resolve)",
                },
                "include_resolved": {
                    "type": "boolean",
                    "description": "Include resolved gaps in list (default: false)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Max results for list (default: 20)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_schema",
        "description": "Evolve a hypothesis into a new version (SUPERSEDES chain). "
        "Use when understanding changes — preserves belief evolution history.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["evolve", "history", "compare"],
                    "description": "evolve=create new version, history=version chain, compare=diff two versions",
                },
                "hypothesis_id": {
                    "type": "string",
                    "description": "Neuron ID of the hypothesis to evolve or inspect",
                },
                "content": {
                    "type": "string",
                    "description": "Updated content for the new version (required for evolve)",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.99,
                    "description": "Initial confidence for the new version (inherits from old if not set)",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the hypothesis is being evolved (stored as synapse metadata)",
                },
                "other_id": {
                    "type": "string",
                    "description": "Second hypothesis ID for compare action",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the new version",
                },
            },
            "required": ["action", "hypothesis_id"],
        },
    },
    {
        "name": "nmem_explain",
        "description": "Explain how two concepts connect in the neural graph (shortest path with synapse types and weights). "
        "Use to understand relationships between entities.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_entity": {
                    "type": "string",
                    "description": "Source entity name to start from (e.g. 'React', 'authentication')",
                },
                "to_entity": {
                    "type": "string",
                    "description": "Target entity name to reach (e.g. 'performance', 'JWT')",
                },
                "max_hops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum path length (default: 6)",
                },
            },
            "required": ["from_entity", "to_entity"],
        },
    },
    {
        "name": "nmem_edit",
        "description": "Edit a memory's type, content, priority, or tier. Preserves all synapses. "
        "Use when auto-typing was wrong or content needs correction. For complete replacement, forget+remember.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The fiber ID or neuron ID of the memory to edit",
                },
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
                        "boundary",
                    ],
                    "description": "New memory type",
                },
                "content": {
                    "type": "string",
                    "description": "New content for the anchor neuron",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "New priority (0-10)",
                },
                "tier": {
                    "type": "string",
                    "enum": ["hot", "warm", "cold"],
                    "description": "New memory tier: hot, warm, or cold",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "nmem_forget",
        "description": "Delete a memory. Soft delete by default; hard=true for permanent removal. "
        "Use to close completed TODOs or remove outdated memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The fiber ID of the memory to forget",
                },
                "hard": {
                    "type": "boolean",
                    "description": "Permanent deletion with cascade cleanup (default: false = soft delete)",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this memory is being forgotten (stored in logs)",
                },
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "nmem_consolidate",
        "description": "Run brain consolidation (sleep-like maintenance). Strategy 'all' runs everything in order. "
        "Use periodically or after bulk imports. dry_run=true to preview.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": [
                        "prune",
                        "merge",
                        "summarize",
                        "mature",
                        "infer",
                        "enrich",
                        "dream",
                        "learn_habits",
                        "dedup",
                        "semantic_link",
                        "compress",
                        "process_tool_events",
                        "detect_drift",
                        "all",
                    ],
                    "description": "Consolidation strategy to run (default: all)",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview changes without applying (default: false)",
                },
                "prune_weight_threshold": {
                    "type": "number",
                    "description": "Synapse weight threshold for pruning (default: 0.05)",
                },
                "merge_overlap_threshold": {
                    "type": "number",
                    "description": "Jaccard overlap threshold for merging fibers (default: 0.5)",
                },
                "prune_min_inactive_days": {
                    "type": "number",
                    "description": "Grace period in days before pruning inactive synapses (default: 7.0)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "nmem_drift",
        "description": "Find tags that mean the same thing (Jaccard similarity). Detect clusters, then merge/alias/dismiss.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["detect", "list", "merge", "alias", "dismiss"],
                    "description": "detect=run drift analysis, list=show existing clusters, "
                    "merge/alias/dismiss=resolve a specific cluster",
                },
                "cluster_id": {
                    "type": "string",
                    "description": "Cluster ID to resolve (required for merge/alias/dismiss)",
                },
                "status": {
                    "type": "string",
                    "enum": ["detected", "merged", "aliased", "dismissed"],
                    "description": "Filter clusters by status (for list action)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_surface",
        "description": "Knowledge Surface (.nm file) — compact graph (~1000 tokens) loaded every session. "
        "Generate to rebuild, show to inspect.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["generate", "show"],
                    "description": "generate=rebuild surface from brain.db, show=display current surface info",
                },
                "token_budget": {
                    "type": "integer",
                    "minimum": 200,
                    "maximum": 5000,
                    "description": "Token budget for surface (default: 1200). Only used with generate action.",
                },
                "max_graph_nodes": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 100,
                    "description": "Max graph nodes to include (default: 30). Only used with generate action.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "nmem_tool_stats",
        "description": "Agent tool usage analytics: frequency, success rates, daily trends. "
        "Use to understand which NM tools are being used and how effectively.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["summary", "daily"],
                    "description": "summary=top tools with success rates, daily=usage breakdown by day",
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "description": "Time window in days (default: 30)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Max tools to return (default: 20)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_lifecycle",
        "description": "Manage memory lifecycle: view compression states, freeze/thaw individual memories, "
        "recover compressed content. Use when a memory was incorrectly compressed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "recover", "freeze", "thaw", "at_risk"],
                    "description": "status=show lifecycle distribution, recover=rehydrate compressed memory, "
                    "freeze=prevent compression, thaw=resume normal lifecycle, "
                    "at_risk=show memories expiring soon (forgetting curve)",
                },
                "id": {
                    "type": "string",
                    "description": "Neuron ID (required for recover/freeze/thaw). "
                    "For recover, fiber_id is also accepted.",
                },
                "within_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 90,
                    "description": "For at_risk: number of days to look ahead for expiring memories (default: 7).",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_refine",
        "description": "Refine an instruction/workflow: update content, add failure modes, add trigger patterns. "
        "Use to improve instructions based on real-world execution outcomes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "neuron_id": {
                    "type": "string",
                    "description": "ID of the instruction or workflow memory to refine "
                    "(use the fiber_id returned by nmem_remember or nmem_recall).",
                },
                "new_content": {
                    "type": "string",
                    "description": "Updated instruction text. Replaces current content and increments version.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this refinement was made (stored in refinement_history for auditability).",
                },
                "add_failure_mode": {
                    "type": "string",
                    "description": "Description of a failure mode to append to the failure_modes list "
                    "(deduped, capped at 20).",
                },
                "add_trigger": {
                    "type": "string",
                    "description": "Keyword or phrase to append to trigger_patterns "
                    "(boosts recall when query overlaps, deduped, capped at 10).",
                },
            },
            "required": ["neuron_id"],
        },
    },
    {
        "name": "nmem_report_outcome",
        "description": "Report instruction execution outcome (success/fail). Builds track record — "
        "high success rate boosts recall priority. Call after executing an instruction.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "neuron_id": {
                    "type": "string",
                    "description": "ID of the instruction or workflow memory that was executed.",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether execution succeeded.",
                },
                "failure_description": {
                    "type": "string",
                    "description": "If failed, brief description of what went wrong "
                    "(appended to failure_modes, deduped, capped at 20).",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about the execution (stored for auditability).",
                },
            },
            "required": ["neuron_id", "success"],
        },
    },
    {
        "name": "nmem_budget",
        "description": "Token budget analysis: estimate recall cost, profile brain token usage, "
        "find compression candidates. Use when context window is tight.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["estimate", "analyze", "optimize"],
                    "description": "Action: 'estimate' (dry-run recall cost), 'analyze' (brain token profile), 'optimize' (find compression candidates).",
                },
                "query": {
                    "type": "string",
                    "description": "Query to estimate recall cost for (used with action='estimate').",
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 50,
                    "maximum": 100000,
                    "description": "Token budget to estimate against (default: 4000).",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_tier",
        "description": "Auto-tier: promote/demote memories between HOT/WARM/COLD by access patterns (Pro). "
        "Evaluate (dry-run) before apply. Free users: manual tiers only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "evaluate", "apply", "history", "config", "analytics"],
                    "description": "Action: 'status' (distribution), 'evaluate' (dry-run), 'apply' (execute), "
                    "'history' (fiber tier log), 'config' (thresholds), 'analytics' (type breakdown + velocity).",
                },
                "fiber_id": {
                    "type": "string",
                    "description": "Fiber ID for 'history' action.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true with action='apply', show changes without applying (default: false).",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_boundaries",
        "description": "View domain-scoped boundaries (safety rules, always HOT tier). "
        "List boundaries by domain or view domain summary.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "domains"],
                    "description": "Action: list (show boundaries, optionally filtered by domain), "
                    "domains (list unique domains with boundary counts). Default: list.",
                },
                "domain": {
                    "type": "string",
                    "maxLength": 50,
                    "description": "Filter boundaries by domain (e.g. 'financial', 'security'). "
                    "Only used with action=list.",
                },
            },
        },
    },
    {
        "name": "nmem_milestone",
        "description": "Brain growth milestones (100, 250, 500...10K neurons). "
        "Check for new achievements, view progress to next, or generate growth report.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check", "progress", "history", "report"],
                    "description": "check=detect+record new milestones, progress=distance to next milestone, "
                    "history=all recorded milestones, report=generate growth report for current state",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_store",
        "description": "Brain Store — browse, preview, import, export, and delete knowledge brains. "
        "Share curated brains with the community or import others' expertise.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["browse", "preview", "import", "export", "publish", "delete"],
                    "description": "browse=search community brain registry, preview=view brain details before import, "
                    "import=download and import a brain, export=export current brain as .brain file, "
                    "publish=export and publish brain to community store (requires API key), "
                    "delete=permanently delete a local brain and all its data (requires brain_id)",
                },
                "brain_name": {
                    "type": "string",
                    "description": "Brain name in registry (required for preview/import)",
                },
                "brain_id": {
                    "type": "string",
                    "description": "Local brain ID to delete (required for delete action)",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to actually delete. Without it, returns a preview of what would be deleted.",
                },
                "search": {
                    "type": "string",
                    "description": "Search query for browse (matches name, description, tags)",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "programming",
                        "devops",
                        "writing",
                        "science",
                        "personal",
                        "security",
                        "data",
                        "design",
                        "general",
                    ],
                    "description": "Filter by category (browse) or set category (export)",
                },
                "tag": {
                    "type": "string",
                    "description": "Filter by tag (browse only)",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["created_at", "rating_avg", "download_count"],
                    "description": "Sort order for browse results (default: created_at)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Max results for browse (default: 20, max: 50)",
                },
                "display_name": {
                    "type": "string",
                    "description": "Display name for exported brain (required for export)",
                },
                "description": {
                    "type": "string",
                    "description": "Description for exported brain (export only)",
                },
                "author": {
                    "type": "string",
                    "description": "Author name for exported brain (export only, default: anonymous)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for exported brain (export only)",
                },
                "output_path": {
                    "type": "string",
                    "description": "File path to save exported .brain package (export only)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_goal",
        "description": "Goal-directed recall — manage active goals that bias memory retrieval toward relevant context. "
        "Active goals make recall relevance-based (proximity to goal) instead of just similarity-based.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "subgoals", "activate", "pause", "complete"],
                    "description": "create=new goal, list=show goals, subgoals=list children of a goal, "
                    "activate/pause/complete=change state",
                },
                "goal": {
                    "type": "string",
                    "description": "Goal description (required for create). e.g. 'optimize API latency for sync-hub'",
                },
                "goal_id": {
                    "type": "string",
                    "description": "Goal ID (required for activate/pause/complete/subgoals)",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Goal priority 1-10 (default 5). Higher = stronger recall boost",
                },
                "parent_goal_id": {
                    "type": "string",
                    "description": "Parent goal ID (create only). Makes this a subgoal that inherits parent priority boost",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords for topic EMA seeding (auto-extracted if omitted)",
                },
                "state": {
                    "type": "string",
                    "enum": ["active", "paused", "completed"],
                    "description": "Filter by state (list only)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_causal",
        "description": "Trace causal chains and temporal event sequences through the memory graph. "
        "Use 'trace' to follow CAUSED_BY/LEADS_TO synapses, 'sequence' to follow BEFORE/AFTER, "
        "'temporal_range' to list fibers within a time window, "
        "'temporal_neighborhood' to find fibers temporally adjacent to a given fiber.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["trace", "sequence", "temporal_range", "temporal_neighborhood"],
                    "description": "trace=follow causal synapses (CAUSED_BY/LEADS_TO), "
                    "sequence=follow temporal synapses (BEFORE/AFTER), "
                    "temporal_range=fibers in a time window (requires start, end), "
                    "temporal_neighborhood=fibers near a fiber in time (requires fiber_id)",
                },
                "neuron_id": {
                    "type": "string",
                    "description": "Starting neuron ID (required for trace/sequence)",
                },
                "direction": {
                    "type": "string",
                    "description": "For trace: 'causes' (what caused this) or 'effects' (what this caused). "
                    "For sequence: 'forward' (what happened next) or 'backward' (what happened before). "
                    "Defaults: trace→'causes', sequence→'forward'.",
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum traversal depth for trace/sequence (default: 5)",
                },
                "start": {
                    "type": "string",
                    "description": "ISO-8601 start datetime for temporal_range "
                    "(e.g., '2026-04-01T00:00:00'). Required for temporal_range.",
                },
                "end": {
                    "type": "string",
                    "description": "ISO-8601 end datetime for temporal_range. Required for temporal_range.",
                },
                "fiber_id": {
                    "type": "string",
                    "description": "Anchor fiber ID for temporal_neighborhood. Required for temporal_neighborhood.",
                },
                "window_hours": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 8760,
                    "description": "Hours before/after anchor fiber to search "
                    "(temporal_neighborhood only, default: 24)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum fibers to return for temporal queries (default: 50)",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_cache",
        "description": "Manage activation cache for warm-start recall. "
        "Cache saves neuron activation states at session end for faster recall on restart. "
        "Auto-saved on MCP shutdown, auto-loaded on startup. Use status to check hit rate.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "clear", "save", "load"],
                    "description": "status=show cache stats (hit rate, entries, age), "
                    "clear=invalidate cache (use after brain modifications), "
                    "save=force snapshot current activations, "
                    "load=force restore cached states",
                },
            },
            "required": ["action"],
        },
    },
]
