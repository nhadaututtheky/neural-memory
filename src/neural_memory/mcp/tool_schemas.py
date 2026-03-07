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


def _with_parameters_alias(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``parameters`` as an alias for ``inputSchema`` on each tool.

    MCP clients read ``inputSchema``, but OpenAI-compatible bridges
    (Cursor, LiteLLM, etc.) read ``parameters``.  Including both keys
    prevents HTTP 400 errors when tools are forwarded to OpenAI API.
    """
    out: list[dict[str, Any]] = []
    for tool in schemas:
        t = {**tool}
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
        "description": "Store a memory. Auto-detects type if not specified. "
        "Error resolution: when a new memory contradicts a stored error (type='error'), "
        "the system automatically creates a RESOLVED_BY synapse and demotes the error's "
        "activation by >=50%, so the agent stops repeating outdated errors. Detection is "
        "automatic via tag overlap (>50%) and factual contradiction patterns — no manual "
        "tagging needed. Sensitive content is auto-encrypted when encryption is enabled, "
        "instead of being rejected.",
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
            },
            "required": ["content"],
        },
    },
    {
        "name": "nmem_recall",
        "description": "Query memories by semantic search with confidence ranking.",
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
            },
            "required": ["query"],
        },
    },
    {
        "name": "nmem_context",
        "description": "Get recent memories as context.",
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
            },
        },
    },
    {
        "name": "nmem_todo",
        "description": "Add a TODO memory (30-day expiry).",
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
        "description": "Brain stats: memory counts and freshness.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "nmem_auto",
        "description": "Auto-capture memories from text. 'process' analyzes+saves, 'flush' for emergency capture.",
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
        "description": "Autocomplete suggestions from brain neurons. "
        "When called with no prefix, returns idle neurons that have never been "
        "accessed — useful for discovering neglected knowledge that needs reinforcement.",
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
        "description": "Track session state: task, feature, progress.",
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
        "description": "Index codebase for code-aware recall.",
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
        "description": "Import from external systems (ChromaDB, Mem0, Cognee, etc.).",
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
        "description": "Save project context, decisions, instructions for cross-session persistence.",
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
        "description": "Load saved project context, decisions, and progress.",
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
        "description": "Brain health: purity score, grade, warnings.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "nmem_evolution",
        "description": "Brain evolution: maturation, plasticity, coherence.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "nmem_habits",
        "description": "Workflow habits: suggest, list, or clear.",
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
        "description": "Brain version control: snapshot, list, rollback, diff.",
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
        "description": "Transplant memories between brains by tags/types.",
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
        "description": "Memory conflicts: list, resolve, or pre-check.",
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
        "description": "Train brain from documentation files. "
        "Supports PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV (requires: pip install neural-memory[extract]). "
        "Trained memories are pinned by default (no decay, no compression, permanent KB).",
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
        "description": "Pin or unpin memories. Pinned memories skip decay, pruning, and compression — "
        "use for permanent knowledge base content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fiber_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fiber IDs to pin or unpin",
                },
                "pinned": {
                    "type": "boolean",
                    "description": "true to pin, false to unpin (default: true)",
                },
            },
            "required": ["fiber_ids"],
        },
    },
    {
        "name": "nmem_train_db",
        "description": "Train brain from database schema.",
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
        "description": "Brain health alerts: list or acknowledge.",
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
        "description": "Generate narratives: timeline, topic, or causal chain.",
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
        "name": "nmem_review",
        "description": "Spaced repetition reviews (Leitner box system).",
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
        "description": "Trigger manual sync with hub server.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["push", "pull", "full"],
                    "description": "push=send local changes, pull=get remote changes, full=bidirectional sync",
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
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_sync_status",
        "description": "Show sync status: pending changes, devices, last sync.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "nmem_sync_config",
        "description": "View or update sync configuration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set"],
                    "description": "get=view current config, set=update config",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Enable/disable sync",
                },
                "hub_url": {
                    "type": "string",
                    "description": "Hub server URL",
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
            },
            "required": ["action"],
        },
    },
    {
        "name": "nmem_telegram_backup",
        "description": "Send brain database file as backup to Telegram.",
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
        "description": "Create, list, or inspect hypotheses — evolving beliefs with Bayesian "
        "confidence tracking. Hypotheses auto-resolve when evidence is strong enough "
        "(confirmed at >=0.9 confidence with >=3 evidence-for, refuted at <=0.1 with "
        ">=3 evidence-against). Use nmem_evidence to add supporting/opposing evidence.",
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
        "description": "Add evidence for or against a hypothesis. Updates confidence via "
        "Bayesian update with surprise weighting and diminishing returns. "
        "Auto-resolves hypothesis when evidence threshold is met.",
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
        "description": "Create, list, or inspect predictions — falsifiable claims about future "
        "observations. Predictions track confidence, optional deadlines, and can link "
        "to hypotheses via PREDICTED synapse. Verified predictions propagate evidence "
        "back to linked hypotheses. Use nmem_verify to record outcomes.",
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
        "description": "Verify a prediction as correct or wrong. Optionally records an observation, "
        "creates VERIFIED_BY or FALSIFIED_BY synapse, and propagates evidence to "
        "linked hypotheses. Returns updated calibration score.",
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
        "description": "Cognitive overview — O(1) summary of active hypotheses, pending predictions, "
        "calibration score, and knowledge gaps. Use 'summary' for instant dashboard, "
        "'refresh' to recompute scores from current state.",
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
        "description": "Metacognition — track what the brain doesn't know. Detect knowledge gaps "
        "from contradictions, low-confidence hypotheses, recall misses, or manual flagging. "
        "Resolve gaps when new information fills them.",
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
        "description": "Schema evolution — evolve hypotheses into new versions. "
        "Creates a version chain via SUPERSEDES synapse so the brain tracks how beliefs changed over time. "
        "Use when a hypothesis needs updating with new understanding.",
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
        "description": "Find and explain the shortest path between two entities in the neural graph. Returns a step-by-step explanation with synapse types, weights, and supporting memory evidence.",
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
        "description": "Edit an existing memory's type, content, or priority. "
        "Use when a memory was auto-typed incorrectly or needs content correction. "
        "Preserves all connections (synapses) and fiber associations.",
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
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "nmem_forget",
        "description": "Explicitly delete or close a specific memory. "
        "Soft delete by default (marks as expired). Use hard=true for permanent removal. "
        "Use for closing completed TODOs or removing outdated/incorrect memories.",
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
]
