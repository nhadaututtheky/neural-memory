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
            "description": "Index codebase into neural memory for code-aware recall. Scans source files (Python, JS/TS, Go, Rust, Java, C/C++) and creates neurons for functions, classes, imports. Enables 'where is X implemented?' queries.",
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
            "description": "Save project context, decisions, and instructions into neural memory for cross-session persistence. All data is stored in the neural graph and discoverable by recall.",
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
        {
            "name": "nmem_health",
            "description": "Get brain health diagnostics including purity score, grade, component metrics, and actionable warnings.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "nmem_evolution",
            "description": "Measure brain evolution dynamics: maturation progress, learning plasticity, topology coherence, and proficiency level. Shows how the brain has evolved through usage.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "nmem_habits",
            "description": "Manage learned workflow habits. Suggest next actions, list learned habits, or clear habit data.",
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
            "description": "Brain version control — create snapshots, list versions, rollback, and diff. Use to checkpoint brain state before changes or restore previous states.",
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
            "description": "Transplant memories from one brain to another. Extract a filtered subgraph (by tags, memory types) and merge into the current brain.",
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
            "description": "View and manage memory conflicts. List detected contradictions, resolve them manually, or pre-check content for potential conflicts before storing.",
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
            "description": "Train a brain from documentation files. Parses markdown into semantic chunks, encodes through the NLP pipeline, builds heading hierarchy, and runs consolidation. Use to create expert domain brains.",
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
                            "enum": [".md", ".mdx", ".txt", ".rst"],
                        },
                        "description": "File extensions to include (default: ['.md'])",
                    },
                    "consolidate": {
                        "type": "boolean",
                        "description": "Run ENRICH consolidation after encoding (default: true)",
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "nmem_train_db",
            "description": "Train a brain from database schema. Extracts table structures, relationships, and patterns as knowledge — NOT raw data rows. Enables schema-aware recall for database understanding.",
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
            "description": "View and manage brain health alerts. Alerts are created automatically from health checks and persist across sessions. Use 'list' to see active alerts, 'acknowledge' to mark as handled.",
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
    ]
