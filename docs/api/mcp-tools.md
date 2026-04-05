# MCP Tools Reference

Complete reference for all NeuralMemory MCP tools.
**56 tools** available via MCP stdio transport.

!!! tip
    Tools are called as MCP tool calls, not CLI commands. In Claude Code, call `nmem_recall` directly — do not run `nmem recall` in terminal.

## Table of Contents

- [Core Memory](#core)
  - [`nmem_remember`](#nmem_remember)
  - [`nmem_remember_batch`](#nmem_remember_batch)
  - [`nmem_recall`](#nmem_recall)
  - [`nmem_show`](#nmem_show)
  - [`nmem_context`](#nmem_context)
  - [`nmem_todo`](#nmem_todo)
  - [`nmem_auto`](#nmem_auto)
  - [`nmem_suggest`](#nmem_suggest)
- [Session & Context](#session)
  - [`nmem_session`](#nmem_session)
  - [`nmem_eternal`](#nmem_eternal)
  - [`nmem_recap`](#nmem_recap)
- [Provenance & Sources](#provenance)
  - [`nmem_provenance`](#nmem_provenance)
  - [`nmem_source`](#nmem_source)
- [Analytics & Health](#analytics)
  - [`nmem_stats`](#nmem_stats)
  - [`nmem_health`](#nmem_health)
  - [`nmem_evolution`](#nmem_evolution)
  - [`nmem_habits`](#nmem_habits)
  - [`nmem_narrative`](#nmem_narrative)
- [Cognitive Reasoning](#cognitive)
  - [`nmem_hypothesize`](#nmem_hypothesize)
  - [`nmem_evidence`](#nmem_evidence)
  - [`nmem_predict`](#nmem_predict)
  - [`nmem_verify`](#nmem_verify)
  - [`nmem_cognitive`](#nmem_cognitive)
  - [`nmem_gaps`](#nmem_gaps)
  - [`nmem_schema`](#nmem_schema)
  - [`nmem_explain`](#nmem_explain)
- [Training & Import](#training)
  - [`nmem_train`](#nmem_train)
  - [`nmem_train_db`](#nmem_train_db)
  - [`nmem_index`](#nmem_index)
  - [`nmem_import`](#nmem_import)
- [Memory Management](#management)
  - [`nmem_edit`](#nmem_edit)
  - [`nmem_forget`](#nmem_forget)
  - [`nmem_pin`](#nmem_pin)
  - [`nmem_consolidate`](#nmem_consolidate)
  - [`nmem_drift`](#nmem_drift)
  - [`nmem_review`](#nmem_review)
  - [`nmem_alerts`](#nmem_alerts)
- [Cloud Sync & Backup](#sync)
  - [`nmem_sync`](#nmem_sync)
  - [`nmem_sync_status`](#nmem_sync_status)
  - [`nmem_sync_config`](#nmem_sync_config)
  - [`nmem_telegram_backup`](#nmem_telegram_backup)
- [Versioning & Transfer](#meta)
  - [`nmem_version`](#nmem_version)
  - [`nmem_transplant`](#nmem_transplant)
  - [`nmem_conflicts`](#nmem_conflicts)
- [Other](#other)
  - [`nmem_visualize`](#nmem_visualize)
  - [`nmem_watch`](#nmem_watch)
  - [`nmem_surface`](#nmem_surface)
  - [`nmem_tool_stats`](#nmem_tool_stats)
  - [`nmem_lifecycle`](#nmem_lifecycle)
  - [`nmem_refine`](#nmem_refine)
  - [`nmem_report_outcome`](#nmem_report_outcome)
  - [`nmem_budget`](#nmem_budget)
  - [`nmem_tier`](#nmem_tier)
  - [`nmem_boundaries`](#nmem_boundaries)
  - [`nmem_milestone`](#nmem_milestone)
  - [`nmem_store`](#nmem_store)

---

## Core Memory {#core}

### `nmem_remember`

Store a memory. Auto-detects type, auto-resolves contradicted errors (RESOLVED_BY synapse). Use after completing a task, fixing a bug, or making a decision. Don't use for temporary notes (use ephemeral=true) or project context (use nmem_eternal).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | — | The content to remember |
| `type` | string (`fact`, `decision`, `preference`, `todo`, `insight`, `context`, `instruction`, `error`, `workflow`, `reference`, `boundary`) | No | — | Memory type (auto-detected if not specified) |
| `tier` | string (`hot`, `warm`, `cold`) | No | — | Memory tier: hot (always in context, slow decay), warm (default, semantic match), cold (explicit recall only, fast de... |
| `domain` | string | No | — | Domain scope for boundary memories (e.g. 'financial', 'security', 'code-review'). Adds a domain:{value} tag. Boundari... |
| `priority` | integer | No | — | Priority 0-10 (5=normal, 10=critical) |
| `tags` | array[string] | No | — | Tags for categorization |
| `expires_days` | integer | No | — | Days until memory expires |
| `encrypted` | boolean | No | default: false | Force encrypt this memory's neuron content (default: false). When true, content is encrypted with the brain's Fernet ... |
| `event_at` | string | No | — | ISO datetime of when the event originally occurred (e.g. '2026-03-02T08:00:00'). Defaults to current time if not prov... |
| `trust_score` | number | No | — | Trust level 0.0-1.0. Capped by source ceiling (user_input max 0.9, ai_inference max 0.7). NULL = unscored. |
| `source_id` | string | No | — | Link this memory to a registered source. Creates a SOURCE_OF synapse for provenance tracking. |
| `context` | object | No | — | Structured context dict merged into content server-side using type-specific templates. Keys like 'reason', 'alternati... |
| `ephemeral` | boolean | No | — | Session-scoped memory: auto-expires after TTL (default 24h), never synced to cloud, excluded from consolidation. Use ... |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_remember_batch`

Store multiple memories at once (max 20). Use when saving 3+ memories together. Partial success — one bad item won't block the rest.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memories` | array[object] | Yes | — | Array of memories to store (max 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_recall`

Query memories via spreading activation. Use when you need past context, decisions, or knowledge. Depth: 0=instant lookup, 1=context (default), 2=cross-time patterns, 3=deep graph. Add tags for precision. Use nmem_context instead for broad recent context.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | The query to search memories |
| `depth` | integer | No | — | Search depth: 0=instant (direct lookup, 1 hop), 1=context (spreading activation, 3 hops), 2=habit (cross-time pattern... |
| `max_tokens` | integer | No | default: 500 | Maximum tokens in response (default: 500) |
| `min_confidence` | number | No | — | Minimum confidence threshold |
| `valid_at` | string | No | — | ISO datetime string to filter memories valid at that point in time (e.g. '2026-02-01T12:00:00') |
| `include_conflicts` | boolean | No | default: false | Include full conflict details in response (default: false). When false, only has_conflicts flag and conflict_count ar... |
| `warn_expiry_days` | integer | No | — | If set, warn about memories expiring within this many days. Adds expiry_warnings to response. |
| `brains` | array[string] | No | — | Optional list of brain names to query across (max 5). When provided, runs parallel recall across all specified brains... |
| `min_trust` | number | No | — | Filter: only return memories with trust_score >= this value. Unscored memories (NULL) are always included. |
| `tags` | array[string] | No | — | Filter by tags. Checks tags, auto_tags, and agent_tags columns. |
| `tag_mode` | string (`and`, `or`) | No | — | Tag matching mode: 'and' (default, all tags must match) or 'or' (any tag matches). |
| `mode` | string (`associative`, `exact`) | No | — | Recall mode: 'associative' (default) returns formatted context, 'exact' returns raw neuron contents verbatim without ... |
| `include_citations` | boolean | No | default: true | Include citation and audit trail in exact recall results (default: true). |
| `recall_token_budget` | integer | No | — | When set, activates budget-aware fiber selection: ranks fibers by value-per-token and selects the most efficient ones... |
| `permanent_only` | boolean | No | — | Exclude ephemeral (session-scoped) memories from results. Default: false (include all). |
| `clean_for_prompt` | boolean | No | — | Return clean bullet-point text without section headers or neuron-type tags. Use when injecting recall output into pro... |
| `tier` | string (`hot`, `warm`, `cold`) | No | — | Filter results by memory tier. Only return memories matching this tier. |
| `domain` | string | No | — | Domain scope filter. When set, HOT context injection only includes boundaries tagged with this domain (plus unscoped ... |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_show`

Get full verbatim content + metadata + synapses for a memory by ID. Use after recall when you need exact content, not the summarized version.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | — | The fiber_id or neuron_id of the memory to retrieve |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_context`

Get recent memories as auto-injected context. Use for broad task context. For specific queries use nmem_recall. For project-level context use nmem_recap.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | default: 10 | Number of recent memories (default: 10) |
| `fresh_only` | boolean | No | — | Only include memories < 30 days old |
| `warn_expiry_days` | integer | No | — | If set, warn about memories expiring within this many days. Adds expiry_warnings to response. |
| `include_ghosts` | boolean | No | default: true | Include faded ghost memories at bottom of context with recall keys (default: true). Set false to suppress. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_todo`

Quick TODO memory (auto-expires in 30 days). Use nmem_forget to close when done.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | Yes | — | The task to remember |
| `priority` | integer | No | default: 5 | Priority 0-10 (default: 5) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_auto`

Auto-extract memories from text. 'process'=analyze+save, 'flush'=emergency capture before compaction. Use at session end or when processing large text blocks.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`status`, `enable`, `disable`, `analyze`, `process`, `flush`) | Yes | — | Action: 'process' analyzes and saves, 'analyze' only detects, 'flush' emergency capture before compaction (skips dedu... |
| `text` | string | No | — | Text to analyze (required for 'analyze' and 'process') |
| `save` | boolean | No | — | Force save even if auto-capture disabled (for 'analyze') |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_suggest`

Autocomplete from brain neurons. No prefix = idle/neglected neurons needing reinforcement.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prefix` | string | No | — | The prefix text to autocomplete |
| `limit` | integer | No | default: 5 | Max suggestions (default: 5) |
| `type_filter` | string (`time`, `spatial`, `entity`, `action`, `state`, `concept`, `sensory`, `intent`) | No | — | Filter by neuron type |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Session & Context {#session}

### `nmem_session`

Track current session state (task, feature, progress). Single-session only. For cross-session persistence use nmem_eternal.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`get`, `set`, `end`) | Yes | — | get=load current session, set=update session state, end=close session |
| `feature` | string | No | — | Current feature being worked on |
| `task` | string | No | — | Current specific task |
| `progress` | number | No | — | Progress 0.0 to 1.0 |
| `notes` | string | No | — | Additional context notes |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_eternal`

SAVE project context, decisions, instructions that persist across sessions. Pair with nmem_recap to LOAD. Use for project-level facts, not task-specific memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`status`, `save`) | Yes | — | status=view memory counts and session state, save=store project context/decisions/instructions |
| `project_name` | string | No | — | Set project name (saved as FACT) |
| `tech_stack` | array[string] | No | — | Set tech stack (saved as FACT) |
| `decision` | string | No | — | Add a key decision (saved as DECISION) |
| `reason` | string | No | — | Reason for the decision |
| `instruction` | string | No | — | Add a persistent instruction (saved as INSTRUCTION) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_recap`

LOAD project context saved by nmem_eternal. Call at SESSION START to restore cross-session state. Level 1=quick (~500 tokens), 2=detailed, 3=full.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `level` | integer | No | — | Detail level: 1=quick (~500 tokens), 2=detailed (~1300 tokens), 3=full (~3300 tokens). Default: 1 |
| `topic` | string | No | — | Search for a specific topic in context (e.g., 'auth', 'database') |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Provenance & Sources {#provenance}

### `nmem_provenance`

Trace or audit a memory's origin chain. Use when verifying where a fact came from or adding verification/approval stamps.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`trace`, `verify`, `approve`) | Yes | — | Action: trace (view chain), verify (mark verified), approve (mark approved). |
| `neuron_id` | string | Yes | — | Neuron ID to trace/verify/approve. |
| `actor` | string | No | default: mcp_agent | Who is performing the verification/approval (default: mcp_agent). |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_source`

Register external sources (docs, laws, APIs) for provenance tracking. Use before nmem_train to link trained memories to their origin.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`register`, `list`, `get`, `update`, `delete`) | Yes | — | Action to perform on sources. |
| `source_id` | string | No | — | Source ID (required for get/update/delete). |
| `name` | string | No | — | Source name (required for register). |
| `source_type` | string (`law`, `contract`, `ledger`, `document`, `api`, `manual`, `website`, `book`, `research`) | No | default: document | Type of source (default: document). |
| `version` | string | No | — | Version string (e.g. '2024-01', 'v2.0'). |
| `status` | string (`active`, `superseded`, `repealed`, `draft`) | No | — | Source lifecycle status. |
| `file_hash` | string | No | — | File hash for integrity checking. |
| `metadata` | object | No | — | Additional metadata. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Analytics & Health {#analytics}

### `nmem_stats`

Quick brain stats: counts and freshness. For quality assessment use nmem_health instead.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_health`

Primary health check — purity score, grade, warnings. Call FIRST, then fix top penalty. For specific alerts use nmem_alerts. For trends use nmem_evolution.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_evolution`

Long-term brain growth trends: maturation, plasticity, coherence. Use for trend analysis, not immediate health (use nmem_health for that).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_habits`

Learned workflow habits from tool usage patterns. Suggest next action, list habits, or clear.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`suggest`, `list`, `clear`) | Yes | — | suggest=get next action suggestions, list=show learned habits, clear=remove all habits |
| `current_action` | string | No | — | Current action type for suggestions (required for suggest action) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_narrative`

Generate memory narratives: timeline (date range), topic (spreading activation), or causal chain. Use to understand how knowledge connects.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`timeline`, `topic`, `causal`) | Yes | — | timeline=date-range narrative, topic=SA-driven topic narrative, causal=causal chain narrative |
| `topic` | string | No | — | Topic to explore (required for topic and causal actions) |
| `start_date` | string | No | — | Start date in ISO format (required for timeline, e.g., '2026-02-01') |
| `end_date` | string | No | — | End date in ISO format (required for timeline, e.g., '2026-02-18') |
| `max_fibers` | integer | No | default: 20 | Max fibers in narrative (default: 20) |
| `max_depth` | integer | No | default: 5, for causal action only | Max causal chain depth (default: 5, for causal action only) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Cognitive Reasoning {#cognitive}

### `nmem_hypothesize`

Create or inspect hypotheses (Bayesian confidence). Cognitive workflow: hypothesize -> evidence -> predict -> verify -> cognitive (dashboard). Auto-resolves at >=0.9 (confirmed) or <=0.1 (refuted) with 3+ evidence.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`create`, `list`, `get`) | Yes | — | create=new hypothesis, list=show all, get=detail view |
| `content` | string | No | — | Hypothesis statement (required for create) |
| `confidence` | number | No | default: 0.5 | Initial confidence level (default: 0.5) |
| `tags` | array[string] | No | — | Tags for categorization |
| `priority` | integer | No | default: 6 | Priority 0-10 (default: 6) |
| `hypothesis_id` | string | No | — | Hypothesis neuron ID (required for get) |
| `status` | string (`active`, `confirmed`, `refuted`, `superseded`, `pending`, `expired`) | No | — | Filter by status (for list action) |
| `limit` | integer | No | default: 20 | Max results for list (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_evidence`

Add evidence for/against a hypothesis. Bayesian confidence update with auto-resolve. Requires an existing hypothesis_id from nmem_hypothesize.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `hypothesis_id` | string | Yes | — | Target hypothesis neuron ID |
| `content` | string | Yes | — | Evidence content — what was observed/discovered |
| `type` | string (`for`, `against`) | Yes | — | Evidence direction: 'for' supports, 'against' weakens |
| `weight` | number | No | default: 0.5 | Evidence strength (default: 0.5). Higher = stronger evidence |
| `tags` | array[string] | No | — | Tags for the evidence memory |
| `priority` | integer | No | default: 5 | Priority 0-10 (default: 5) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_predict`

Create falsifiable predictions linked to hypotheses. Use nmem_verify to record outcomes (propagates evidence back to hypothesis).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`create`, `list`, `get`) | Yes | — | create=new prediction, list=show all, get=detail view |
| `content` | string | No | — | Prediction statement (required for create) |
| `confidence` | number | No | default: 0.7 | How confident you are in this prediction (default: 0.7) |
| `deadline` | string | No | — | ISO datetime deadline for verification (e.g. '2026-04-01T00:00:00') |
| `hypothesis_id` | string | No | — | Link prediction to a hypothesis (creates PREDICTED synapse) |
| `tags` | array[string] | No | — | Tags for categorization |
| `priority` | integer | No | default: 5 | Priority 0-10 (default: 5) |
| `prediction_id` | string | No | — | Prediction neuron ID (required for get) |
| `status` | string (`active`, `confirmed`, `refuted`, `superseded`, `pending`, `expired`) | No | — | Filter by status (for list action) |
| `limit` | integer | No | default: 20 | Max results for list (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_verify`

Record prediction outcome (correct/wrong). Propagates evidence to linked hypothesis. Use after observing the predicted event.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prediction_id` | string | Yes | — | Target prediction neuron ID |
| `outcome` | string (`correct`, `wrong`) | Yes | — | Whether the prediction was correct or wrong |
| `content` | string | No | — | Observation content — what actually happened (optional) |
| `tags` | array[string] | No | — | Tags for the observation memory |
| `priority` | integer | No | default: 5 | Priority 0-10 (default: 5) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_cognitive`

Cognitive dashboard — instant O(1) summary of hypotheses, predictions, calibration, and gaps. Use as overview after cognitive workflow steps.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`summary`, `refresh`) | Yes | — | summary=get current hot index, refresh=recompute scores |
| `limit` | integer | No | default: 10, for summary | Max hot items to return (default: 10, for summary) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_gaps`

Track what the brain doesn't know. Detect gaps from contradictions, low-confidence, or recall misses. Resolve when new info fills them.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`detect`, `list`, `resolve`, `get`) | Yes | — | detect=flag new gap, list=show gaps, resolve=mark filled, get=detail |
| `topic` | string | No | — | What knowledge is missing (required for detect) |
| `source` | string (`contradicting_evidence`, `low_confidence_hypothesis`, `user_flagged`, `recall_miss`, `stale_schema`) | No | default: user_flagged | How the gap was detected (default: user_flagged) |
| `priority` | number | No | — | Gap priority (auto-set from source if not provided) |
| `related_neuron_ids` | array[string] | No | — | Neuron IDs related to this gap (max 10) |
| `gap_id` | string | No | — | Gap ID (required for resolve and get) |
| `resolved_by_neuron_id` | string | No | — | Neuron that resolved the gap (optional for resolve) |
| `include_resolved` | boolean | No | default: false | Include resolved gaps in list (default: false) |
| `limit` | integer | No | default: 20 | Max results for list (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_schema`

Evolve a hypothesis into a new version (SUPERSEDES chain). Use when understanding changes — preserves belief evolution history.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`evolve`, `history`, `compare`) | Yes | — | evolve=create new version, history=version chain, compare=diff two versions |
| `hypothesis_id` | string | Yes | — | Neuron ID of the hypothesis to evolve or inspect |
| `content` | string | No | — | Updated content for the new version (required for evolve) |
| `confidence` | number | No | — | Initial confidence for the new version (inherits from old if not set) |
| `reason` | string | No | — | Why the hypothesis is being evolved (stored as synapse metadata) |
| `other_id` | string | No | — | Second hypothesis ID for compare action |
| `tags` | array[string] | No | — | Tags for the new version |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_explain`

Explain how two concepts connect in the neural graph (shortest path with synapse types and weights). Use to understand relationships between entities.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `from_entity` | string | Yes | — | Source entity name to start from (e.g. 'React', 'authentication') |
| `to_entity` | string | Yes | — | Target entity name to reach (e.g. 'performance', 'JWT') |
| `max_hops` | integer | No | default: 6 | Maximum path length (default: 6) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Training & Import {#training}

### `nmem_train`

Train brain from docs (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV). Pinned by default as permanent KB. Requires: pip install neural-memory[extract].

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`train`, `status`) | Yes | — | train=process docs into brain, status=show training stats |
| `path` | string | No | default: current directory | Directory or file path to train from (default: current directory) |
| `domain_tag` | string | No | — | Domain tag for all chunks (e.g., 'react', 'kubernetes') |
| `brain_name` | string | No | default: current brain | Target brain name (default: current brain) |
| `extensions` | array[string] | No | default: ['.md'] | File extensions to include (default: ['.md']). Rich formats (PDF, DOCX, PPTX, HTML, XLSX) require: pip install neural... |
| `consolidate` | boolean | No | default: true | Run ENRICH consolidation after encoding (default: true) |
| `pinned` | boolean | No | default: true | Pin trained memories as permanent KB — skip decay/prune/compress (default: true) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_train_db`

Train brain from database schema (tables, columns, relationships). SQLite supported.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`train`, `status`) | Yes | — | train=extract schema into brain, status=show training stats |
| `connection_string` | string | No | — | Database connection string (v1: sqlite:///path/to/db.db) |
| `domain_tag` | string | No | — | Domain tag for schema knowledge (e.g., 'ecommerce', 'analytics') |
| `brain_name` | string | No | default: current brain | Target brain name (default: current brain) |
| `consolidate` | boolean | No | default: true | Run ENRICH consolidation after encoding (default: true) |
| `max_tables` | integer | No | default: 100 | Maximum tables to process (default: 100) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_index`

Index codebase for code-aware recall. Extracts symbols, imports, and relationships. Run once per project, re-scan after major changes.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`scan`, `status`) | Yes | — | scan=index codebase, status=show what's indexed |
| `path` | string | No | default: current working directory | Directory to index (default: current working directory) |
| `extensions` | array[string] | No | default: [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cc"] | File extensions to index (default: [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".h", ".... |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_import`

Import memories from external systems (ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex). One-time migration tool.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | string (`chromadb`, `mem0`, `awf`, `cognee`, `graphiti`, `llamaindex`) | Yes | — | Source system to import from |
| `connection` | string | No | — | Connection string/path (e.g., '/path/to/chroma', graph URI, or index dir path). For API keys, prefer env vars: MEM0_A... |
| `collection` | string | No | — | Collection/namespace to import from |
| `limit` | integer | No | — | Maximum records to import |
| `user_id` | string | No | — | User ID filter (for Mem0) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Memory Management {#management}

### `nmem_edit`

Edit a memory's type, content, priority, or tier. Preserves all synapses. Use when auto-typing was wrong or content needs correction. For complete replacement, forget+remember.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | — | The fiber ID or neuron ID of the memory to edit |
| `type` | string (`fact`, `decision`, `preference`, `todo`, `insight`, `context`, `instruction`, `error`, `workflow`, `reference`, `boundary`) | No | — | New memory type |
| `content` | string | No | — | New content for the anchor neuron |
| `priority` | integer | No | — | New priority (0-10) |
| `tier` | string (`hot`, `warm`, `cold`) | No | — | New memory tier: hot, warm, or cold |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_forget`

Delete a memory. Soft delete by default; hard=true for permanent removal. Use to close completed TODOs or remove outdated memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | — | The fiber ID of the memory to forget |
| `hard` | boolean | No | default: false = soft delete | Permanent deletion with cascade cleanup (default: false = soft delete) |
| `reason` | string | No | — | Why this memory is being forgotten (stored in logs) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_pin`

Pin memories as permanent KB (skip decay/pruning/compression). Use for critical knowledge.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`pin`, `unpin`, `list`) | No | — | Action: pin (default), unpin, or list pinned memories |
| `fiber_ids` | array[string] | No | — | Fiber IDs to pin or unpin (required for pin/unpin, ignored for list) |
| `limit` | integer | No | default: 50, max: 200 | Max results for list action (default: 50, max: 200) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_consolidate`

Run brain consolidation (sleep-like maintenance). Strategy 'all' runs everything in order. Use periodically or after bulk imports. dry_run=true to preview.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `strategy` | string (`prune`, `merge`, `summarize`, `mature`, `infer`, `enrich`, `dream`, `learn_habits`, `dedup`, `semantic_link`, `compress`, `process_tool_events`, `detect_drift`, `all`) | No | default: all | Consolidation strategy to run (default: all) |
| `dry_run` | boolean | No | default: false | Preview changes without applying (default: false) |
| `prune_weight_threshold` | number | No | default: 0.05 | Synapse weight threshold for pruning (default: 0.05) |
| `merge_overlap_threshold` | number | No | default: 0.5 | Jaccard overlap threshold for merging fibers (default: 0.5) |
| `prune_min_inactive_days` | number | No | default: 7.0 | Grace period in days before pruning inactive synapses (default: 7.0) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_drift`

Find tags that mean the same thing (Jaccard similarity). Detect clusters, then merge/alias/dismiss.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`detect`, `list`, `merge`, `alias`, `dismiss`) | Yes | — | detect=run drift analysis, list=show existing clusters, merge/alias/dismiss=resolve a specific cluster |
| `cluster_id` | string | No | — | Cluster ID to resolve (required for merge/alias/dismiss) |
| `status` | string (`detected`, `merged`, `aliased`, `dismissed`) | No | — | Filter clusters by status (for list action) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_review`

Spaced repetition reviews (Leitner 5-box system). Queue due reviews, mark success/fail, view stats.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`queue`, `mark`, `schedule`, `stats`) | Yes | — | queue=get due reviews, mark=record review result, schedule=manually schedule a fiber, stats=review statistics |
| `fiber_id` | string | No | — | Fiber ID (required for mark and schedule actions) |
| `success` | boolean | No | — | Whether recall was successful (for mark action, default: true) |
| `limit` | integer | No | default: 20 | Max items in queue (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_alerts`

Actionable health alerts. Call after nmem_health to see specific issues. Acknowledge alerts after fixing them.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`list`, `acknowledge`) | Yes | — | list=view active/seen alerts, acknowledge=mark alert as handled |
| `alert_id` | string | No | — | Alert ID to acknowledge (required for acknowledge action) |
| `limit` | integer | No | default: 50 | Max alerts to list (default: 50) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Cloud Sync & Backup {#sync}

### `nmem_sync`

Manual sync with cloud hub. Push local changes, pull remote, or full bidirectional sync.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`push`, `pull`, `full`, `seed`) | Yes | — | push=send local changes, pull=get remote changes, full=bidirectional sync, seed=populate change log from existing dat... |
| `hub_url` | string | No | — | Hub server URL (overrides config). Must be http:// or https:// |
| `strategy` | string (`prefer_recent`, `prefer_local`, `prefer_remote`, `prefer_stronger`) | No | default: from config | Conflict resolution strategy (default: from config) |
| `api_key` | string | No | default: from config | API key override (default: from config) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_sync_status`

View sync status: pending changes, connected devices, last sync time.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_sync_config`

Configure sync: setup (onboarding), activate (license key), get/set settings.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`get`, `set`, `setup`, `activate`) | Yes | — | get=view config, set=update config, setup=guided onboarding, activate=activate purchased license key |
| `enabled` | boolean | No | — | Enable/disable sync |
| `hub_url` | string | No | default: cloud hub | Hub server URL (default: cloud hub) |
| `api_key` | string | No | — | API key for cloud hub (starts with nmk_) |
| `auto_sync` | boolean | No | — | Enable/disable auto-sync |
| `sync_interval_seconds` | integer | No | — | Sync interval in seconds |
| `conflict_strategy` | string (`prefer_recent`, `prefer_local`, `prefer_remote`, `prefer_stronger`) | No | — | Default conflict strategy |
| `license_key` | string | No | — | NM license key to activate (for action='activate', starts with nm_) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_telegram_backup`

Backup brain database to Telegram. Requires Telegram bot config.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `brain_name` | string | No | default: active brain | Brain name to backup (default: active brain) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Versioning & Transfer {#meta}

### `nmem_version`

Brain version control: snapshot current state, rollback, or diff between versions. Use before risky consolidation or major changes.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`create`, `list`, `rollback`, `diff`) | Yes | — | create=snapshot current state, list=show versions, rollback=restore version, diff=compare versions |
| `name` | string | No | — | Version name (required for create) |
| `description` | string | No | — | Version description (optional for create) |
| `version_id` | string | No | — | Version ID (required for rollback) |
| `from_version` | string | No | — | Source version ID (required for diff) |
| `to_version` | string | No | — | Target version ID (required for diff) |
| `limit` | integer | No | default: 20 | Max versions to list (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_transplant`

Copy memories from another brain by tags/types. Use for sharing knowledge between project brains.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_brain` | string | Yes | — | Name of the source brain to extract from |
| `tags` | array[string] | No | — | Tags to filter — fibers matching ANY tag will be included |
| `memory_types` | array[string] | No | — | Memory types to filter (fact, decision, etc.) |
| `strategy` | string (`prefer_local`, `prefer_remote`, `prefer_recent`, `prefer_stronger`) | No | default: prefer_local | Conflict resolution strategy (default: prefer_local) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_conflicts`

Detect and resolve conflicting memories. Pre-check new content for contradictions before saving.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`list`, `resolve`, `check`) | Yes | — | list=view active conflicts, resolve=manually resolve a conflict, check=pre-check content for conflicts |
| `neuron_id` | string | No | — | Neuron ID of the disputed memory (required for resolve) |
| `resolution` | string (`keep_existing`, `keep_new`, `keep_both`) | No | — | How to resolve: keep_existing=undo dispute, keep_new=supersede old, keep_both=accept both |
| `content` | string | No | — | Content to pre-check for conflicts (required for check) |
| `tags` | array[string] | No | — | Optional tags for more accurate conflict checking |
| `limit` | integer | No | default: 50 | Max conflicts to list (default: 50) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Other {#other}

### `nmem_visualize`

Generate charts from memory data (Vega-Lite/markdown/ASCII). Use for financial metrics, trends, or any structured data in memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | What to visualize (e.g., 'ROE trend across quarters', 'revenue by product') |
| `chart_type` | string (`line`, `bar`, `pie`, `scatter`, `table`, `timeline`) | No | — | Chart type (auto-detected if omitted based on data shape) |
| `format` | string (`vega_lite`, `markdown_table`, `ascii`, `all`) | No | default: vega_lite | Output format (default: vega_lite) |
| `limit` | integer | No | default: 20 | Max data points (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_watch`

Watch directories for file changes, auto-ingest into memory. Scan for one-shot, start/stop for continuous monitoring.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`scan`, `start`, `stop`, `status`, `list`) | Yes | — | scan=one-shot ingest directory, start=background watch, stop=stop watching, status=show stats, list=tracked files |
| `directory` | string | No | — | Directory path to scan (for scan action) |
| `directories` | array[string] | No | — | List of directory paths to watch (for start action, max 10) |
| `status` | string (`active`, `deleted`) | No | — | Filter files by status (for list action) |
| `limit` | integer | No | default: 50, for list action | Max files to return (default: 50, for list action) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_surface`

Knowledge Surface (.nm file) — compact graph (~1000 tokens) loaded every session. Generate to rebuild, show to inspect.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`generate`, `show`) | No | — | generate=rebuild surface from brain.db, show=display current surface info |
| `token_budget` | integer | No | default: 1200 | Token budget for surface (default: 1200). Only used with generate action. |
| `max_graph_nodes` | integer | No | default: 30 | Max graph nodes to include (default: 30). Only used with generate action. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |

### `nmem_tool_stats`

Agent tool usage analytics: frequency, success rates, daily trends. Use to understand which NM tools are being used and how effectively.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`summary`, `daily`) | Yes | — | summary=top tools with success rates, daily=usage breakdown by day |
| `days` | integer | No | default: 30 | Time window in days (default: 30) |
| `limit` | integer | No | default: 20 | Max tools to return (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_lifecycle`

Manage memory lifecycle: view compression states, freeze/thaw individual memories, recover compressed content. Use when a memory was incorrectly compressed.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`status`, `recover`, `freeze`, `thaw`) | Yes | — | status=show lifecycle distribution, recover=rehydrate compressed memory, freeze=prevent compression, thaw=resume norm... |
| `id` | string | No | — | Neuron ID (required for recover/freeze/thaw). For recover, fiber_id is also accepted. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_refine`

Refine an instruction/workflow: update content, add failure modes, add trigger patterns. Use to improve instructions based on real-world execution outcomes.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `neuron_id` | string | Yes | — | ID of the instruction or workflow memory to refine (use the fiber_id returned by nmem_remember or nmem_recall). |
| `new_content` | string | No | — | Updated instruction text. Replaces current content and increments version. |
| `reason` | string | No | — | Why this refinement was made (stored in refinement_history for auditability). |
| `add_failure_mode` | string | No | — | Description of a failure mode to append to the failure_modes list (deduped, capped at 20). |
| `add_trigger` | string | No | — | Keyword or phrase to append to trigger_patterns (boosts recall when query overlaps, deduped, capped at 10). |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_report_outcome`

Report instruction execution outcome (success/fail). Builds track record — high success rate boosts recall priority. Call after executing an instruction.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `neuron_id` | string | Yes | — | ID of the instruction or workflow memory that was executed. |
| `success` | boolean | Yes | — | Whether execution succeeded. |
| `failure_description` | string | No | — | If failed, brief description of what went wrong (appended to failure_modes, deduped, capped at 20). |
| `context` | string | No | — | Optional context about the execution (stored for auditability). |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_budget`

Token budget analysis: estimate recall cost, profile brain token usage, find compression candidates. Use when context window is tight.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`estimate`, `analyze`, `optimize`) | Yes | — | Action: 'estimate' (dry-run recall cost), 'analyze' (brain token profile), 'optimize' (find compression candidates). |
| `query` | string | No | — | Query to estimate recall cost for (used with action='estimate'). |
| `max_tokens` | integer | No | default: 4000 | Token budget to estimate against (default: 4000). |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_tier`

Auto-tier: promote/demote memories between HOT/WARM/COLD by access patterns (Pro). Evaluate (dry-run) before apply. Free users: manual tiers only.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`status`, `evaluate`, `apply`, `history`, `config`, `analytics`) | Yes | — | Action: 'status' (distribution), 'evaluate' (dry-run), 'apply' (execute), 'history' (fiber tier log), 'config' (thres... |
| `fiber_id` | string | No | — | Fiber ID for 'history' action. |
| `dry_run` | boolean | No | default: false | If true with action='apply', show changes without applying (default: false). |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_boundaries`

View domain-scoped boundaries (safety rules, always HOT tier). List boundaries by domain or view domain summary.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`list`, `domains`) | No | — | Action: list (show boundaries, optionally filtered by domain), domains (list unique domains with boundary counts). De... |
| `domain` | string | No | — | Filter boundaries by domain (e.g. 'financial', 'security'). Only used with action=list. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_milestone`

Brain growth milestones (100, 250, 500...10K neurons). Check for new achievements, view progress to next, or generate growth report.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`check`, `progress`, `history`, `report`) | Yes | — | check=detect+record new milestones, progress=distance to next milestone, history=all recorded milestones, report=gene... |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_store`

Brain Store — browse, preview, import, and export community knowledge brains. Share curated brains with the community or import others' expertise.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`browse`, `preview`, `import`, `export`, `publish`) | Yes | — | browse=search community brain registry, preview=view brain details before import, import=download and import a brain,... |
| `brain_name` | string | No | — | Brain name in registry (required for preview/import) |
| `search` | string | No | — | Search query for browse (matches name, description, tags) |
| `category` | string (`programming`, `devops`, `writing`, `science`, `personal`, `security`, `data`, `design`, `general`) | No | — | Filter by category (browse) or set category (export) |
| `tag` | string | No | — | Filter by tag (browse only) |
| `sort_by` | string (`created_at`, `rating_avg`, `download_count`) | No | default: created_at | Sort order for browse results (default: created_at) |
| `limit` | integer | No | default: 20, max: 50 | Max results for browse (default: 20, max: 50) |
| `display_name` | string | No | — | Display name for exported brain (required for export) |
| `description` | string | No | — | Description for exported brain (export only) |
| `author` | string | No | — | Author name for exported brain (export only, default: anonymous) |
| `tags` | array[string] | No | — | Tags for exported brain (export only) |
| `output_path` | string | No | — | File path to save exported .brain package (export only) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

---

*Auto-generated by `scripts/gen_mcp_docs.py` from `tool_schemas.py` — 56 tools.*
