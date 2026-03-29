# MCP Tools Reference

Complete reference for all NeuralMemory MCP tools.
**52 tools** available via MCP stdio transport.

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

---

## Core Memory {#core}

### `nmem_remember`

Store a memory. Auto-detects type if not specified. Error resolution: when a new memory contradicts a stored error (type='error'), the system automatically creates a RESOLVED_BY synapse and demotes the error's activation by >=50%, so the agent stops repeating outdated errors. Detection is automatic via tag overlap (>50%) and factual contradiction patterns — no manual tagging needed. Sensitive content is auto-encrypted when encryption is enabled, instead of being rejected.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | — | The content to remember |
| `type` | string (`fact`, `decision`, `preference`, `todo`, `insight`, `context`, `instruction`, `error`, `workflow`, `reference`, `boundary`) | No | — | Memory type (auto-detected if not specified) |
| `tier` | string (`hot`, `warm`, `cold`) | No | — | Memory tier: hot (always in context, slow decay), warm (default, semantic match), cold (explicit recall only, fast de... |
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

Store multiple memories in a single call. Max 20 items, 500K total chars. Each item supports the same fields as nmem_remember. Returns per-item results (partial success — one bad item won't block the rest).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memories` | array[object] | Yes | — | Array of memories to store (max 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_recall`

Query memories by semantic search with confidence ranking.

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
| `tags` | array[string] | No | — | Filter by tags (AND — all must match). Checks tags, auto_tags, and agent_tags columns. |
| `mode` | string (`associative`, `exact`) | No | — | Recall mode: 'associative' (default) returns formatted context, 'exact' returns raw neuron contents verbatim without ... |
| `include_citations` | boolean | No | default: true | Include citation and audit trail in exact recall results (default: true). |
| `recall_token_budget` | integer | No | — | When set, activates budget-aware fiber selection: ranks fibers by value-per-token and selects the most efficient ones... |
| `permanent_only` | boolean | No | — | Exclude ephemeral (session-scoped) memories from results. Default: false (include all). |
| `clean_for_prompt` | boolean | No | — | Return clean bullet-point text without section headers or neuron-type tags. Use when injecting recall output into pro... |
| `tier` | string (`hot`, `warm`, `cold`) | No | — | Filter results by memory tier. Only return memories matching this tier. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_show`

Get full verbatim content + metadata + synapses for a specific memory by ID. Use this when you need the exact, unmodified content of a stored memory.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | — | The fiber_id or neuron_id of the memory to retrieve |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_context`

Get recent memories as context.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | default: 10 | Number of recent memories (default: 10) |
| `fresh_only` | boolean | No | — | Only include memories < 30 days old |
| `warn_expiry_days` | integer | No | — | If set, warn about memories expiring within this many days. Adds expiry_warnings to response. |
| `include_ghosts` | boolean | No | default: true | Include faded ghost memories at bottom of context with recall keys (default: true). Set false to suppress. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_todo`

Add a TODO memory (30-day expiry).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | Yes | — | The task to remember |
| `priority` | integer | No | default: 5 | Priority 0-10 (default: 5) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_auto`

Auto-capture memories from text. 'process' analyzes+saves, 'flush' for emergency capture.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`status`, `enable`, `disable`, `analyze`, `process`, `flush`) | Yes | — | Action: 'process' analyzes and saves, 'analyze' only detects, 'flush' emergency capture before compaction (skips dedu... |
| `text` | string | No | — | Text to analyze (required for 'analyze' and 'process') |
| `save` | boolean | No | — | Force save even if auto-capture disabled (for 'analyze') |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_suggest`

Autocomplete suggestions from brain neurons. When called with no prefix, returns idle neurons that have never been accessed — useful for discovering neglected knowledge that needs reinforcement.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prefix` | string | No | — | The prefix text to autocomplete |
| `limit` | integer | No | default: 5 | Max suggestions (default: 5) |
| `type_filter` | string (`time`, `spatial`, `entity`, `action`, `state`, `concept`, `sensory`, `intent`) | No | — | Filter by neuron type |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Session & Context {#session}

### `nmem_session`

Track session state: task, feature, progress.

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

Save project context, decisions, instructions for cross-session persistence.

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

Load saved project context, decisions, and progress.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `level` | integer | No | — | Detail level: 1=quick (~500 tokens), 2=detailed (~1300 tokens), 3=full (~3300 tokens). Default: 1 |
| `topic` | string | No | — | Search for a specific topic in context (e.g., 'auth', 'database') |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Provenance & Sources {#provenance}

### `nmem_provenance`

Trace provenance, verify, or approve a memory neuron. Use 'trace' to see full provenance chain (source, stored_by, verified, approved). Use 'verify' or 'approve' to add audit trail entries.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`trace`, `verify`, `approve`) | Yes | — | Action: trace (view chain), verify (mark verified), approve (mark approved). |
| `neuron_id` | string | Yes | — | Neuron ID to trace/verify/approve. |
| `actor` | string | No | default: mcp_agent | Who is performing the verification/approval (default: mcp_agent). |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_source`

Manage memory sources (provenance). Register external documents, laws, APIs, or other origins so memories can answer 'where did this come from?'.

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

Brain stats: memory counts and freshness.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_health`

Brain health: purity score, grade, warnings.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_evolution`

Brain evolution: maturation, plasticity, coherence.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_habits`

Workflow habits: suggest, list, or clear.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`suggest`, `list`, `clear`) | Yes | — | suggest=get next action suggestions, list=show learned habits, clear=remove all habits |
| `current_action` | string | No | — | Current action type for suggestions (required for suggest action) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_narrative`

Generate narratives: timeline, topic, or causal chain.

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

Create, list, or inspect hypotheses — evolving beliefs with Bayesian confidence tracking. Hypotheses auto-resolve when evidence is strong enough (confirmed at >=0.9 confidence with >=3 evidence-for, refuted at <=0.1 with >=3 evidence-against). Use nmem_evidence to add supporting/opposing evidence.

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

Add evidence for or against a hypothesis. Updates confidence via Bayesian update with surprise weighting and diminishing returns. Auto-resolves hypothesis when evidence threshold is met.

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

Create, list, or inspect predictions — falsifiable claims about future observations. Predictions track confidence, optional deadlines, and can link to hypotheses via PREDICTED synapse. Verified predictions propagate evidence back to linked hypotheses. Use nmem_verify to record outcomes.

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

Verify a prediction as correct or wrong. Optionally records an observation, creates VERIFIED_BY or FALSIFIED_BY synapse, and propagates evidence to linked hypotheses. Returns updated calibration score.

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

Cognitive overview — O(1) summary of active hypotheses, pending predictions, calibration score, and knowledge gaps. Use 'summary' for instant dashboard, 'refresh' to recompute scores from current state.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`summary`, `refresh`) | Yes | — | summary=get current hot index, refresh=recompute scores |
| `limit` | integer | No | default: 10, for summary | Max hot items to return (default: 10, for summary) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_gaps`

Metacognition — track what the brain doesn't know. Detect knowledge gaps from contradictions, low-confidence hypotheses, recall misses, or manual flagging. Resolve gaps when new information fills them.

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

Schema evolution — evolve hypotheses into new versions. Creates a version chain via SUPERSEDES synapse so the brain tracks how beliefs changed over time. Use when a hypothesis needs updating with new understanding.

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

Find and explain the shortest path between two entities in the neural graph. Returns a step-by-step explanation with synapse types, weights, and supporting memory evidence.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `from_entity` | string | Yes | — | Source entity name to start from (e.g. 'React', 'authentication') |
| `to_entity` | string | Yes | — | Target entity name to reach (e.g. 'performance', 'JWT') |
| `max_hops` | integer | No | default: 6 | Maximum path length (default: 6) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Training & Import {#training}

### `nmem_train`

Train brain from documentation files. Supports PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV (requires: pip install neural-memory[extract]). Trained memories are pinned by default (no decay, no compression, permanent KB).

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

Train brain from database schema.

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

Index codebase for code-aware recall.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`scan`, `status`) | Yes | — | scan=index codebase, status=show what's indexed |
| `path` | string | No | default: current working directory | Directory to index (default: current working directory) |
| `extensions` | array[string] | No | default: [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cc"] | File extensions to index (default: [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".h", ".... |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_import`

Import from external systems (ChromaDB, Mem0, Cognee, etc.).

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

Edit an existing memory's type, content, or priority. Use when a memory was auto-typed incorrectly or needs content correction. Preserves all connections (synapses) and fiber associations.

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

Explicitly delete or close a specific memory. Soft delete by default (marks as expired). Use hard=true for permanent removal. Use for closing completed TODOs or removing outdated/incorrect memories.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memory_id` | string | Yes | — | The fiber ID of the memory to forget |
| `hard` | boolean | No | default: false = soft delete | Permanent deletion with cascade cleanup (default: false = soft delete) |
| `reason` | string | No | — | Why this memory is being forgotten (stored in logs) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_pin`

Pin, unpin, or list pinned memories. Pinned memories skip decay, pruning, and compression — use for permanent knowledge base content.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`pin`, `unpin`, `list`) | No | — | Action: pin (default), unpin, or list pinned memories |
| `fiber_ids` | array[string] | No | — | Fiber IDs to pin or unpin (required for pin/unpin, ignored for list) |
| `limit` | integer | No | default: 50, max: 200 | Max results for list action (default: 50, max: 200) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_consolidate`

Run memory consolidation on the current brain. Strategies: prune (remove weak synapses/orphans), merge (combine overlapping fibers), summarize (cluster topic neurons), mature (episodic→semantic), infer (co-activation synapses), enrich (metadata extraction), dream (synthetic bridges), learn_habits (workflow patterns), dedup (merge near-duplicates), semantic_link (cross-domain connections), compress (old fibers), process_tool_events, detect_drift (find tag synonyms/aliases), all (run all in dependency order). Use dry_run=true to preview without applying changes.

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

Semantic drift detection — find tag clusters that should be merged or aliased. Detects when different tags refer to the same concept using Jaccard similarity. Actions: detect (run analysis), list (show clusters), merge (apply canonical tag), alias (mark as related), dismiss (ignore cluster).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`detect`, `list`, `merge`, `alias`, `dismiss`) | Yes | — | detect=run drift analysis, list=show existing clusters, merge/alias/dismiss=resolve a specific cluster |
| `cluster_id` | string | No | — | Cluster ID to resolve (required for merge/alias/dismiss) |
| `status` | string (`detected`, `merged`, `aliased`, `dismissed`) | No | — | Filter clusters by status (for list action) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_review`

Spaced repetition reviews (Leitner box system).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`queue`, `mark`, `schedule`, `stats`) | Yes | — | queue=get due reviews, mark=record review result, schedule=manually schedule a fiber, stats=review statistics |
| `fiber_id` | string | No | — | Fiber ID (required for mark and schedule actions) |
| `success` | boolean | No | — | Whether recall was successful (for mark action, default: true) |
| `limit` | integer | No | default: 20 | Max items in queue (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_alerts`

Brain health alerts: list or acknowledge.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`list`, `acknowledge`) | Yes | — | list=view active/seen alerts, acknowledge=mark alert as handled |
| `alert_id` | string | No | — | Alert ID to acknowledge (required for acknowledge action) |
| `limit` | integer | No | default: 50 | Max alerts to list (default: 50) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Cloud Sync & Backup {#sync}

### `nmem_sync`

Trigger manual sync with hub server.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`push`, `pull`, `full`, `seed`) | Yes | — | push=send local changes, pull=get remote changes, full=bidirectional sync, seed=populate change log from existing dat... |
| `hub_url` | string | No | — | Hub server URL (overrides config). Must be http:// or https:// |
| `strategy` | string (`prefer_recent`, `prefer_local`, `prefer_remote`, `prefer_stronger`) | No | default: from config | Conflict resolution strategy (default: from config) |
| `api_key` | string | No | default: from config | API key override (default: from config) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_sync_status`

Show sync status: pending changes, devices, last sync.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_sync_config`

View or update sync configuration. Use action='setup' for guided onboarding, action='activate' to activate a purchased license key.

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

Send brain database file as backup to Telegram.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `brain_name` | string | No | default: active brain | Brain name to backup (default: active brain) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

## Versioning & Transfer {#meta}

### `nmem_version`

Brain version control: snapshot, list, rollback, diff.

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

Transplant memories between brains by tags/types.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_brain` | string | Yes | — | Name of the source brain to extract from |
| `tags` | array[string] | No | — | Tags to filter — fibers matching ANY tag will be included |
| `memory_types` | array[string] | No | — | Memory types to filter (fact, decision, etc.) |
| `strategy` | string (`prefer_local`, `prefer_remote`, `prefer_recent`, `prefer_stronger`) | No | default: prefer_local | Conflict resolution strategy (default: prefer_local) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_conflicts`

Memory conflicts: list, resolve, or pre-check.

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

Generate charts from memory data. Returns Vega-Lite, markdown table, or ASCII chart.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | What to visualize (e.g., 'ROE trend across quarters', 'revenue by product') |
| `chart_type` | string (`line`, `bar`, `pie`, `scatter`, `table`, `timeline`) | No | — | Chart type (auto-detected if omitted based on data shape) |
| `format` | string (`vega_lite`, `markdown_table`, `ascii`, `all`) | No | default: vega_lite | Output format (default: vega_lite) |
| `limit` | integer | No | default: 20 | Max data points (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_watch`

Watch directories for file changes and auto-ingest into memory.

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

Knowledge Surface management — generate or inspect the .nm surface file. The surface is a compact knowledge graph (~1000 tokens) loaded every session. Actions: generate (rebuild from brain.db), show (display current surface info).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`generate`, `show`) | No | — | generate=rebuild surface from brain.db, show=display current surface info |
| `token_budget` | integer | No | default: 1200 | Token budget for surface (default: 1200). Only used with generate action. |
| `max_graph_nodes` | integer | No | default: 30 | Max graph nodes to include (default: 30). Only used with generate action. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |

### `nmem_tool_stats`

Tool usage analytics: which tools agents use, frequency, success rates, and daily trends.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`summary`, `daily`) | Yes | — | summary=top tools with success rates, daily=usage breakdown by day |
| `days` | integer | No | default: 30 | Time window in days (default: 30) |
| `limit` | integer | No | default: 20 | Max tools to return (default: 20) |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_lifecycle`

Memory lifecycle management — view lifecycle states and manage compression resistance. Hot memories (accessed recently or high priority) resist compression automatically. Actions: status (distribution of lifecycle states), recover (rehydrate a compressed memory), freeze (prevent a memory from compressing), thaw (resume normal lifecycle).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`status`, `recover`, `freeze`, `thaw`) | Yes | — | status=show lifecycle distribution, recover=rehydrate compressed memory, freeze=prevent compression, thaw=resume norm... |
| `id` | string | No | — | Neuron ID (required for recover/freeze/thaw). For recover, fiber_id is also accepted. |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_refine`

Refine an instruction or workflow memory — update its content, record a failure mode, or add a trigger pattern. Each refinement increments the version counter and stores a snapshot in refinement_history. Use this to improve instructions based on real-world usage: update text when the instruction needs correction, add failure_mode when something went wrong, add trigger to improve recall.

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

Report execution outcome for an instruction or workflow memory. Increments execution_count, updates success_rate, and optionally records failure modes. Instructions with high success_rate + many executions are boosted during recall. Call this after executing an instruction to build up its track record.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `neuron_id` | string | Yes | — | ID of the instruction or workflow memory that was executed. |
| `success` | boolean | Yes | — | Whether execution succeeded. |
| `failure_description` | string | No | — | If failed, brief description of what went wrong (appended to failure_modes, deduped, capped at 20). |
| `context` | string | No | — | Optional context about the execution (stored for auditability). |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

### `nmem_budget`

Token budget analysis for recall — estimate, analyze, or optimize context window usage. Use 'estimate' to dry-run a query and see token cost breakdown. Use 'analyze' to profile the brain's average fiber token costs by memory type. Use 'optimize' to find low-value-per-token fibers that are candidates for compression.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string (`estimate`, `analyze`, `optimize`) | Yes | — | Action: 'estimate' (dry-run recall cost), 'analyze' (brain token profile), 'optimize' (find compression candidates). |
| `query` | string | No | — | Query to estimate recall cost for (used with action='estimate'). |
| `max_tokens` | integer | No | default: 4000 | Token budget to estimate against (default: 4000). |
| `compact` | boolean | No | — | Return compact response (strip metadata hints, truncate lists). Saves 60-80% tokens. |
| `token_budget` | integer | No | — | Max tokens for response. Progressively strips content to fit budget. |

---

*Auto-generated by `scripts/gen_mcp_docs.py` from `tool_schemas.py` — 52 tools.*
