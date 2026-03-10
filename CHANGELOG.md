# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.29.0] - 2026-03-10

### Added

- **Reciprocal Rank Fusion (RRF)** — Multi-retriever score blending for anchor ranking. Combines BM25/FTS5, embedding similarity, and graph expansion ranks into unified scores using the RRF formula (`score = Σ weight_i / (k + rank_i)`). Anchors now start with differentiated activation levels instead of uniform 1.0. Config: `rrf_k` (default 60).
- **Graph-based query expansion** — 1-hop neighbor traversal from entity/concept anchors adds soft expansion anchors. Exploits knowledge graph structure for associative priming (e.g., "auth" → OAuth2 → JWT, session). Config: `graph_expansion_enabled`, `graph_expansion_max`, `graph_expansion_min_weight`.
- **Personalized PageRank (PPR) activation** — Optional replacement for classic BFS spreading activation. Distributes activation proportional to edge weights / out-degree with damping (teleport back to seed set), naturally handling hub dampening. Opt-in via `activation_strategy = "ppr"` or `"hybrid"` (PPR + reflex). Config: `ppr_damping`, `ppr_iterations`, `ppr_epsilon`.
- **Tag filtering in Query API and MCP** — `POST /query` accepts `tags: list[str]` (AND filter, max 20). `nmem_recall` accepts `tags: list[str]` to scope results to specific tag sets. Filters across `tags`, `auto_tags`, and `agent_tags` columns. Backward compatible — `tags=None` returns all results as before.

### Fixed

- **Marketplace plugin install** — Removed unrecognized `features` key from `marketplace.json` that caused Claude Code `/plugin marketplace add` to fail with schema validation error (#49).

## [2.28.0] - 2026-03-08

### Added

- **`nmem_remember_batch`** — Bulk remember up to 20 memories in a single call. Partial success supported (individual failures don't block others). Added to `standard` tool tier.
- **Trust score** — First-class `trust_score` (0.0–1.0) and `source` fields on TypedMemory. Source-specific ceiling caps: `user_input=0.9`, `ai_inference=0.7`, `auto_capture=0.5`, `verified=1.0`. Schema v22 migration adds columns + index.
- **`min_trust` filter** — `nmem_recall` accepts optional `min_trust` parameter to filter out low-confidence memories.
- **Auto-promote context→fact** — Frequently-recalled context memories (frequency ≥ 5) are automatically promoted to `fact` during consolidation. Audit trail in metadata (`auto_promoted`, `promoted_from`, `promoted_at`).
- **SEMANTIC alternative path** — Memories can reach SEMANTIC stage via intensive reinforcement (`rehearsal_count ≥ 15` + `5 distinct 2h-windows`) as alternative to the 3-distinct-days spacing requirement. Enables agents with burst usage patterns.

### Fixed

- **FK constraint race condition** — `update_fiber()` no longer raises ValueError when a fiber is deleted between deferred-write enqueue and flush. Gracefully skips with debug log.

### Changed

- **MCP startup 3x faster** — Lazy-import `cli.setup` (defer until first-time init actually needed) and `sync.client`/`sync.sync_engine` (defer aiohttp until first sync call). Cold start: 611ms → 197ms.

## [2.27.3] - 2026-03-08

### Fixed

- **OpenAI-compatible client HTTP 400** — Tool schemas now include `parameters` alias alongside `inputSchema`, fixing "schema must be type object, got type None" errors when MCP tools are forwarded through OpenAI-compatible bridges (Cursor, LiteLLM, etc.)

### Added

- **Cognitive Reasoning Guide** — Full workflow documentation: hypothesize, evidence, predict, verify loop with Bayesian confidence formula, end-to-end examples (`docs/guides/cognitive-reasoning.md`)
- **Schema v21 Migration Guide** — New tables, auto-migration behavior, rollback instructions (`docs/guides/schema-v21-migration.md`)
- **Learning Habits Guide** — 3-stage pipeline, thresholds, confidence calculation, suggestion engine (`docs/guides/learning-habits.md`)
- **Pre-ship smoke tests** — Auto-type classifier (13 cases) and cognitive engine integration test in `scripts/pre_ship.py`

## [2.27.2] - 2026-03-07

### Fixed

- **OpenClaw plugin: lazy auto-connect** — Fixed tools returning "NeuralMemory service not running" when OpenClaw calls `register()` multiple times across subsystems (gateway, agent worker, CLI). Agent worker instance now lazily connects on first tool call via `ensureConnected()` with connection mutex to prevent race conditions (#38)

## [2.27.1] - 2026-03-06

### Added

- **`nmem_edit`** — Edit memory type, content, or priority by fiber ID. Preserves all neural connections. Supports typed_memory path (type/priority) and anchor neuron path (content update)
- **`nmem_forget`** — Soft delete (sets expires_at for natural decay) or hard delete (permanent removal with cascade to fiber + typed_memory). Also handles orphan neuron deletion
- **Enhanced MCP instructions** — Richer behavioral directives: brain growth tips, rich language patterns (causal/temporal/relational/decisional/comparative), memory correction guidance, all 38 tools listed
- **Enhanced plugin instructions** — Comprehensive agent guidance in `.claude-plugin/plugin.json` for proactive memory usage

### Fixed

- **FK constraint errors** — `INSERT OR REPLACE INTO neuron_states` and `save_maturation` now catch `sqlite3.IntegrityError` when neuron was deleted by consolidation prune (previously crashed with FOREIGN KEY constraint failed)
- **Auto-type classifier bias** — Reordered `suggest_memory_type()`: DECISION now checked before INSIGHT to prevent "because" from hijacking decisions. Removed overly broad "because"/"pattern" from INSIGHT keywords. Added "rejected"/"went with" to DECISION, "prefers"/"preferred" to PREFERENCE. Tightened TODO keywords and added guard against descriptive "should"
- **DECISION_PATTERNS greediness** — Removed overly broad patterns (`"we're going to"`, `"let's use"`, `"going to"`) from `auto_capture.py` that caused false decision captures
- **Synapse FK error message** — Distinguished FOREIGN KEY violations from UNIQUE violations in `add_synapse()` for clearer error messages

- **Cognitive Reasoning Layer** — 8 new MCP tools for hypothesis-driven reasoning (38 tools total)
  - `nmem_hypothesize` — Create and manage hypotheses with Bayesian confidence tracking and auto-resolution
  - `nmem_evidence` — Submit evidence for/against hypotheses, auto-updates confidence via sigmoid-dampened shift
  - `nmem_predict` — Make falsifiable predictions with deadlines, linked to hypotheses via PREDICTED synapse
  - `nmem_verify` — Verify predictions as correct/wrong, propagates result to linked hypothesis
  - `nmem_cognitive` — Hot index: ranked summary of active hypotheses + pending predictions with calibration score
  - `nmem_gaps` — Knowledge gap metacognition: detect, track, prioritize, and resolve what the brain doesn't know
  - `nmem_schema` — Schema evolution: evolve hypotheses into new versions via SUPERSEDES synapse chain
  - `nmem_explain` — (moved to cognitive) Trace shortest path between concepts with evidence
- **Schema v21** — Three new tables: `cognitive_state` (hypothesis/prediction tracking), `hot_index` (ranked cognitive summary), `knowledge_gaps` (metacognition)
- **Pure cognitive engine** (`engine/cognitive.py`) — Stateless functions: `update_confidence`, `detect_auto_resolution`, `compute_calibration`, `score_hypothesis`, `score_prediction`, `gap_priority`
- **Bayesian confidence model** — Sigmoid-dampened shift with surprise factor and diminishing returns from total evidence
- **Auto-resolution** — Hypotheses with confidence ≥0.9 + 3 supporting evidence auto-confirm; ≤0.1 + 3 against auto-refute
- **Prediction calibration** — Tracks correct/wrong ratio across all resolved predictions
- **Schema version chain** — `parent_schema_id` column + `get_schema_history()` walks the SUPERSEDES chain with cycle guard
- **Knowledge gap detection sources** — `contradiction`, `low_confidence_hypothesis`, `user_flagged`, `recall_miss`, `stale_schema`

## [2.26.1] - 2026-03-05

### Added

- **Dashboard: actionable health penalties** — Top penalties section shows ranked cards with score bar, penalty points lost, estimated gain if fixed, and exact action to improve each component
- **API: `top_penalties` field** in `/api/dashboard/health` response — exposes diagnostics engine penalty analysis to frontend
- **i18n: penalty translations** — English and Vietnamese keys for top penalties section

## [2.26.0] - 2026-03-05

### Added

- **Brain Health Guide** (`docs/guides/brain-health.md`) — comprehensive guide explaining all 7 health metrics, thresholds, improvement roadmap (F through A), common issues, maintenance schedule
- **Connection Tracing docs** (`nmem_explain`) — added to README, MCP prompt, brain health guide. Previously undocumented feature that traces shortest path between concepts
- **Embedding auto-detection** (`provider = "auto"`) — automatically detects best available embedding provider: Ollama → sentence-transformers → Gemini → OpenAI. Lowers barrier for cross-language recall
- **Consolidation post-run hints** — warns about orphan neurons (>20%) and missing consolidation after running `nmem consolidate`
- **Pre-ship verification script** (`scripts/pre_ship.py`) — automated quality gate: version consistency, ruff, mypy, import smoke test, fast tests, plugin checks
- **MCP instructions update** — health interpretation, priority scale, tagging strategy, maintenance schedule added to system prompt

### Changed

- README: added nmem_explain to tools table, brain health section, connection tracing section, embedding auto-detect
- OpenClaw npm package renamed to `neuralmemory` (published on npm)

## [2.25.1] - 2026-03-05

### Fixed

- **`nmem flush` stdin blocking** — Process hangs forever when spawned as subprocess without piped input; `sys.stdin.read()` blocks because no EOF is sent. Added 5s timeout via `ThreadPoolExecutor` (fixes #27)
- **Consolidation prune** — Protects fiber members from orphan pruning + invariant tests
- **Orphan rate** — Counts fiber membership correctly, isolated E2E tests from production DB
- **Dashboard dist** — Bundled for `pip install` compatibility

### Changed

- Published v2.25.0 release (was stuck in draft)

## [OpenClaw Plugin 1.5.0] - 2026-03-05

### Fixed

- **Plugin ID mismatch warning** — Renamed package from `@neuralmemory/openclaw-plugin` to `neuralmemory` to match manifest `id`. OpenClaw's `deriveIdHint()` extracts the unscoped package name as `idHint`, which previously produced `openclaw-plugin` ≠ `neuralmemory`
- **Tool schema provider compatibility** — Replaced `integer` with `number` (Gemini rejects `integer`), added `additionalProperties: false` (OpenAI strict mode), removed constraint keywords (`maxLength`, `maxItems`, `minimum`, `maximum`) that some providers reject. MCP server validates these server-side
- **Pre-existing test bugs** — Config test missing `initTimeout` in expected defaults; execute tests passing args as `id` parameter

## [2.25.0] - 2026-03-04

### Added

- **Proactive Memory Auto-Save** — 4-layer system ensures agents use NeuralMemory without explicit instructions
  - **MCP `instructions`** — Behavioral directives in InitializeResult, auto-injected into agent context
  - **Post-tool passive capture** — Server-side auto-analysis of recall/context/recap/explain results with rate limiting (3/min)
  - **Plugin `instructions` field** — Short nudge for all plugin users
  - **Enhanced stop hook** — Transcript capture 80→150 lines, session summary extraction, always saves at least one context memory
- **Ollama embedding provider** — Local zero-cost inference via Ollama API (contributed by @xthanhn91)

### Fixed

- **Scale performance bottlenecks** — Consolidation prune, neuron dedup, cache improvements (PR #23)
- **OpenClaw plugin `execute()` signature** — Missing `id` parameter broke all agent tool calls (issue #19)
- **Auto-consolidation crash** — `ValueError: 'none' is not a valid ConsolidationStrategy` (issue #20)
- **`nmem remember --stdin`** — CLI now supports piped input for safe shell usage (issue #21)
- **CI test compatibility** — `test_remember_sensitive_content` mock fix for Python 3.11

## [2.24.2] - 2026-03-03

### Added

- **Dashboard Phase 2** — Complete visual dashboard overhaul
  - **Sigma.js graph visualization** — WebGL-rendered neural graph with ForceAtlas2 layout, node limit selector (100-1000), click-to-inspect detail panel, color-coded by neuron type
  - **ReactFlow mindmap** — Interactive fiber mindmap with dagre left-to-right tree layout, custom nodes (root/group/leaf), MiniMap, zoom/pan, click-to-select neuron details
  - **Theme toggle** — Light / Dark / System cycle button in TopBar, warm cream light mode (`#faf8f3`), class-based TailwindCSS 4 dark mode via `@custom-variant`
  - **Delete brain** — Trash icon on inactive brains in Overview table with confirmation dialog
  - **Click-to-switch brain** — Click inactive brain row to switch active brain
- **CLI update check fix** — Editable/dev installs no longer show misleading "Update available" prompts

### Removed

- **Legacy dashboard UI** — Removed `dashboard.html`, `index.html`, legacy JS/CSS/locales (4,451 LOC), `/static` mount from FastAPI

### Dependencies

- Added `@xyflow/react`, `@dagrejs/dagre` (ReactFlow mindmap)
- Added `graphology-layout-forceatlas2` (Sigma.js graph layout)

## [2.24.1] - 2026-03-03

### Fixed

- **IntegrityError in consolidation** — `save_maturation` FK constraint failed when orphaned maturation records referenced deleted fibers
  - Added `cleanup_orphaned_maturations()` to purge stale records before stage advancement
  - Defensive try/except for any remaining FK errors during `_mature()`

### Tests

- 2 new tests for orphaned maturation handling
- Total: 3145 passing

## [2.24.0] - 2026-03-03

### Fixed

- **[CRITICAL] SQL Injection Prevention** — `get_synapses_for_neurons` direction param validated against whitelist instead of raw f-string
- **[HIGH] BFS max_hops off-by-one** — Nodes at depth=max_hops no longer uselessly enqueued then discarded
- **[HIGH] Bidirectional path search** — `memory_store.get_path()` now respects `bidirectional=True` via `to_undirected()`
- **[HIGH] JSON-RPC parse errors** — Returns proper `{"code": -32700}` error instead of silently dropping malformed messages
- **[HIGH] Encryption failure policy** — Returns error instead of silently storing plaintext when encryption fails
- **[HIGH] `disable_auto_save` placement** — Moved inside `try` block in tool_handlers and conflict_handler so `finally` always re-enables
- **[HIGH] Cross-brain depth validation** — Added int coercion + 0-3 clamping for depth parameter
- **[HIGH] Factory sync exception handling** — Narrowed bare `except Exception` to specific exception types
- **[HIGH] SSN pattern false positives** — Excluded invalid prefixes (000, 666, 900-999); raised base64/hex minimums to 64 chars
- **[MEDIUM] MCP notification handling** — Unknown notifications return None instead of error responses
- **[MEDIUM] Brain ID error propagation** — New `_get_brain_or_error()` helper prevents uncaught ValueError in 6 handlers
- **[MEDIUM] Connection handler I/O** — Removed unused brain fetch in `_explain`
- **[MEDIUM] Evidence fetch optimization** — Removed wasted source neuron from evidence query
- **[MEDIUM] Narrative date validation** — Added `end_date < start_date` guard
- **[MEDIUM] CORS port handling** — Enumerate common dev ports instead of invalid `:*` wildcard
- **[MEDIUM] Embedding config** — Graceful fallback instead of crash on invalid provider
- **[LOW] Type coercion** — max_hops/max_fibers/max_depth safely coerced to int
- **[LOW] Immutability** — Dict mutations replaced with spread patterns in review_handler and encoder
- **[LOW] Schema cleanup** — Removed empty `"required": []` from nmem_suggest

### Tests

- Fixed and added 5 tests (max_hops_capped, avg_weight, default_hops, tier assertions, embedding fallback)
- Total: 3143 passing

## [2.23.0] - 2026-03-03

### Added

- **nmem_explain — Connection Explainer** — New MCP tool to explain how two entities are related
  - Finds shortest path through synapse graph via bidirectional BFS
  - Hydrates path with fiber evidence (memory summaries)
  - Returns structured steps + human-readable markdown explanation
  - New engine module: `connection_explainer.py` with `ConnectionStep` and `ConnectionExplanation` dataclasses
  - New handler mixin: `ConnectionHandler` following established mixin pattern
  - Args: `from_entity`, `to_entity` (required), `max_hops` (optional, 1-10, default 6)

### Fixed

- **OpenClaw Compatibility** — Handle JSON string arguments in MCP `tools/call` handler
  - OpenClaw sends `arguments` as JSON string instead of dict — now auto-parsed
  - Prevents crash when receiving `"arguments": "{\"content\": \"...\"}"` format

### Improved

- **Bidirectional BFS** — `get_path()` in SQLite storage now supports `bidirectional=True`
  - Uses `UNION ALL` to traverse both outgoing and incoming synapse edges
  - Updated abstract base + all 5 storage implementations

### Tests

- 11 new tests for connection explainer (engine + MCP handler + integration)
- Total: 3140+ passing

## [2.22.0] - 2026-03-03

### Fixed

- **#12 Version Mismatch** — Detect editable installs in update hint, show version in `nmem_stats`
- **#14 Dedup on Remember** — Enable SimHash dedup (Tier 1) by default, surface `dedup_hint` in remember response, skip content < 20 chars
- **#11 SEMANTIC Stage Blocked** — Rehearse maturation records on retrieval so memories can reach SEMANTIC stage (requires 3+ distinct reinforcement days)
- **#15 Low Activation Efficiency** — Fix Hebbian learning None activation floor (0.1 instead of None → delta > 0), add dormant neuron reactivation during consolidation

### Added

- **#10 Semantic Linking** — `SemanticLinkingStep` cross-links entity/concept neurons to existing similar neurons (reduces orphan rate)
- **#13 Neuron Diversity** — `ExtractActionNeuronsStep` + `ExtractIntentNeuronsStep` extract ACTION/INTENT neurons from verb/goal phrases (improves type diversity from 4-5 to 6-7 of 8 types)
- **Dormant Reactivation** — Consolidation ENRICH tier bumps up to 20 dormant neurons (access_frequency=0) with +0.05 activation

### Tests

- 55 new tests across 6 test files: version check (12), dedup default (9), maturation rehearsal (5), semantic linking (6), action/intent extraction (15), activation efficiency (8)
- Total: 3127 passing

## [2.21.0] - 2026-03-03

### Added

- **Cross-Language Recall Hint** — Smart detection when recall misses due to language mismatch
  - Detects query language vs brain majority language (Vietnamese ↔ English)
  - Shows actionable `cross_language_hint` in recall response when embedding is not enabled
  - Suggests `pip install` if sentence-transformers not installed, config-only if already installed
  - `detect_language()` extracted as reusable module-level function with Vietnamese-unique char detection

- **Embedding Setup Guide** — Comprehensive docs for all embedding providers
  - New `docs/guides/embedding-setup.md` with provider comparison, config examples, troubleshooting
  - Free multilingual model recommendations: `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, 384D, ~440MB)
  - Provider comparison table: sentence_transformer (free/local) vs Gemini vs OpenAI

- **Embedding Documentation & Onboarding**
  - README: updated "None — pure algorithmic" → "Optional", added embedding quick-start section
  - `.env.example`: added `GEMINI_API_KEY`, `OPENAI_API_KEY` vars
  - Onboarding step 6: suggests cross-language recall setup for new users

### Improved

- **Vietnamese Language Detection** — More accurate short-text detection
  - Added `_VI_UNIQUE_CHARS` set (chars exclusive to Vietnamese, not shared with French/Spanish)
  - Short text like "lỗi xác thực" now correctly detected as Vietnamese

### Tests

- 18 new tests in `test_cross_language_hint.py` (8 detect_language + 10 hint logic)
- All 3090+ tests pass

## [2.20.0] - 2026-03-03

### Added

- **Gemini Embedding Provider** — Cross-language recall via Google Gemini embeddings (PR #9 by @xthanhn91)
  - `GeminiEmbedding` provider: `gemini-embedding-001` (3072D), `text-embedding-004` (768D)
  - Parallel anchor sources: embedding + FTS5 run concurrently (not fallback-only)
  - Config pipeline: `config.toml[embedding]` → `EmbeddingSettings` → `BrainConfig` → SQLite
  - Doc training embeds anchor neurons for cross-language retrieval
  - E2E validated: 100/100 Vietnamese queries on English KB (avg confidence 0.98)
  - Optional dependency: `pip install 'neural-memory[embeddings-gemini]'`

- **Sufficiency Enhancements** — Smarter retrieval gating
  - EMA calibration: per-gate accuracy tracking, auto-downgrade unreliable gates
  - Per-query-type thresholds: strict (factual), lenient (exploratory), default profiles
  - Diminishing returns gate: early-exit when multi-pass retrieval plateaus

### Fixed

- **Comprehensive Audit** — 7 CRITICAL, 17 HIGH, 18 MEDIUM fixes
  - Security: auth guard on consolidation routes, CORS wildcard removal, path traversal fix
  - Performance: `@lru_cache` regex, cached QueryRouter/MemoryEncryptor, `asyncio.gather` embeddings
  - Infrastructure: `.dockerignore`, `.env.example`, bounded exports, async cursor managers
- **PR #9 Review Fixes** — 3 HIGH, 6 MEDIUM, 3 LOW
  - Bare except → specific exceptions in doc_trainer
  - `EmbeddingSettings` frozen + validated (rejects invalid providers)
  - Probe-first early exit in embedding anchor scan (performance)
  - Correct task_type for semantic discovery consolidation
  - Hardcoded paths → env vars in E2E scripts

### Tests

- 33 new sufficiency tests (EMA calibration, query profiles, diminishing returns)
- 6 new EmbeddingSettings validation tests
- 13 new Gemini embedding provider tests
- Full suite: 3054 passed, 0 failed

## [2.19.0] - 2026-03-02

### Added

- **React Dashboard** — Modern dashboard replacing legacy Alpine.js/vis.js
  - Vite 7 + React 19 + TypeScript + TailwindCSS 4 + shadcn/ui
  - Warm cream light theme (`#faf8f3`) with dark mode support
  - 7 pages: Overview, Health (Recharts radar), Graph, Timeline, Evolution, Diagrams, Settings
  - TanStack Query 5 for data fetching, Zustand 5 for state
  - Lazy-loaded routes with skeleton loaders
  - `/ui` and `/dashboard` serve React SPA, legacy at `/ui-legacy` and `/dashboard-legacy`
  - Brain file info: paths, sizes, disk usage in Settings page

- **Telegram Backup Integration** — Send brain `.db` files to Telegram
  - `TelegramClient` (aiohttp): `send_message` (auto-split >4096 chars), `send_document`, `backup_brain`
  - `TelegramConfig` frozen dataclass in `unified_config.py` (`[telegram]` TOML section)
  - CLI: `nmem telegram status`, `nmem telegram test`, `nmem telegram backup [--brain NAME]`
  - MCP tool: `nmem_telegram_backup` (28 total tools)
  - Dashboard API: `GET /api/dashboard/telegram/status`, `POST .../test`, `POST .../backup`
  - Dashboard Settings page: status indicator, test button, backup button
  - Bot token via `NMEM_TELEGRAM_BOT_TOKEN` env var only (never in config file)
  - Chat IDs in `config.toml` under `[telegram]` section

- **Brain Files API** — `GET /api/dashboard/brain-files`
  - Returns brains directory path, per-brain file path + size, total disk usage

### Tests

- 15 new Telegram tests: config, token, client, status, MCP handler
- MCP tool count updated (27→28)

## [2.18.0] - 2026-03-02

### Added

- **Export Markdown** — `nmem brain export --format markdown -o brain.md`
  - Human-readable brain export grouped by memory type (facts, decisions, insights, etc.)
  - Tag index with occurrence counts
  - Statistics table with neuron/synapse/fiber breakdowns
  - Pinned memory indicators and sensitive content exclusion support
  - New module: `cli/markdown_export.py` (~180 LOC)

- **Original Timestamp** — `event_at` parameter on `nmem_remember`
  - MCP: `nmem_remember(content="Meeting at 8am", event_at="2026-03-02T08:00:00")`
  - CLI: `nmem remember "Meeting" --timestamp "2026-03-02T08:00:00"`
  - Time neurons and fiber `time_start/time_end` use the original event time
  - Supports ISO format with optional timezone (auto-stripped for UTC storage)

### Changed

- **Health Roadmap Enhancement** — Concrete metrics in improvement actions
  - Actions now include specific numbers: "Store memories to build ~250 more connections (current: 0.5 synapses/neuron, target: 3.0+)"
  - Added `timeframe` field to roadmap: "~2 weeks with regular use"
  - Dynamic action strings computed from actual brain metrics (neuron counts, orphan rate, etc.)
  - Grade transition messages include estimated timeframe

### Tests

- 31 new tests: `test_markdown_export.py` (11), `test_health_roadmap.py` (13), `test_event_timestamp.py` (7)

## [2.17.0] - 2026-03-02

### Added

- **Knowledge Base Training** — Multi-format document extraction with pinned memories
  - 12 supported formats: .md, .mdx, .txt, .rst (passthrough), .pdf, .docx, .pptx, .html/.htm (rich docs), .json, .xlsx, .csv (structured data)
  - `doc_extractor.py` — Format-specific extractors with 50MB file size limit
  - Optional dependencies via `neural-memory[extract]` for non-text formats (pymupdf4llm, python-docx, python-pptx, beautifulsoup4, markdownify, openpyxl)
- **Pinned Memories** — Permanent knowledge that bypasses decay, pruning, and compression
  - `Fiber.pinned: bool` field — pinned fibers skip all lifecycle operations
  - 4 lifecycle bypass points: decay, pruning, compression, maturation
  - `nmem_pin` MCP tool for manual pin/unpin
- **Training File Dedup** — SHA-256 hash tracking prevents re-ingesting same documents
  - `training_files` table with hash, status, progress tracking
  - Resume support for interrupted training sessions
- **Tool Memory System** — Tracks MCP tool usage patterns and effectiveness
  - `MemoryType.TOOL` — New memory type (90-day expiry, 0.06 decay rate)
  - `SynapseType.EFFECTIVE_FOR` + `USED_WITH` — Tool effectiveness and co-occurrence synapses
  - PostToolUse hook — Fast JSONL buffer capture (<50ms, no SQLite on hot path)
  - `engine/tool_memory.py` — Batch processing during consolidation
  - `PROCESS_TOOL_EVENTS` consolidation strategy

### Fixed (Comprehensive Audit — 4 CRITICAL, 8 HIGH, 12 MEDIUM)

- **CRITICAL**: Auth guard on consolidation routes, CORS wildcard removal, path traversal fix, coverage threshold enforcement
- **HIGH**: Reject null client IP, sanitize error messages, Windows ACL key protection, FalkorDB password warning
- **Performance**: Module-level regex compilation with `@lru_cache`, cached QueryRouter + MemoryEncryptor (lazy singleton), `asyncio.gather` for parallel embeddings, batch neuron delete (chunked 500), SQL FILTER clause combining queries
- **Infrastructure**: `.dockerignore`, `.env.example`, bounded export (LIMIT 50000), `asyncio.Lock` for storage cache, cursor context managers

### Changed

- Schema version 18 → 20 (tool_events table, pinned column on fibers, training_files table)
- SynapseType enum: 22 → 24 types (EFFECTIVE_FOR, USED_WITH)
- MemoryType enum: 10 → 11 types (TOOL)
- MCP tools: 26 → 27 (added nmem_pin)
- ROADMAP.md — Complete rewrite as forward-looking 5-phase vision
- Agent instructions — 7 new sections covering all 28 MCP tools
- MCP prompt — Added KB training, pin, health, review, import instructions

---

## [2.16.0] - 2026-02-28

### Added

- **Algorithmic Sufficiency Check** — Post-stabilization gate that early-exits when activation signal is too weak
  - 8-gate evaluation (priority-ordered, first match wins): no_anchors, empty_landscape, unstable_noise, ambiguous_spread, intersection_convergence, high_coverage_strong_hit, focused_result, default_pass
  - Unified confidence formula from 7 weighted inputs (activation, focus_ratio, coverage, intersection_ratio, proximity, stability, path_diversity)
  - Conservative bias — false-INSUFFICIENT penalized 10× more than false-SUFFICIENT
  - `engine/sufficiency.py` (~302 LOC), `storage/sqlite_calibration.py` (~133 LOC)
  - Schema migration v17 → v18 (`retrieval_calibration` table)

---

## [2.15.1] - 2026-02-28

### Fixed

- **SharedStorage CRUD Endpoint Mismatch** — Client called endpoints that didn't exist on server
  - Added 14 CRUD endpoints to `server/routes/memory.py` (neurons + synapses full lifecycle, state, neighbors, path)
  - 6 new Pydantic models in `server/models.py`
- **Brain Import Deduplication** — Changed `INSERT` → `INSERT OR REPLACE` in `sqlite_brain_ops.py` for idempotent imports

---

## [2.15.0] - 2026-02-28

### Added

- **Trusted Networks for Docker/Container Deployments** — Configurable non-localhost access via `NEURAL_MEMORY_TRUSTED_NETWORKS` env var (CIDR notation)
  - `is_trusted_host()` function with safe `ipaddress` module validation
  - Default remains localhost-only (secure by default)

### Fixed

- **OpenClaw Plugin Zod Peer Dependency** — Pinned `zod` to `^3.0.0`

---

## [2.14.0] - 2026-02-27

### Added

- **MCP Tool Tiers** — 3-tier system (minimal/standard/full) for controlling exposed tools
  - `ToolTierConfig` frozen dataclass with case-insensitive tier parsing
  - `get_tool_schemas_for_tier()` filters tools by tier level
  - Minimal: 4 core tools, Standard: 8 tools, Full: all 27 tools
  - Hidden tools still callable via dispatch (tier controls visibility, not access)
- **Consolidation Eligibility Hints** — `_eligibility_hints()` explains why 0 changes happened
- **Habits Status** — Progress bars for emerging patterns
- **Diagnostics Improvements** — Actionable recommendations with specific numbers
- **Graph SVG Export** — Pure Python SVG export with dark theme, zero external deps

---

## [2.13.0] - 2026-02-27

### Added

- **Error Resolution Learning** — When a new FACT/INSIGHT contradicts an existing ERROR memory, the system creates a `RESOLVED_BY` synapse linking fix → error instead of just flagging a conflict
  - `RESOLVED_BY` synapse type added to `SynapseType` enum (22 types total)
  - Resolved errors get ≥50% activation demotion (2x stronger than normal conflicts)
  - Error neurons marked with `_conflict_resolved` and `_resolved_by` metadata
  - Auto-detection via neuron metadata `{"type": "error"}` — no caller changes needed
  - Zero-cost: pure graph manipulation, no LLM calls
  - 7 new tests in `test_error_resolution.py`

### Changed

- `resolve_conflicts()` accepts optional `existing_memory_type` parameter
- `conflict_detection.py` now imports `logging` module for RESOLVED_BY synapse debug logging

---

## [2.8.1] - 2026-02-23

### Added

- **FalkorDB Graph Storage Backend** — Optional graph-native storage replacing SQLite for high-performance traversal
  - `FalkorDBStorage` composite class implementing full `NeuralStorage` ABC via 5 specialized mixins
  - `FalkorDBBaseMixin` — connection pooling, query helpers (`_query`, `_query_ro`), index management
  - `FalkorDBNeuronMixin` — neuron CRUD with graph node operations
  - `FalkorDBSynapseMixin` — synapse CRUD with typed graph edges
  - `FalkorDBFiberMixin` — fiber CRUD with `CONTAINS` relationships, batch operations
  - `FalkorDBGraphMixin` — native Cypher spreading activation (1-4 hop BFS via variable-length paths)
  - `FalkorDBBrainMixin` — brain registry graph, import/export, graph-level clear
  - Brain-per-graph isolation (`brain_{id}`) for native multi-tenancy
  - Read-only query routing via `ro_query` for registry reads and fiber lookups
  - Per-neuron limit enforcement in `find_fibers_batch` via UNWIND+collect/slice Cypher pattern
  - Connection health verification via Redis PING with automatic reconnect
  - `docker-compose.falkordb.yml` — standalone FalkorDB service configuration
  - Migration CLI: `nmem migrate falkordb` to move SQLite brain data to FalkorDB
  - 69 tests across 6 test files (auto-skip when FalkorDB unavailable)
  - SQLite remains default — FalkorDB is opt-in via `[storage]` TOML config

### Fixed

- **mypy: `set_brain` missing from ABC** — Added `set_brain(brain_id)` to `NeuralStorage` base class, resolving 2 mypy errors in `unified_config.py`
- **Registry reads used write queries** — Added `_registry_query_ro()` for read-only brain registry operations (`get_brain`, `find_brain_by_name`)
- **`find_fibers_batch` ignored `limit_per_neuron`** — Rewrote with UNWIND+collect/slice Cypher for proper per-neuron limiting
- **FalkorDB health check was superficial** — `_get_falkordb_storage()` now performs actual Redis PING instead of just `_db is not None` check
- **`export_brain` leaked `brain_id` in error** — Sanitized to generic "Brain not found" message
- **Import sorting (I001)** — Fixed `falkordb.asyncio` before `redis.asyncio` in `falkordb_store.py`
- **Unused import (F401)** — Removed stale `SQLiteStorage` import from `unified_config.py`
- **Quoted annotation (UP037)** — Unquoted `_storage_cache` and `_falkordb_storage` type annotations
- **Silent error logging** — Upgraded index creation and connection close errors from debug to warning level

## [2.8.0] - 2026-02-22

### Added

- **Adaptive Recall (Bayesian Depth Prior)** — System learns optimal retrieval depth per entity pattern
  - Beta distribution priors per (entity, depth) pair — picks depth with highest E[Beta(a,b)]
  - 5% epsilon exploration to discover better depths for known entities
  - Fallback to rule-based detection when < 5 queries or no priors exist
  - Outcome recording: updates alpha (success) or beta (failure) based on confidence + fibers_matched
  - 30-day decay (a *= 0.9, b *= 0.9) to forget stale patterns
  - `DepthPrior`, `DepthDecision` frozen dataclasses + `AdaptiveDepthSelector` engine
  - `SQLiteDepthPriorMixin` with batch fetch, upsert, stale decay, delete operations
  - Configurable: `adaptive_depth_enabled` (default True), `adaptive_depth_epsilon` (default 0.05)
- **Tiered Memory Compression** — Age-based compression preserving entity graph structure (zero-LLM)
  - 5 tiers: Full (< 7d), Extractive (7-30d), Entity-only (30-90d), Template (90-180d), Graph-only (180d+)
  - Entity density scoring: `count(neurons_referenced) / word_count` per sentence
  - Reversible for tiers 1-2 (backup stored), irreversible for tiers 3-4
  - Integrated as `COMPRESS` strategy in `ConsolidationEngine` (Tier 2)
  - `CompressionTier` IntEnum, `CompressionConfig`, `CompressionResult` frozen dataclasses
  - `SQLiteCompressionMixin` for backup storage with stats
  - Configurable: `compression_enabled` (default True), `compression_tier_thresholds` (7, 30, 90, 180 days)
- **Multi-Device Sync** — Hub-and-spoke incremental sync via change log + sequence numbers
  - **Device Identity**: UUID-based device_id generation, persisted in config, `DeviceInfo` frozen dataclass
  - **Change Tracking**: Append-only `change_log` table recording all neuron/synapse/fiber mutations
    - `ChangeEntry` frozen dataclass, `SQLiteChangeLogMixin` with 6 CRUD methods
    - `record_change()`, `get_changes_since(sequence)`, `mark_synced()`, `prune_synced_changes()`
  - **Incremental Sync Protocol**: Delta-based merge using neural-aware conflict resolution
    - `SyncRequest`, `SyncResponse`, `SyncChange`, `SyncConflict` frozen dataclasses
    - `ConflictStrategy` enum: prefer_recent, prefer_local, prefer_remote, prefer_stronger
    - Neural merge rules: weight=max, access_frequency=sum, tags=union, conductivity=max, delete wins
  - **Sync Engine**: `SyncEngine` orchestrator with `prepare_sync_request()`, `process_sync_response()`, `handle_hub_sync()`
  - **Hub Server Endpoints** (localhost-only by default):
    - `POST /hub/register` — register device for brain
    - `POST /hub/sync` — push/pull incremental changes
    - `GET /hub/status/{brain_id}` — sync status + device count
    - `GET /hub/devices/{brain_id}` — list registered devices
  - **3 new MCP tools** (full tier only):
    - `nmem_sync` — trigger manual sync (push/pull/full)
    - `nmem_sync_status` — show pending changes, devices, last sync
    - `nmem_sync_config` — configure hub URL, auto-sync, conflict strategy
  - `SyncConfig` frozen dataclass: enabled (default False), hub_url, auto_sync, sync_interval_seconds, conflict_strategy
  - Device tracking columns on neurons/synapses/fibers: `device_id`, `device_origin`, `updated_at`
  - Schema migrations v15 → v16 (depth_priors, compression_backups, fiber compression_tier) → v17 (change_log, devices, device columns)

### Changed

- **SQLite schema** — Version 15 → 17 (two migrations)
- **MCP tools** — Expanded from 23 to 26 (`nmem_sync`, `nmem_sync_status`, `nmem_sync_config`)
- **MCPServer mixin chain** — Added `SyncToolHandler` mixin
- **`Fiber` model** — Added `compression_tier: int = 0` field
- **`BrainConfig`** — Added 4 new fields: `adaptive_depth_enabled`, `adaptive_depth_epsilon`, `compression_enabled`, `compression_tier_thresholds`
- **`UnifiedConfig`** — Added `device_id` field and `SyncConfig` dataclass
- **`ConsolidationEngine`** — Added `COMPRESS` strategy enum + Tier 2 registration + `fibers_compressed`/`tokens_saved` report fields
- **Hub endpoints** — Pydantic request validation with regex-based brain_id/device_id format checks
- Tests: 2687 passed (up from 2527), +160 new tests across 8 test files

## [2.7.1] - 2026-02-21

### Added

- **MCP Tool Tiers** — Config-based filtering to reduce token overhead per API turn
  - 3 tiers: `minimal` (4 tools, ~84% savings), `standard` (8 tools, ~69% savings), `full` (all 23, default)
  - `ToolTierConfig` frozen dataclass in `unified_config.py` with `from_dict()`/`to_dict()`
  - `get_tool_schemas_for_tier(tier)` in `tool_schemas.py` — filters schemas by tier
  - `[tool_tier]` TOML section in `config.toml` for persistent configuration
  - Hidden tools remain callable via dispatch — only schema exposure changes
  - CLI command: `nmem config tier [--show | minimal | standard | full]`
- **Description Compression** — All 23 tool descriptions compressed (~22% token reduction at full tier)

### Changed

- `MCPServer.get_tools()` now respects `config.tool_tier.tier` setting
- `tool_schemas.py` refactored: `_ALL_TOOL_SCHEMAS` module-level list + `TOOL_TIERS` dict
- Tests: added 28 new tests in `test_tool_tiers.py`

## [2.7.0] - 2026-02-18

### Added

- **Spaced Repetition Engine** — Leitner box system (5 boxes: 1d, 3d, 7d, 14d, 30d) for memory reinforcement
  - `ReviewSchedule` frozen dataclass: fiber_id, brain_id, box (1–5), next_review, streak, review_count
  - `SpacedRepetitionEngine`: `get_review_queue()`, `process_review()` (calls `ReinforcementManager`), `auto_schedule_fiber()`
  - `advance(success)` returns new schedule instance — box increments on success (max 5), resets to 1 on failure
  - Auto-scheduling: fibers with `priority >= 7` are automatically scheduled in `_remember`
  - `SQLiteReviewsMixin`: upsert, get_due, get_stats with `min(limit, 100)` cap
  - `InMemoryReviewsMixin` for testing
  - `ReviewHandler` MCP mixin: `nmem_review` tool (queue/mark/schedule/stats actions)
  - Schema migration v14 → v15 (`review_schedules` table + 2 indexes)
- **Memory Narratives** — Template-based markdown narrative generation (no LLM)
  - 3 modes: `timeline` (date range), `topic` (spreading activation via `ReflexPipeline`), `causal` (CAUSED_BY chain traversal)
  - `NarrativeItem` + `Narrative` frozen dataclasses with `to_markdown()` rendering
  - Timeline mode: queries fibers by date range, sorts chronologically, groups by date headers
  - Topic mode: runs SA query, fetches matched fibers, sorts by relevance
  - Causal mode: uses `trace_causal_chain()` to follow CAUSED_BY synapses, builds cause→effect narrative
  - `NarrativeHandler` MCP mixin: `nmem_narrative` tool (timeline/topic/causal actions)
  - Configurable `max_fibers` with server-side cap of 50
- **Semantic Synapse Discovery** — Offline consolidation using embeddings to find latent connections
  - Batch embeds CONCEPT + ENTITY neurons, evaluates cosine similarity pairs above threshold
  - Creates SIMILAR_TO synapses with `weight = similarity * 0.6` and `{"_semantic_discovery": True}` metadata
  - Configurable: `semantic_discovery_similarity_threshold` (default 0.7), `semantic_discovery_max_pairs` (default 100)
  - Integrated as Tier 5 (`SEMANTIC_LINK`) in `ConsolidationEngine` strategy dispatch
  - 2× faster decay for unreinforced semantic synapses in `_prune` (reinforced_count < 2 → decay factor 0.5)
  - Optional — gracefully skipped if `sentence-transformers` not installed
  - `SemanticDiscoveryResult` dataclass: neurons_embedded, pairs_evaluated, synapses_created, skipped_existing
- **Cross-Brain Recall** — Parallel spreading activation across multiple brains
  - Extends `nmem_recall` with optional `brains` array parameter (max 5 brains)
  - Resolves brain names → DB paths via `UnifiedConfig`, opens temporary `SQLiteStorage` per brain
  - Parallel query via `asyncio.gather`, each brain runs independent `ReflexPipeline`
  - SimHash-based deduplication across brain results (keeps higher confidence on collision)
  - Confidence-sorted merge with `[brain_name]` prefixed context sections
  - `CrossBrainFiber` + `CrossBrainResult` frozen dataclasses
  - Temporary storage instances closed in `finally` blocks

### Changed

- **MCPServer mixin chain** — Added `ReviewHandler` + `NarrativeHandler` mixins (16 → 18 handler mixins)
- **MCP tools** — Expanded from 21 to 23 (`nmem_review`, `nmem_narrative`)
- **SQLite schema** — Version 14 → 15 (`review_schedules` table)
- **`nmem_recall` schema** — Added `brains` array property for cross-brain queries
- **`BrainConfig`** — Added `semantic_discovery_similarity_threshold` and `semantic_discovery_max_pairs` fields
- **`ConsolidationEngine`** — Added `SEMANTIC_LINK` strategy enum + Tier 5 + `semantic_synapses_created` report field
- **Consolidation prune** — Unreinforced semantic synapses (`_semantic_discovery` metadata) decay at 2× rate
- Tests: 2399 passed (up from 2314), +85 new tests across 4 features

## [2.6.0] - 2026-02-18

### Added

- **Smart Context Optimizer** — Composite scoring replaces naive loop in `nmem_context`
  - 5-factor weighted score: activation (0.30) + priority (0.25) + frequency (0.20) + conductivity (0.15) + freshness (0.10)
  - SimHash-based deduplication removes near-duplicate content before token budgeting
  - Proportional token budget allocation: items get budget proportional to their composite score
  - Items below minimum budget (20 tokens) are dropped; oversized items are truncated
  - `optimization_stats` field in response shows `items_dropped` and `top_score`
- **Proactive Alerts Queue** — Persistent brain health alerts with full lifecycle management
  - `Alert` frozen dataclass with `AlertStatus` (active → seen → acknowledged → resolved) and 7 `AlertType` enum values
  - `SQLiteAlertsMixin` with CRUD operations: `record_alert` (6h dedup cooldown), `get_active_alerts`, `mark_alerts_seen`, `mark_alert_acknowledged`, `resolve_alerts_by_type`
  - `AlertHandler` MCP mixin: `nmem_alerts` tool (list/acknowledge actions)
  - Auto-creation from health pulse hints; auto-resolution when conditions clear
  - Pending alert count surfaced in `nmem_remember`, `nmem_recall`, `nmem_context` responses
  - Schema migration v13 → v14 (alerts table + indexes)
- **Recall Pattern Learning** — Discover and materialize query topic co-occurrence patterns
  - `extract_topics()` — keyword-based topic extraction from recall queries (min_length=3, cap 10)
  - `mine_query_topic_pairs()` — session-grouped, time-windowed (600s default) pair mining
  - `extract_pattern_candidates()` — frequency filtering + confidence scoring
  - `learn_query_patterns()` — materializes patterns as CONCEPT neurons + BEFORE synapses with `{"_query_pattern": True}` metadata
  - `suggest_follow_up_queries()` — follows BEFORE synapses for related topic suggestions
  - Integrated into LEARN_HABITS consolidation strategy
  - `related_queries` field added to `nmem_recall` response

### Changed

- **MCPServer mixin chain** — Added `AlertHandler` mixin (15 → 16 handler mixins)
- **MCP tools** — Expanded from 20 to 21 (`nmem_alerts`)
- **SQLite schema** — Version 13 → 14 (alerts table)
- **`nmem_context` response** — Now includes `optimization_stats` when items are dropped
- **`nmem_recall` response** — Now includes `related_queries` from learned patterns
- Tests: 2314 passed (up from 2291)

## [2.5.0] - 2026-02-18

### Added

- **Onboarding flow** — Detects fresh brain (0 neurons + 0 fibers) and surfaces a 4-step getting-started guide on the first tool call (`_remember`, `_recall`, `_context`, `_stats`). Shows once per server instance.
- **Background expiry cleanup** — Fire-and-forget task auto-deletes expired `TypedMemory` + underlying fibers on a configurable interval (default 12h, max 100/run). Fires `MEMORY_EXPIRED` hooks. Piggybacks on `_check_maintenance()`.
- **Scheduled consolidation** — Background `asyncio` loop runs consolidation every 24h (configurable strategies: prune, merge, enrich). Shares `_last_consolidation_at` with `MaintenanceHandler` to prevent overlap. Initial delay of one full interval avoids triggering on restart.
- **Version check handler** — Background task checks PyPI every 24h for newer versions of `neural-memory`. Caches result and surfaces `update_hint` in `_remember`, `_recall`, `_stats` responses when an update is available. Uses `urllib` (no extra deps), validates HTTPS scheme.
- **Expiry alerts** — `warn_expiry_days` parameter on `nmem_recall`; expiring-soon count in health pulse thresholds
- **Evolution dashboard** — `/api/evolution` REST endpoint + dashboard UI tab for brain maturation metrics (stage distribution, plasticity, proficiency)

### Changed

- **MaintenanceConfig** — Added 8 new config fields: `expiry_cleanup_enabled`, `expiry_cleanup_interval_hours`, `expiry_cleanup_max_per_run`, `scheduled_consolidation_enabled`, `scheduled_consolidation_interval_hours`, `scheduled_consolidation_strategies`, `version_check_enabled`, `version_check_interval_hours`
- **MCPServer mixin chain** — Added `OnboardingHandler`, `ExpiryCleanupHandler`, `ScheduledConsolidationHandler`, `VersionCheckHandler` mixins
- **Server lifecycle** — `run_mcp_server()` now starts scheduled consolidation + version check at startup, cancels all background tasks on shutdown

## [2.4.0] - 2026-02-17

### Security

- **6-phase security audit** — Comprehensive audit across 142K LOC / 190 files covering engine, storage, server, config, MCP/CLI, core, safety, utils, sync, integration, and extraction modules
- **Path traversal fixes** — 3 CRITICAL path injection vulnerabilities in CLI commands (tools, brain import, shortcuts) patched with `resolve()` + `is_relative_to()`
- **CORS hardening** — Replaced wildcard patterns with explicit localhost origins in FastAPI server
- **TOML injection prevention** — Added `_sanitize_toml_str()` for user-provided dedup config fields
- **API key masking** — `BrainModeConfig.to_dict()` now serializes api_key as `"***"` instead of plaintext
- **Info leak prevention** — Removed internal IDs, adapter names, and filesystem paths from 5 error messages across MCP, integration, and sync modules
- **WebSocket validation** — Brain ID format + length validation on subscribe action
- **Path normalization** — `SQLiteStorage` and `NEURALMEMORY_DIR` env var paths now resolved with `Path.resolve()`

### Fixed

- **Frozen core models** — `Synapse`, `Fiber`, `NeuronState`, `BrainSnapshot`, `FreshnessResult`, `MemoryFreshnessReport`, `Entity`, `WeightedKeyword`, `TimeHint` dataclasses are now `frozen=True` per immutability contract
- **merge_brain() atomicity** — Restore from backup on import failure instead of leaving empty brain
- **import_brain() orphan** — Brain record INSERT moved inside transaction to prevent orphan on failure
- **Division-by-zero guards** — `_predicates_conflict()` and homeostatic normalization protected against empty inputs
- **Datetime hardening** — 4 `datetime.fromisoformat()` call sites wrapped with try/except + naive UTC enforcement
- **Lateral inhibition** — Ceiling division for fair slot allocation across clusters
- **suggest_memory_type** — Word boundary matching prevents false positives (e.g. "add" no longer matches "address")
- **Git update command** — Detects current branch instead of hardcoded 'main'
- **Dead code removal** — Removed unused `updated_at` field, duplicate index, stale imports

### Performance

- **N+1 query elimination** — `consolidation._prune()` pre-fetches neighbor synapses in batch (was 500+ serial queries); `activation.activate()` caches neighbors + batch state pre-fetch (was ~1000 queries); `conflict_detection` uses `asyncio.gather()` for parallel searches
- **Export safety caps** — `export_brain()` limited to 50K neurons, 100K synapses, 50K fibers
- **Bounds enforcement** — 15+ storage methods capped with `min(limit, MAX)`, schema tool limits enforced
- **Regex pre-compilation** — `sensitive.py` and `trigger_engine.py` patterns compiled at module level with cache
- **Enrichment optimization** — Early exit on empty tags + zero intersection in O(n^2) Jaccard loop
- **ReDoS prevention** — Content length cap (100K chars) before regex matching in sensitive content detection

### Changed

- **BrainConfig.with_updates()** — Replaced 80-line manual field copy with `dataclasses.replace()`
- **DriftReport.variants** — Changed from mutable `list` to `tuple` on frozen dataclass
- **Mutable constants** — `VI_PERSON_PREFIXES` and `LOCATION_INDICATORS` converted to `frozenset`
- **Error handling** — 8 bare `except Exception` blocks narrowed to specific exception types with logging

## [2.2.0] - 2026-02-13

### Added

- **Config presets** — Three built-in profiles: `safe-cost` (token-efficient), `balanced` (defaults), `max-recall` (maximum retention). CLI: `nmem config preset <name> [--list] [--dry-run]`
- **Consolidation delta report** — `run_with_delta()` wrapper computes before/after health snapshots around consolidation, showing purity, connectivity, and orphan rate changes. CLI consolidate now shows health delta.

### Fixed

- **CI lint parity** — CI now passes: fixed 14 lint errors in test files (unused imports, sorting, Yoda conditions)
- **Release workflow idempotency** — `gh release create` no longer fails when release already exists; uploads assets to existing release instead
- **CI test timeouts** — Added `pytest-timeout` (60s default) and `timeout-minutes: 15` to prevent stuck CI jobs

### Changed

- **Makefile** — Added `verify` target matching CI exactly (lint + format-check + typecheck + test-cov + security)
- **Auto-consolidation observability** — Background auto-consolidation now logs purity delta for monitoring

## [2.1.0] - 2026-02-13

### Fixed

- **Brain reset on config migration** — When upgrading to unified config (config.toml), `current_brain` is now migrated from legacy config.json so users don't lose their active brain selection
- **EternalHandler stale brain cache** — Eternal context now detects brain switches and re-creates the context instead of caching the initial brain ID indefinitely
- **Ruff lint errors** — Fixed 7 pre-existing lint violations (unused imports, naming convention, import ordering)
- **Mypy type errors** — Fixed 2 pre-existing type errors (`Any` import, `set()` arg-type)

### Added

- **CLI `--version` flag** — `nmem --version` / `nmem -V` now prints version and exits (standard CLI convention)
- **Actionable health scoring** — `nmem_health` now returns `top_penalties`: top 3 ranked penalty factors with estimated gain and suggested action
- **Semantic stage progress** — `nmem_evolution` now returns `stage_distribution` (fiber counts per maturation stage) and `closest_to_semantic` (top 3 EPISODIC fibers with progress % and next step)
- **Composable encoding pipeline** — Refactored monolithic `encode()` into 14 composable async pipeline steps (`PipelineContext` / `PipelineStep` / `Pipeline`)

### Changed

- **Dependency warning suppression** — pyvi/NumPy DeprecationWarnings are now suppressed at import time with targeted `filterwarnings`

## [2.3.1] - 2026-02-17

### Refactored

- **Engine cleanup** — Removed 176 lines of dead code across 6 engine modules
  - Deduplicated stop-word sets into shared `_STOP_WORDS` frozenset in `conflict_detection.py`
  - Replaced manual `Fiber()` constructor with `dc_replace()` in `consolidation.py`
  - Removed unused `reconstitute_answer()` from `retrieval_context.py`
  - Hoisted expansion suffix/prefix constants to module level in `retrieval.py`
  - Used `heapq.nlargest` instead of sorted+slice in retrieval reinforcement
  - Typed consolidation dispatch dict with `Callable[[], Awaitable[None]]` instead of `Any`

### Fixed

- **Unreachable break in dream** — Outer loop guard added to prevent quadratic blowup when activated neuron list is large (max 50K pairs)
- **JSON snapshot validation** — `brain_versioning.py` now validates parsed JSON is a dict before field access

## [2.3.0] - 2026-02-16

### Added

- **PreCompact + Stop auto-flush hooks** — Pre-compaction hook fires before context compression, parallel CI tests support
- **Emergency flush** (`nmem_auto action="flush"`) — Pre-compaction emergency capture that skips dedup, lowers confidence threshold to 0.5, enables all memory types regardless of config, and boosts priority +2. Tag `emergency_flush` applied to all captured memories. Inspired by OpenClaw Memory's Layer 3 (`memoryFlush`)
- **Session gap detection** — `nmem_session(action="get")` now returns `gap_detected: true` when content may have been lost between sessions (e.g. user ran `/new` without saving). Uses MD5 fingerprint stored on `session_set`/`session_end` to detect gaps from older code paths missing fingerprints
- **Auto-capture preference patterns** — Detects explicit preferences ("I prefer...", "always use..."), corrections ("that's wrong...", "actually, it should be..."), and Vietnamese equivalents. New memory type `preference` with 0.85 confidence
- **Windows surrogate crash fix** — MCP server now strips lone surrogate characters (U+D800-U+DFFF) from tool arguments before processing, preventing `UnicodeEncodeError` on Windows stdio pipes

### Fixed

- **CI lint failure** — Fixed ruff RUF002 (ambiguous EN DASH `–` in docstring) in `mcp/server.py`
- **CI stress test timeouts** — Skipped stress tests on GitHub runners to prevent CI timeout failures

### Changed

- **Release workflow hardened** — `release.yml` now validates tag version matches `pyproject.toml` + `__init__.py` before publishing, and runs full CI (lint + typecheck + test) as a gate before PyPI upload

## [Unreleased]

### Fixed

- **Agent forgets tools after `/new`** — `before_agent_start` hook now always injects `systemPrompt` with tool instructions, ensuring the agent knows about NeuralMemory tools even after session reset. Previously only `prependContext` (data) was injected, leaving the agent unaware of available tools
- **Agent confuses CLI vs MCP tool calls** — `systemPrompt` injection explicitly states "call as tool, NOT CLI command", preventing agents from running `nmem remember` in terminal instead of calling the `nmem_remember` tool
- **`openclaw plugins list` not recognizing plugin on Windows** — Changed `main` and `openclaw.extensions` from TypeScript source (`src/index.ts`) to compiled output (`dist/index.js`). Added `prepublishOnly` and `postinstall` build scripts. Fixed `tsconfig.json` module resolution from `bundler` to `Node16` for broader compatibility
- **OpenClaw plugin ID mismatch** — Added explicit `"id": "neuralmemory"` to `openclaw` section in `package.json`, fixing the `plugin id mismatch (manifest uses "neuralmemory", entry hints "openclaw-plugin")` warning
- **Content-Length framing bug** — Switched from string-based buffer to raw `Buffer` for byte-accurate MCP message parsing. Fixes silent data corruption with non-ASCII content (Vietnamese, emoji, CJK)
- **Null dereference after close()** — `writeMessage()` and `notify()` now guard against null process reference
- **Unhandled tool call errors** — `callTool()` exceptions in tools.ts now caught and returned as structured error responses instead of crashing OpenClaw

### Added

- **Configurable MCP timeout** — New `timeout` plugin config option (default: 30s, max: 120s) for users on slow machines or first-time init
- **Actionable MCP error messages** — Initialize failures now include Python stderr output and specific hints:
  - `ENOENT` → tells user to check `pythonPath` in plugin config
  - Exit code 1 → suggests `pip install neural-memory`
  - Timeout → prints captured stderr + verify command (`python -m neural_memory.mcp`)

### Security

- **Least-privilege child env** — MCP subprocess now receives only whitelisted env vars (`PATH`, `HOME`, `PYTHONPATH`, `NEURALMEMORY_*`) instead of full `process.env`. Prevents leaking API keys and secrets to child process
- **Config validation** — `resolveConfig()` now validates types, ranges, and brain name pattern (`^[a-zA-Z0-9_\-.]{1,64}$`). Invalid values fall back to defaults instead of passing through
- **Input bounds on all tools** — Zod schemas now enforce max lengths: content (100K chars), query (10K), tags (50 items × 100 chars), expires_days (1–3650), context limit (1–200)
- **Buffer overflow protection** — 10 MB cap on stdio buffer; process killed if exceeded
- **Stderr cap** — Max 50 lines collected during init to prevent unbounded memory growth
- **Auto-capture truncation** — Agent messages truncated to 50K chars before sending to MCP
- **Graceful shutdown** — `close()` now removes listeners, waits up to 3s for exit, then escalates to SIGKILL
- **Config schema hardened** — Added `additionalProperties: false` and brain name `pattern` constraint

## [1.7.4] - 2026-02-11

### Fixed

- **Full mypy compliance**: Resolved all 341 mypy errors across 79 files (0 errors in 170 source files)
  - Added `TYPE_CHECKING` protocol stubs to all mixin classes (storage, MCP handlers)
  - Added generic type parameters to all bare `dict`/`list` annotations
  - Narrowed `str | None` → `str` before passing to typed parameters
  - Removed 14 stale `# type: ignore` comments
  - Added proper type annotations to `HybridStorage` factory delegate methods
  - Fixed variable name reuse across different types in same scope
  - Fixed missing `await` on coroutine calls in CLI commands

### Added

- **CLAUDE.md — Type Safety Rules**: New section documenting mixin protocol stubs, generic type params, Optional narrowing, and `# type: ignore` discipline to prevent future mypy regressions

## [1.7.3] - 2026-02-11

### Added

- **Bundled skills** — 3 Claude Code agent skills (memory-intake, memory-audit, memory-evolution) now ship inside the pip package under `src/neural_memory/skills/`
- **`nmem install-skills`** — new CLI command to install skills to `~/.claude/skills/`
  - `--list` shows available skills with descriptions
  - `--force` overwrites existing with latest version
  - Detects unchanged files (skip), changed files (report "update available"), missing `~/.claude/` (graceful error)
- **`nmem init --skip-skills`** — skills are now installed as part of `nmem init`; use `--skip-skills` to opt out
- Tests: 25 new unit tests for `setup_skills`, `_discover_bundled_skills`, `_classify_status`, `_extract_skill_description`

### Changed

- `_classify_status()` now recognizes "installed" and "updated" as success states
- `skills/README.md` updated: manual copy instructions replaced with `nmem install-skills`

## [1.7.2] - 2026-02-11

### Security

- **CORS hardening**: Default CORS origins changed from `["*"]` to `["http://localhost:*", "http://127.0.0.1:*"]` (C2)
- **Bind address**: Default server bind changed from `0.0.0.0` to `127.0.0.1` (C4)
- **Migration safety**: Non-benign migration errors now halt and raise instead of silently advancing schema version (C8)
- **Info leakage**: Removed available brain names from 404 error responses (H21)
- **URI validation**: Graphiti adapter validates `bolt://`/`bolt+s://` URI scheme before connecting (H23)
- **Error masking**: Exception type names no longer leaked in MCP training error responses (H27)
- **Import screening**: `RecordMapper.map_record()` now runs `check_sensitive_content()` before importing external records (H33)

### Fixed

- Fix `RuntimeError: Event loop is closed` from aiosqlite worker thread on CLI exit (Python 3.12+)
  - **Root cause**: 4 CLI commands (`decay`, `consolidate`, `export`, `import`) called `get_shared_storage()` directly, bypassing `_active_storages` tracking — aiosqlite connections were never closed before event loop teardown
  - Route all CLI storage creation through `get_storage()` in `_helpers.py` so connections are properly tracked and cleaned up
  - Add `await asyncio.sleep(0)` after storage cleanup to drain pending aiosqlite worker thread callbacks before `asyncio.run()` tears down the loop
- **Bounds hardening**: MCP `_habits` fiber fetch reduced 10K→1K; `_context` limit capped at 200; REST `list_neurons` capped at 1000; `EncodeRequest.content` max 100K chars (H11-H13, H32)
- **Data integrity**: `import_brain` wrapped in `BEGIN IMMEDIATE` with rollback on failure (H14)
- **Code quality**: AWF adapter gets ImportError guard; redundant `enable_auto_save()` removed from train handler (C7, H26)
- **Public API**: Added `current_brain_id` property to `NeuralStorage`, `SQLiteStorage`, `InMemoryStorage` — replaces private `_current_brain_id` access (H25)

### Added

- **CLAUDE.md**: Project-level AI coding standards (architecture, immutability, datetime, security, bounds, testing, error handling, naming conventions)
- **Quality gates**: Automated enforcement via ruff, mypy, pytest, and CI
  - 8 new ruff rule sets: S (bandit), A (builtins), DTZ (datetimez), T20 (print), PT (pytest), PERF (perflint), PIE, ERA (eradicate)
  - Per-file-ignores for intentional patterns (CLI print, simhash MD5, SQL column names, etc.)
  - Coverage threshold: 67% enforced in CI and Makefile
  - CI: typecheck job now fails build (removed `continue-on-error` and `|| true`); build requires `[lint, typecheck, test]`; added security scan job
  - Pre-commit: updated hooks (ruff v0.9.6, mypy v1.15.0); added `no-commit-to-branch` and `bandit`
  - Makefile: added `security`, `audit` targets; `check` now includes `security`

### Changed

- Tests: 1759 passed (up from 1696)

## [1.7.1] - 2026-02-11

### Fixed

- Fix `__version__` reporting "1.6.1" instead of "1.7.0" in PyPI package (runtime version mismatch)

## [1.7.0] - 2026-02-11

### Added

- **Proactive Brain Intelligence** — 3 features that make the brain self-aware during normal usage
  - **Related Memories on Write** — `nmem_remember` now discovers and returns up to 3 related existing memories via 2-hop SpreadingActivation from the new anchor neuron. Always-on (~5-10ms overhead), non-intrusive. Response includes `related_memories` list with `fiber_id`, `preview`, and `similarity` score.
  - **Expired Memory Hint** — Health pulse detects expired memories via cheap COUNT query on `typed_memories` table. Surfaces hint when count exceeds threshold (default: 10): `"N expired memories found. Consider cleanup via nmem list --expired."`
  - **Stale Fiber Detection** — Health pulse detects fibers with decayed conductivity (last conducted >90 days ago or never). Surfaces hint when stale ratio exceeds threshold (default: 30%): `"N% of fibers are stale. Consider running nmem_health for review."`
- **MaintenanceConfig extensions** — 3 new configuration fields:
  - `expired_memory_warn_threshold` (default: 10)
  - `stale_fiber_ratio_threshold` (default: 0.3)
  - `stale_fiber_days` (default: 90)
- **Storage layer** — 2 new optional methods on `NeuralStorage`:
  - `get_expired_memory_count()` — COUNT of expired typed memories (SQLite + InMemory)
  - `get_stale_fiber_count(brain_id, stale_days)` — COUNT of stale fibers (SQLite + InMemory)
- **HealthPulse extensions** — `expired_memory_count` and `stale_fiber_ratio` fields
- **HEALTH_DEGRADATION trigger** — `TriggerType.HEALTH_DEGRADATION` for maintenance events

### Changed

- Tests: 1696 passed (up from 1695)

## [1.6.1] - 2026-02-10

### Fixed

- CLI brain commands (`export`, `import`, `create`, `delete`, `health`, `transplant`) now work correctly in SQLite mode
- `brain export` no longer produces empty files when brain was created with `brain create`
- `brain delete` correctly removes `.db` files in unified config mode
- `brain health` uses storage-agnostic `find_neurons()` instead of JSON-internal `_neurons` dict
- All `version` subcommands (`create`, `list`, `rollback`, `diff`) now find brains in SQLite mode
- `shared sync` uses correct storage backend

## [1.6.0] - 2026-02-10

### Added

- **DB-to-Brain Schema Training (`nmem_train_db`)** — Teach brains to understand database structure
  - 3-layer pipeline: `SchemaIntrospector` → `KnowledgeExtractor` → `DBTrainer`
  - Extracts **schema knowledge** (table structures, relationships, patterns) — NOT raw data rows
  - SQLite dialect (v1) via `aiosqlite` read-only connections
  - Schema fingerprint (SHA256) for re-training detection
- **Schema Introspection** — `engine/db_introspector.py`
  - `SchemaDialect` protocol with `SQLiteDialect` implementation
  - Frozen dataclasses: `ColumnInfo`, `ForeignKeyInfo`, `IndexInfo`, `TableInfo`, `SchemaSnapshot`
  - PRAGMA-based metadata extraction (table_info, foreign_key_list, index_list)
- **Knowledge Extraction** — `engine/db_knowledge.py`
  - FK-to-SynapseType mapping with confidence scoring (IS_A, INVOLVES, AT_LOCATION, RELATED_TO)
  - Structure-based join table detection (2+ FKs, ≤1 business column → CO_OCCURS synapse)
  - 5 schema pattern detectors: audit_trail, soft_delete, tree_hierarchy, polymorphic, enum_table
- **Training Orchestrator** — `engine/db_trainer.py`
  - Mirrors DocTrainer architecture: batch save, per-table error isolation, shared domain neuron
  - Configurable: `max_tables` (1-500), `salience_ceiling`, `consolidate`, `domain_tag`
- **MCP Tool: `nmem_train_db`** — `train` and `status` actions

### Fixed

- Security: read-only SQLite connections, absolute path rejection, SQL identifier sanitization, info leakage prevention

### Changed

- MCP tools expanded from 17 to 18
- Tests: 1648 passed (up from 1596)

### Skills

- **3 composable AI agent skills** — ship-faster SKILL.md pattern, installable to `~/.claude/skills/`
  - `memory-intake` — structured memory creation from messy notes, 1-question-at-a-time clarification, batch store with preview
  - `memory-audit` — 6-dimension quality review (purity, freshness, coverage, clarity, relevance, structure), A-F grading
  - `memory-evolution` — evidence-based optimization from usage patterns, consolidation, enrichment, pruning, checkpoint Q&A

## [1.5.0] - 2026-02-10

### Added

- **Conflict Management MCP Tool (`nmem_conflicts`)** — List, resolve, and pre-check memory conflicts
  - `list`, `resolve` (keep_existing/keep_new/keep_both), `check` actions
  - `ConflictHandler` mixin with full input validation
- **Recall Conflict Surfacing** — `has_conflicts` flag and `conflict_count` in default recall response
- **Provenance Source Enrichment** — `NEURALMEMORY_SOURCE` env var → `mcp:{source}` provenance
- **Purity Score Conflict Penalty** — Unresolved CONTRADICTS reduce health score (max -10 points)

### Fixed

- 20+ performance bottlenecks — storage index optimization, encoder batch operations
- 25+ bugs across engine/storage/MCP — deep audit fixes including deprecated `datetime.utcnow()` replacement

### Changed

- MCP tools expanded from 16 to 17
- Tests: 1372 passed (up from 1352)

## [1.4.0] - 2026-02-09

### Added

- **OpenClaw Memory Plugin** — `@neuralmemory/openclaw-plugin` npm package
  - MCP stdio client: JSON-RPC 2.0 with Content-Length framing
  - 6 core tools, 2 hooks (before_agent_start, agent_end), 1 service
  - Plugin manifest with `configSchema` + `uiHints`

### Changed

- Dashboard Integrations tab simplified to status-only with deep links (Option B)

## [1.3.0] - 2026-02-09

### Added

- **Deep Integration Status** — Enhanced status cards, activity log, setup wizards, import sources
- **Source Attribution** — `NEURALMEMORY_SOURCE` env var for integration tracking
- 25 new i18n keys in EN + VI (87 total)

### Changed

- Tests: 1352 passed (up from 1340)

## [1.2.0] - 2026-02-09

### Added

- **Dashboard** — Full-featured SPA at `/dashboard` (Alpine.js + Tailwind CDN, zero-build)
  - 5 tabs: Overview, Neural Graph (Cytoscape.js), Integrations, Health (radar chart), Settings
  - Graph toolbar, toast notifications, skeleton loading, brain management, EN/VI i18n
  - ARIA accessibility, 44px mobile touch targets, design system

### Fixed

- `ModuleNotFoundError: typing_extensions` on fresh Python 3.12 — added dependency

### Changed

- Tests: 1340 passed (up from 1264)

## [1.1.0] - 2026-02-09

### Added

- **ClawHub SKILL.md** — Published `neural-memory@1.0.0` to ClawHub
- **Nanobot Integration** — 4 tools adapted for Nanobot's action interface
- **Architecture Doc** — `docs/ARCHITECTURE_V1_EXTENDED.md`

### Changed

- OpenClaw PR [#12596](https://github.com/openclaw/openclaw/pull/12596) submitted

## [1.0.2] - 2026-02-09

### Fixed

- Empty recall for broad queries — `format_context()` truncates long fiber content to fit token budget
- Diversity metric normalization — Shannon entropy normalized against 8 expected synapse types
- Temporal synapse diversity — `_link_temporal_neighbors()` creates BEFORE/AFTER instead of always RELATED_TO
- Consolidation prune crash — Fixed `Fiber(tags=...)` TypeError, uses `dataclasses.replace()`

## [1.0.0] - 2026-02-09

### Added

- **Brain Versioning** — Snapshot, rollback, diff (schema v11, `brain_versions` table)
- **Partial Brain Transplant** — Topic-filtered merge between brains with conflict resolution
- **Brain Quality Badge** — Grade A-F from BrainHealthReport, marketplace eligibility
- **Optional Embedding Layer** — SentenceTransformer + OpenAI providers (OFF by default)
- **Optional LLM Extraction** — Enhanced relation extraction beyond regex (OFF by default)

### Changed

- Version 1.0.0 — Production/Stable, schema v10 → v11
- MCP tools expanded from 14 to 16 (nmem_version, nmem_transplant)

## [0.20.0] - 2026-02-09

### Added

- **Habitual Recall** — ENRICH, DREAM, LEARN_HABITS consolidation strategies
  - Action event log (hippocampal buffer), sequence mining, workflow suggestions
  - `nmem_habits` MCP tool, `nmem habits` CLI, `nmem update` CLI
  - Prune enhancements: dream synapse 10x decay, high-salience resistance
- Schema v10: `action_events` table
- 6 new BrainConfig fields for habit/dream configuration

### Changed

- `ConsolidationStrategy` extended with ENRICH, DREAM, LEARN_HABITS
- Schema version 9 → 10

## [0.19.0] - 2026-02-08

### Added

- **Temporal Reasoning** — Causal chain traversal, temporal range queries, event sequence tracing
  - `trace_causal_chain()`, `query_temporal_range()`, `trace_event_sequence()`
  - `CAUSAL_CHAIN` and `TEMPORAL_SEQUENCE` synthesis methods
  - Pipeline integration: "Why?" → causal, "When?" → temporal, "What happened after?" → event sequence
  - Router enhancement with traversal metadata in `RouteDecision`

### Changed

- Tests: 1019 passed (up from 987)

## [0.17.0] - 2026-02-08

### Added

- **Brain Diagnostics** — `BrainHealthReport` with 7 component scores and composite purity (0-100)
  - Grade A/B/C/D/F, 7 warning codes, automatic recommendations
  - Tag drift detection via `TagNormalizer.detect_drift()`
- **MCP tool: `nmem_health`** — Brain health diagnostics
- **CLI command: `nmem health`** — Terminal health report with ASCII progress bars

## [0.16.0] - 2026-02-08

### Added

- **Emotional Valence** — Lexicon-based sentiment extraction (EN + VI, zero LLM)
  - `SentimentExtractor`, `Valence` enum, 7 emotion tag categories
  - Negation handling, intensifier detection
  - `FELT` synapses from anchor → emotion STATE neurons
- **Emotional Resonance Scoring** — Up to +0.1 retrieval boost for matching-valence memories
- **Emotional Decay Modulation** — High-intensity emotions decay slower (trauma persistence)

### Changed

- Tests: 950 passed (up from 908)

## [0.15.0] - 2026-02-08

### Added

- **Associative Inference Engine** — Co-activation patterns → persistent CO_OCCURS synapses
  - `compute_inferred_weight()`, `identify_candidates()`, `create_inferred_synapse()`
  - `generate_associative_tags()` from BFS clustering
- **Co-Activation Persistence** — `co_activation_events` table (schema v8 → v9)
  - `record_co_activation()`, `get_co_activation_counts()`, `prune_co_activations()`
- **INFER Consolidation Strategy** — Create synapses from co-activation patterns
- **Tag Normalizer** — ~25 synonym groups + SimHash fuzzy matching + drift detection
- 6 new BrainConfig fields for co-activation configuration

### Changed

- Schema version 8 → 9
- Tests: 908 passed (up from 838)

## [0.14.0] - 2026-02-08

### Added

- **Relation extraction engine**: Regex-based causal, comparative, and sequential pattern detection from content — auto-creates CAUSED_BY, LEADS_TO, BEFORE, SIMILAR_TO, CONTRADICTS synapses during encoding
- **Tag origin tracking**: Separate `auto_tags` (content-derived) from `agent_tags` (user-provided) with backward-compatible `fiber.tags` union property
- **Auto memory type inference**: `suggest_memory_type()` fallback when no explicit type provided at encode time
- **Confirmatory weight boost**: Hebbian +0.1 boost on anchor synapses when agent tags confirm auto tags; RELATED_TO synapses (weight 0.3) for divergent agent tags
- **Bilingual pattern support**: English + Vietnamese regex patterns for causal ("because"/"vì"), comparative ("similar to"), and sequential ("then"/"sau khi") relations
- `RelationType`, `RelationCandidate`, `RelationExtractor` in new `extraction/relations.py`
- `Fiber.auto_tags`, `Fiber.agent_tags` fields with `Fiber.add_auto_tags()` method
- SQLite schema migration v7→v8 with backward-compatible column additions and backfill
- 62 new tests: relation extraction (25), tag origin (10), confirmatory boost (5), relation encoding (7), auto-tags update (15)
- `ROADMAP.md` with versioned plan from v0.14.0 → v1.0.0

### Fixed

- **"Event loop is closed" noise on CLI exit**: aiosqlite connections now properly closed before event loop teardown via centralized `run_async()` helper
- MCP server shutdown now closes storage connection in `finally` block

### Changed

- All 32 CLI `asyncio.run()` calls replaced with `run_async()` for proper cleanup
- Encoder pipeline extended with relation extraction (step 6b) and confirmatory boost (step 6c)
- `Fiber.create(tags=...)` preserved for backward compat — maps to `agent_tags`
- 838 tests passing

## [0.13.0] - 2026-02-07

### Added

- **Ground truth evaluation dataset**: 30 curated memories across 5 sessions (Day 1→Day 30) covering project setup, development, integration, sprint review, and production launch
- **Standard IR metrics**: Precision@K, Recall@K, MRR (Mean Reciprocal Rank), NDCG@K with per-query and per-category aggregation
- **25 evaluation queries**: 8 factual, 6 temporal, 4 causal, 4 pattern, 3 multi-session coherence queries with expected relevant results
- **Naive keyword-overlap baseline**: Tokenize-and-rank strawman that NeuralMemory's activation-based recall must beat
- **Long-horizon coherence test framework**: 5-session simulation across 30 days with recall tracking per session (target: >= 60% at day 30)
- `benchmarks/ground_truth.py` — ground truth memories, queries, session schedule
- `benchmarks/metrics.py` — IR metrics: `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `evaluate_query`, `BenchmarkReport`
- `benchmarks/naive_baseline.py` — keyword overlap ranking and baseline evaluation
- `benchmarks/coherence_test.py` — multi-session coherence test with `CoherenceReport`
- Ground-truth evaluation section in `run_benchmarks.py` comparing NeuralMemory vs baseline
- 27 new unit tests: precision (6), recall (4), MRR (5), NDCG (4), query evaluation (1), report aggregation (2), baseline (5)

### Changed

- `run_benchmarks.py` now includes ground-truth evaluation with NeuralMemory vs naive baseline comparison in generated markdown output

## [0.12.0] - 2026-02-07

### Added

- **Real-time conflict detection**: Detects factual contradictions and decision reversals at encode time using predicate extraction — no LLM required
- **Factual contradiction detection**: Regex-based extraction of `"X uses/chose/decided Y"` patterns, compares predicates across memories with matching subjects
- **Decision reversal detection**: Identifies when a new DECISION contradicts an existing one via tag overlap analysis
- **Dispute resolution pipeline**: Anti-Hebbian confidence reduction, `_disputed` and `_superseded` metadata markers, and CONTRADICTS synapse creation
- **Disputed neuron deprioritization**: Retrieval pipeline reduces activation of disputed neurons by 50% and superseded neurons by 75%
- `CONTRADICTS` synapse type for linking contradictory memories
- `ConflictType`, `Conflict`, `ConflictResolution`, `ConflictReport` in new `engine/conflict_detection.py`
- `detect_conflicts()`, `resolve_conflicts()` for encode-time conflict handling
- 32 new unit tests: predicate extraction (5), predicate conflict (4), subject matching (4), tag overlap (4), helpers (4), detection integration (6), resolution (5)

### Changed

- Encoder pipeline runs conflict detection after anchor neuron creation, before fiber assembly
- Retrieval pipeline adds `_deprioritize_disputed()` step after stabilization to suppress disputed neurons
- `SynapseType` enum extended with `CONTRADICTS = "contradicts"`

## [0.11.0] - 2026-02-07

### Added

- **Activation stabilization**: Iterative dampening algorithm settles neural activations into stable patterns after spreading activation — noise floor removal, dampening (0.85x), homeostatic normalization, convergence detection (typically 2-4 iterations)
- **Multi-neuron answer reconstruction**: Strategy-based answer synthesis replacing single-neuron `reconstitute_answer()` — SINGLE mode (high-confidence top neuron), FIBER_SUMMARY mode (best fiber summary), MULTI_NEURON mode (top-5 neurons ordered by fiber pathway position)
- **Memory maturation lifecycle**: Four-stage memory model STM → Working (30min) → Episodic (4h) → Semantic (7d + spacing effect). Stage-aware decay multipliers: STM 5x, Working 2x, Episodic 1x, Semantic 0.3x
- **Spacing effect requirement**: EPISODIC → SEMANTIC promotion requires reinforcement across 3+ distinct calendar days, modeling biological spaced repetition
- **Pattern extraction**: Episodic → semantic concept formation via tag Jaccard clustering (Union-Find). Clusters of 3+ similar fibers generate CONCEPT neurons with IS_A synapses to common entities
- **MATURE consolidation strategy**: New consolidation strategy that advances maturation stages and extracts semantic patterns from mature episodic memories
- `StabilizationConfig`, `StabilizationReport`, `stabilize()` in new `engine/stabilization.py`
- `SynthesisMethod`, `ReconstructionResult`, `reconstruct_answer()` in new `engine/reconstruction.py`
- `MemoryStage`, `MaturationRecord`, `compute_stage_transition()`, `get_decay_multiplier()` in new `engine/memory_stages.py`
- `ExtractedPattern`, `ExtractionReport`, `extract_patterns()` in new `engine/pattern_extraction.py`
- `SQLiteMaturationMixin` in new `storage/sqlite_maturation.py` — maturation CRUD for SQLite backend
- Schema migration v6→v7: `memory_maturations` table with composite key (brain_id, fiber_id)
- `contributing_neurons` and `synthesis_method` fields on `RetrievalResult`
- `stages_advanced` and `patterns_extracted` fields on `ConsolidationReport`
- Maturation abstract methods on `NeuralStorage` base: `save_maturation()`, `get_maturation()`, `find_maturations()`
- 49 new unit tests: stabilization (12), reconstruction (11), memory stages (16), pattern extraction (8), plus 2 consolidation tests

### Changed

- Retrieval pipeline inserts stabilization phase after lateral inhibition and before answer reconstruction
- Answer reconstruction uses multi-strategy `reconstruct_answer()` instead of `reconstitute_answer()`
- Encoder initializes maturation record (STM stage) when creating new fibers
- Consolidation engine supports `MATURE` strategy for stage advancement and pattern extraction

## [0.10.0] - 2026-02-07

### Added

- **Formal Hebbian learning rule**: Principled weight update `Δw = η_eff * pre * post * (w_max - w)` replacing ad-hoc `weight += delta + dormancy_bonus`
- **Novelty-adaptive learning rate**: New synapses learn ~4x faster, frequently reinforced synapses stabilize toward base rate via exponential decay
- **Natural weight saturation**: `(w_max - w)` term prevents runaway weight growth — weights near ceiling barely change
- **Competitive normalization**: `normalize_outgoing_weights()` caps total outgoing weight per neuron at budget (default 5.0), implementing winner-take-most competition
- **Anti-Hebbian update**: `anti_hebbian_update()` for conflict resolution weight reduction (used in Phase 3)
- `learning_rate`, `weight_normalization_budget`, `novelty_boost_max`, `novelty_decay_rate` on `BrainConfig`
- `LearningConfig`, `WeightUpdate`, `hebbian_update`, `compute_effective_rate`, `normalize_outgoing_weights` in new `engine/learning_rule.py`
- 33 new unit tests covering learning rule, normalization, and backward compatibility

### Changed

- `Synapse.reinforce()` accepts optional `pre_activation`, `post_activation`, `now` parameters — uses formal Hebbian rule when activations provided, falls back to direct delta for backward compatibility
- `ReflexPipeline._defer_co_activated()` passes neuron activation levels to Hebbian strengthening
- `ReflexPipeline._defer_reinforce_or_create()` forwards activation levels to `reinforce()`
- Removed dormancy bonus from `Synapse.reinforce()` (novelty adaptation in learning rule replaces it)

## [0.9.6] - 2026-02-07

### Added

- **Sigmoid activation function**: Neurons now use sigmoid gating (`1/(1+e^(-6(x-0.5)))`) instead of raw clamping, producing bio-realistic nonlinear activation curves
- **Firing threshold**: Neurons only propagate signals when activation meets threshold (default 0.3), filtering borderline noise
- **Refractory period**: Cooldown prevents same neuron firing twice within a query pipeline (default 500ms), checked during spreading activation
- **Lateral inhibition**: Top-K winner-take-most competition in retrieval pipeline — top 10 neurons survive unchanged, rest suppressed by 0.7x factor
- **Homeostatic target field**: Reserved `homeostatic_target` field on NeuronState for v2 adaptive regulation
- `fired` and `in_refractory` properties on `NeuronState`
- `sigmoid_steepness`, `default_firing_threshold`, `default_refractory_ms`, `lateral_inhibition_k`, `lateral_inhibition_factor` on `BrainConfig`
- Schema migration v5→v6: four new columns on `neuron_states` table

### Changed

- `NeuronState.activate()` applies sigmoid function and accepts `now` and `sigmoid_steepness` parameters
- `NeuronState.decay()` preserves all new fields (firing_threshold, refractory_until, refractory_period_ms, homeostatic_target)
- `DecayManager.apply_decay()` uses `state.decay()` instead of manual NeuronState construction
- `ReinforcementManager.reinforce()` directly sets activation level (bypasses sigmoid for reinforcement)
- Spreading activation skips neurons in refractory cooldown
- Storage layer (SQLite + SharedStore) serializes/deserializes all new NeuronState fields

## [0.9.5] - 2026-02-07

### Added

- **Type-aware decay rates**: Different memory types now decay at biologically-inspired rates (facts: 0.02/day, todos: 0.15/day). `DEFAULT_DECAY_RATES` dict and `get_decay_rate()` helper in `memory_types.py`
- **Retrieval score breakdown**: `ScoreBreakdown` dataclass exposes confidence components (base_activation, intersection_boost, freshness_boost, frequency_boost) in `RetrievalResult` and MCP `nmem_recall` response
- **SimHash near-duplicate detection**: 64-bit locality-sensitive hashing via `utils/simhash.py`. New `content_hash` field on `Neuron` model. Encoder and auto-capture use SimHash to catch paraphrased duplicates
- **Point-in-time temporal queries**: `valid_at` parameter on `nmem_recall` filters fibers by temporal validity window (`time_start <= valid_at <= time_end`)
- Schema migration v4→v5: `content_hash INTEGER` column on neurons table

### Changed

- `DecayManager.apply_decay()` now uses per-neuron `state.decay_rate` instead of global rate
- `reconstitute_answer()` returns `ScoreBreakdown` as third tuple element
- `_remember()` MCP handler sets type-specific decay rates on neuron states after encoding

## [0.9.4] - 2026-02-07

### Performance

- **SQLite WAL mode** + `synchronous=NORMAL` + 8MB cache for concurrent reads and reduced I/O
- **Batch storage methods**: `get_synapses_for_neurons()`, `find_fibers_batch()`, `get_neuron_states_batch()` — single `IN()` queries replacing N sequential calls
- **Deferred write queue**: Fiber conductivity, Hebbian strengthening, and synapse writes batched after response assembly
- **Parallel anchor finding**: Entity + keyword lookups via `asyncio.gather()` instead of sequential loops
- **Batch fiber discovery**: Single junction-table query replaces 5-15 sequential `find_fibers()` calls
- **Batch subgraph extraction**: Single query replaces 20-50 sequential `get_synapses()` calls
- **BFS state prefetch**: Batch `get_neuron_states_batch()` per hop instead of individual lookups
- Target: 3-5x faster retrieval (800-4500ms → 200-800ms)

## [0.9.0] - 2026-02-06

### Added

- **Codebase indexing** (`nmem_index`): Index Python files into neural graph for code-aware recall
- **Python AST extractor**: Parse functions, classes, methods, imports, constants via stdlib `ast`
- **Codebase encoder**: Map code symbols to neurons (SPATIAL/ACTION/CONCEPT/ENTITY) and synapses (CONTAINS/IS_A/RELATED_TO/CO_OCCURS)
- **Branch-aware sessions**: `nmem_session` auto-detects git branch/commit/repo and stores in metadata + tags
- **Git context utility**: Detect branch, commit SHA, repo root via subprocess (zero deps)
- **CLI `nmem index` command**: Index codebase from command line with `--ext`, `--status`, `--json` options
- 16 new tests for extraction, encoding, and git context

## [0.8.0]

### Added

- Initial project structure
- Core data models: Neuron, Synapse, Fiber, Brain
- In-memory storage backend using NetworkX
- Temporal extraction for Vietnamese and English
- Query parser with stimulus decomposition
- Spreading activation algorithm
- Reflex retrieval pipeline
- Memory encoder
- FastAPI server with memory and brain endpoints
- Unit and integration tests
- Docker support

## [0.1.0] - TBD

### Added

- First public release
- Core memory encoding and retrieval
- Multi-language support (English, Vietnamese)
- REST API server
- Brain export/import functionality
