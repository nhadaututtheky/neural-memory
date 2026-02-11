# Changelog

All notable changes to NeuralMemory are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] - 2026-02-11

### Added

- **Proactive Brain Intelligence** — 3 features that make the brain self-aware during normal usage
  - **Related Memories on Write** — `nmem_remember` discovers and returns up to 3 related existing memories via 2-hop SpreadingActivation. Always-on, non-intrusive. Response includes `related_memories` list with `fiber_id`, `preview`, and `similarity`.
  - **Expired Memory Hint** — Health pulse detects expired memories via cheap COUNT query. Surfaces hint when count exceeds threshold (default: 10).
  - **Stale Fiber Detection** — Health pulse detects fibers with decayed conductivity (>90 days unused). Surfaces hint when stale ratio exceeds 30%.
- **MaintenanceConfig extensions** — `expired_memory_warn_threshold`, `stale_fiber_ratio_threshold`, `stale_fiber_days`
- **Storage layer** — `get_expired_memory_count()` and `get_stale_fiber_count()` on SQLite + InMemory backends
- **HealthPulse extensions** — `expired_memory_count` and `stale_fiber_ratio` fields
- **HEALTH_DEGRADATION trigger** — New trigger type for maintenance events

### Changed

- Tests: 1696 passed (up from 1695)

---

## [1.6.1] - 2026-02-10

### Fixed

- CLI brain commands (`export`, `import`, `create`, `delete`, `health`, `transplant`) now work correctly in SQLite mode
- `brain export` no longer produces empty files when brain was created with `brain create`
- `brain delete` correctly removes `.db` files in unified config mode
- `brain health` uses storage-agnostic `find_neurons()` instead of JSON-internal `_neurons` dict
- All `version` subcommands (`create`, `list`, `rollback`, `diff`) now find brains in SQLite mode
- `shared sync` uses correct storage backend

---

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
  - Structure-based join table detection (2+ FKs, ≤1 business column → CO_OCCURS synapse, no entity node)
  - 5 schema pattern detectors: audit_trail, soft_delete, tree_hierarchy, polymorphic, enum_table
  - Semantic entity descriptions with business purpose inference
- **Training Orchestrator** — `engine/db_trainer.py`
  - Mirrors DocTrainer architecture: batch save, per-table error isolation, shared domain neuron
  - Configurable: `max_tables` (1-500), `salience_ceiling`, `consolidate`, `domain_tag`
  - Optional ENRICH consolidation for cross-cluster linking
- **MCP Tool: `nmem_train_db`** — Train brain from database schema
  - `train` action: provide `connection_string` (SQLite only for v1), `domain_tag`, `max_tables`
  - `status` action: view trained schema entity count
  - Input validation: connection string length, SQLite-only scheme, max_tables bounds

### Fixed

- **Security: read-only SQLite connections** — Introspection uses `file:?mode=ro` URI
- **Security: absolute path rejection** — Connection strings reject absolute paths
- **Security: SQL identifier sanitization** — `_SAFE_IDENTIFIER` regex prevents injection
- **Security: info leakage prevention** — Error messages sanitized, no raw exceptions exposed
- **Security: handler input validation** — `max_tables` bounded 1-500, `consolidate` type-checked

### Changed

- MCP tools expanded from 17 to 18 (`nmem_train_db`)
- Version bumped to 1.6.0
- Tests: 1648 passed (up from 1596)

### Skills

- **3 composable AI agent skills** — ship-faster SKILL.md pattern, installable to `~/.claude/skills/`
  - `memory-intake` — structured memory creation from messy notes, 1-question-at-a-time clarification, batch store with preview
  - `memory-audit` — 6-dimension quality review (purity, freshness, coverage, clarity, relevance, structure), A-F grading
  - `memory-evolution` — evidence-based optimization from usage patterns, consolidation, enrichment, pruning, checkpoint Q&A

---

## [1.5.0] - 2026-02-10

### Added

- **Conflict Management MCP Tool (`nmem_conflicts`)** — Surface conflict detection as user-facing tool
  - `list` action: view active CONTRADICTS synapses with neuron content previews, conflict type, confidence
  - `resolve` action: manual resolution via `keep_existing`, `keep_new`, `keep_both` strategies
  - `check` action: pre-check content for potential conflicts before storing
  - `ConflictHandler` mixin class with full input validation (UUID, content length, tags)
  - Preserves `_pre_dispute_activation` for restore on `keep_existing` resolution
  - `_conflict_resolved` flag prevents re-flagging after manual resolution
  - Atomic resolution via `disable_auto_save → operations → batch_save()` pattern
- **Recall Conflict Surfacing** — `nmem_recall` now returns `has_conflicts` flag and `conflict_count` by default
  - Opt-in `include_conflicts=true` for full conflict details (saves AI client tokens)
- **Remember Conflict Reporting** — `nmem_remember` response includes `conflicts_detected` count when conflicts found during encoding
- **Stats Conflict Count** — `nmem_stats` response includes `conflicts_active` (unresolved CONTRADICTS count)
- **Provenance Source Enrichment** — `NEURALMEMORY_SOURCE` env var → `mcp:{source}` provenance with `mcp_tool` fallback
- **Purity Score Conflict Penalty** — Unresolved CONTRADICTS synapses reduce brain health purity score (max -10 points)
  - New diagnostic warning `HIGH_CONFLICT_COUNT` when conflicts exceed 5
  - Recommendation to run `nmem_conflicts` for resolution

### Fixed

- **`_resolve_keep_new` missing `_disputed=False`** — Existing neuron now properly cleared of dispute flag on `keep_new` resolution
- **Performance: filtered synapse queries** — `_conflicts_list`, `_conflicts_resolve`, and `_stats` now use `get_synapses(type=CONTRADICTS)` instead of `get_all_synapses()` full table scan
- **Diagnostics double fiber fetch** — `analyze()` now fetches fibers once and passes to both `_compute_freshness` and `_generate_diagnostics`
- **UUID validation case sensitivity** — UUID pattern now accepts uppercase hex (A-F) via `re.IGNORECASE`
- **Error message information leak** — All conflict handler errors now return generic messages; raw exceptions logged server-side only
- **Tag validation** — `nmem_conflicts check` and `nmem_remember` now validate tag count (max 50) and length (max 100 chars)
- **`NEURALMEMORY_SOURCE` truncation** — Source env var truncated to 256 chars to prevent unbounded metadata
- **Recall conflict field names** — Aligned with `nmem_conflicts list` (`existing_neuron_id`, `content` with 200-char truncation)
- **20+ performance bottlenecks** — Storage index optimization, encoder batch operations, retrieval pipeline improvements
- **25+ bugs across engine/storage/MCP** — Deep audit fixes including deprecated `datetime.utcnow()` replacement

### Changed

- `_compute_freshness` refactored from async instance method to `@staticmethod` accepting `fibers: list`
- MCP tools expanded from 16 to 17 (`nmem_conflicts`)
- `nmem_recall` schema gains `include_conflicts` boolean parameter
- Tests: 1372 passed (up from 1352)

---

## [1.4.0] - 2026-02-09

### Added

- **OpenClaw Memory Plugin** — NM becomes the memory layer inside OpenClaw (178k-star ecosystem)
  - `@neuralmemory/openclaw-plugin` npm package (TypeScript, zero runtime deps)
  - MCP stdio client: JSON-RPC 2.0 over stdio with Content-Length framing
  - 6 core tools registered: `nmem_remember`, `nmem_recall`, `nmem_context`, `nmem_todo`, `nmem_stats`, `nmem_health`
  - 2 hooks: `before_agent_start` (auto-context injection), `agent_end` (auto-capture)
  - Service registration: spawns `python -m neural_memory.mcp` as subprocess
  - Plugin manifest with `configSchema` + `uiHints` for OpenClaw settings UI
  - Zod parameter schemas for all tool inputs
- **Dashboard refactor** — Integrations tab simplified to status-only with deep links (Option B architecture)
  - Config forms removed — each service manages its own configuration
  - OpenClaw status card shows connection state when plugin is active

### Changed

- Architectural decision: NM Dashboard is a specialist tool, not a hub
- Dashboard Integrations tab now read-only status with deep links to service dashboards

---

## [1.3.0] - 2026-02-09

### Added

- **Deep Integration Status** — Richer integration monitoring in dashboard
  - Enhanced status cards with live metrics (memories/recalls today, last call timestamp, error badges)
  - Activity log: collapsible feed of recent tool calls with source attribution (MCP/OpenClaw/Nanobot)
  - Setup wizards: accordion config snippets for Claude Code, Cursor, OpenClaw, generic MCP with copy-to-clipboard
  - Import sources: detection panel for ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex
- **Source Attribution** — `NEURALMEMORY_SOURCE` env var → `session_id` prefix for integration tracking
- **i18n expansion** — 25 new keys in EN + VI (87 total)

### Changed

- Version bumped to 1.3.0
- Tests: 1352 passed (up from 1340)

---

## [1.2.0] - 2026-02-09

### Added

- **Dashboard** — Full-featured SPA at `/dashboard` (Alpine.js + Tailwind CDN, zero-build)
  - 5 tab sections: Overview, Neural Graph, Integrations (status-only), Health, Settings
  - Cytoscape.js neural graph: COSE force-directed layout, 8 neuron type colors, click → detail panel
  - Graph toolbar: search nodes, filter by type, zoom in/out, fit to view, reload
  - Chart.js radar chart for 7 health diagnostics metrics
  - Brain management: switch brains, export/import JSON, health grade sidebar badge
  - Toast notification system: `nmToast()` with 4 severity types, 4s auto-dismiss
  - Loading states: skeleton shimmer for stats, spinner for graph/health, proper empty states
  - Quick actions: Health Check, Export Brain, View Warnings on Overview tab
- **OAuth Proxy** — Routes to CLIProxyAPI (:8317) for OAuth session management
- **OpenClaw Config API** — CRUD from `~/.neuralmemory/openclaw.json`
- **EN/VI Internationalization** — Auto-detect browser locale, toggle in settings (68 translation keys)
- **Design System** — Dark mode (#0F172A), Fira Code/Sans fonts, #22C55E CTA green, Lucide icons
- **ARIA Accessibility** — `aria-label` on icon buttons, `role="tabpanel"/"tablist"`, `aria-live="polite"` on toasts
- **Mobile** — 44px minimum touch targets, responsive navigation

### Fixed

- **Issue #1**: `ModuleNotFoundError: typing_extensions` on fresh Python 3.12 — added `typing_extensions>=4.0` to dependencies
- **Ruff lint**: Fixed F821/I001 errors in `test_nanobot_integration.py`

### Changed

- Version bumped to 1.2.0
- `pyproject.toml` gains `httpx` and `typing_extensions` dependencies
- Tests: 1340 passed (up from 1264)

---

## [1.1.0] - 2026-02-09

### Added

- **ClawHub SKILL.md** — Published `neural-memory@1.0.0` to ClawHub (OpenClaw's skill registry, 2,999+ curated skills)
  - Instructs OpenClaw's agent to use NM via existing MCP server
- **Nanobot Integration** — Drop-in memory layer for HKUDS/nanobot framework
  - 4 tools adapted for Nanobot's action interface
- **OpenClaw Blog Post** — Comparison article: NM vs Mem0, Cognee, Graphiti, claude-mem
- **Architecture Doc** — `docs/ARCHITECTURE_V1_EXTENDED.md` for post-v1.0 architecture reference

### Changed

- OpenClaw PR [#12596](https://github.com/openclaw/openclaw/pull/12596) submitted — fixes `openclaw status` for third-party memory plugins (2-file fix, 5/5 tests pass)
- Post-v1.0 roadmap added: Dashboard + OpenClaw + Community strategy

---

## [1.0.2] - 2026-02-09

### Fixed

- **Empty recall for broad queries** — `format_context()` now truncates long fiber content to fit within token budget instead of skipping entirely. Queries that hit long anchors (e.g., architecture summaries) no longer return empty "## Relevant Memories" headers.
- **Diversity metric normalization** — Shannon entropy now normalized against 8 expected synapse types (was 20 total defined types). Realistic baseline for typical brain usage.
- **Temporal synapse diversity** — `_link_temporal_neighbors()` now creates BEFORE/AFTER synapses based on temporal ordering instead of always RELATED_TO.
- **Consolidation prune crash** — Fixed `_prune()` using `Fiber(..., tags=fiber.tags)` which would TypeError since `tags` is a computed `@property`. Now uses `dataclasses.replace()`.
- **Tag drift** — Ran one-time normalization sweep on existing fibers via `TagNormalizer.normalize_set()`.
- **Diversity warning threshold** — Warning triggers at `types_used < 3` (was `< 5`), reducing false positives.

## [1.0.0] - 2026-02-09

### Added

- **Brain Versioning** — Snapshot, rollback, and diff brain state (schema v11)
  - `BrainVersion` and `VersionDiff` frozen dataclasses
  - `VersioningEngine` with create, list, rollback, diff operations
  - `brain_versions` table with composite PK (brain_id, id), version numbering, SHA-256 snapshot hash
  - SQLiteVersioningMixin + InMemoryStorage support
  - `nmem version create|list|rollback|diff` CLI commands
  - `nmem_version` MCP tool (actions: create, list, rollback, diff)
- **Partial Brain Transplant** — Extract and merge filtered subgraphs between brains
  - `TransplantFilter` (tags, memory_types, neuron_types, min_salience)
  - `extract_subgraph()` pure function: filter fibers → collect neurons → filter synapses (both endpoints required)
  - `transplant()` async function: export → extract → merge → reimport with conflict resolution
  - `nmem brain transplant` CLI command with `--tag`, `--type`, `--strategy` options
  - `nmem_transplant` MCP tool
- **Brain Quality Badge** — Grade A-F derived from BrainHealthReport
  - `QualityBadge` frozen dataclass (grade, purity_score, marketplace_eligible, badge_label)
  - `compute_quality_badge()` method on `DiagnosticsEngine`
  - Marketplace eligibility threshold: grade B or above
- **Optional Embedding Layer** — Semantic similarity without mandatory LLM dependency (OFF by default)
  - `EmbeddingProvider` ABC with embed, embed_batch, dimension, cosine similarity
  - `SentenceTransformerEmbedding` — lazy-import, `all-MiniLM-L6-v2` default (384 dims)
  - `OpenAIEmbedding` — `text-embedding-3-small` default (1536 dims)
  - `EmbeddingConfig` frozen dataclass
  - 5 new BrainConfig fields: `embedding_enabled`, `embedding_provider`, `embedding_model`, `embedding_similarity_threshold`, `embedding_activation_boost`
  - Optional deps: `pip install neural-memory[embeddings]` or `neural-memory[embeddings-openai]`
- **Optional LLM Extraction** — Enhanced relation extraction beyond regex (OFF by default)
  - `ExtractionProvider` ABC with extract_relations, extract_entities
  - `ExtractionConfig` frozen dataclass (enabled=False, fallback_to_regex=True)
  - `RelationCandidate` frozen dataclass for extracted relations
  - `deduplicate_relations()` — case-insensitive dedup of LLM results against regex results

### Changed

- Version bumped to 1.0.0 — Production/Stable
- SQLite schema version 10 → 11
- `__all__` now exports `BrainVersion`, `VersionDiff`, `VersioningEngine`, `TransplantFilter`, `TransplantResult`
- MCP tools expanded from 14 to 16 (nmem_version, nmem_transplant)
- Classifier updated to "Development Status :: 5 - Production/Stable"

## [0.20.0] - 2026-02-09

### Added

- **Habitual Recall** — Knowledge creation, habit learning, and proactive workflow suggestions
  - **ENRICH consolidation strategy** — Transitive closure (A→B→C infers A→C) and cross-cluster linking via tag Jaccard similarity
  - **DREAM consolidation strategy** — Random exploration via spreading activation to discover hidden connections; dream synapses decay 10× faster
  - **LEARN_HABITS consolidation strategy** — Sequence mining on action event history to detect repeated workflows
  - **Action Event Log** — Lightweight hippocampal buffer (`action_events` table) for recording tool usage without graph bloat
  - **Sequence Mining Engine** — `mine_sequential_pairs()`, `extract_habit_candidates()`, `learn_habits()` for frequency-based pattern detection
  - **Workflow Suggestion Engine** — `suggest_next_action()` with dual threshold (weight + sequential_count) for proactive recommendations
  - **MCP tool `nmem_habits`** — suggest/list/clear actions for learned workflow habits
  - **CLI `nmem habits`** — list/show/clear subcommands for habit management
  - **Action recording** — Tool calls (remember, recall, context) automatically record action events
  - **Prune enhancements** — Dream synapses decay N× faster; high-salience fibers resist pruning
- Schema v10: `action_events` table with indexes on (brain_id, action_type), (brain_id, session_id), (brain_id, created_at)
- 6 new BrainConfig fields: `sequential_window_seconds`, `dream_neuron_count`, `dream_decay_multiplier`, `habit_min_frequency`, `habit_suggestion_min_weight`, `habit_suggestion_min_count`
- `EnrichmentResult`, `DreamResult`, `SequencePair`, `HabitCandidate`, `LearnedHabit`, `HabitReport`, `WorkflowSuggestion`, `ActionEvent` data structures

### Changed

- `ConsolidationStrategy` enum now includes `ENRICH`, `DREAM`, `LEARN_HABITS`
- `ConsolidationReport` now includes `synapses_enriched`, `dream_synapses_created`, `habits_learned`, `action_events_pruned`
- `ReflexPipeline.query()` optionally attaches `workflow_suggestions` to result metadata
- Schema version 9 → 10

## [0.19.0] - 2026-02-08

### Added

- **Temporal Reasoning** — Causal chain traversal, temporal range queries, and event sequence tracing
  - `CausalStep`, `CausalChain` frozen dataclasses for causal traversal results
  - `EventStep`, `EventSequence` frozen dataclasses for temporal event sequences
  - `trace_causal_chain()` — BFS traversal along CAUSED_BY/LEADS_TO synapses with cycle detection
  - `query_temporal_range()` — Retrieve fibers within a time window, sorted chronologically
  - `trace_event_sequence()` — BFS traversal along BEFORE/AFTER synapses with fiber timestamp enrichment
  - Chain confidence via product of step weights (natural decay for long chains)
- **Synthesis Methods** — `CAUSAL_CHAIN` and `TEMPORAL_SEQUENCE` added to `SynthesisMethod` enum
  - `format_causal_chain()` — "A because B because C" or "A leads to B leads to C"
  - `format_event_sequence()` — "First, A; then B; then C" with optional timestamps
  - `format_temporal_range()` — Chronological fiber summary list
- **Pipeline Integration** — Temporal reasoning fast-path in `ReflexPipeline.query()`
  - "Why?" queries → causal chain traversal → `CAUSAL_CHAIN` synthesis
  - "When?" queries with time hints → temporal range query → `TEMPORAL_SEQUENCE` synthesis
  - "What happened after X?" → event sequence tracing → `TEMPORAL_SEQUENCE` synthesis
  - Graceful fallback to standard activation pipeline when traversal finds no results
- **Router Enhancement** — Traversal metadata in `RouteDecision`
  - Causal queries annotated with `traversal: "causal"` and direction
  - Temporal queries classified as `temporal_range` or `event_sequence`
  - Event sequence patterns detected for English and Vietnamese queries

### Changed

- `SynthesisMethod` enum now includes `CAUSAL_CHAIN` and `TEMPORAL_SEQUENCE`
- `QueryRouter.route()` now populates `metadata` with traversal hints
- Tests: 1019 passed (up from 987)

## [0.17.0] - 2026-02-08

### Added

- **Brain Diagnostics** — Health metrics, purity score, and actionable recommendations
  - `BrainHealthReport` frozen dataclass with 7 component scores and composite purity (0-100)
  - `DiagnosticsEngine` computes connectivity, diversity, freshness, consolidation ratio, orphan rate, activation efficiency, recall confidence
  - Letter grade system (A/B/C/D/F) based on weighted purity score
  - Warning system with severity levels (CRITICAL/WARNING/INFO)
  - 7 diagnostic warning codes: EMPTY_BRAIN, STALE_BRAIN, LOW_CONNECTIVITY, LOW_DIVERSITY, HIGH_ORPHAN_RATE, NO_CONSOLIDATION, TAG_DRIFT
  - Automatic recommendations generated from detected warnings
  - Tag drift detection via `TagNormalizer.detect_drift()` integration
- **MCP tool: `nmem_health`** — Brain health diagnostics via MCP protocol
  - Returns grade, purity score, all component metrics, warnings, and recommendations
- **CLI command: `nmem health`** — Terminal-based health report with ASCII progress bars
  - Color-coded grade display and warning severity
  - `--json` flag for machine-readable output

## [0.16.0] - 2026-02-08

### Added

- **Emotional Valence** — Lexicon-based sentiment extraction with FELT synapses
  - `SentimentExtractor` — Pure-logic, zero-LLM sentiment analysis (EN + VI)
  - `Valence` enum (POSITIVE/NEGATIVE/NEUTRAL) and `SentimentResult` dataclass
  - Curated lexicons: ~80 positive + ~80 negative English words, ~30+30 Vietnamese
  - Negation handling with 2-token window ("not good" → negative)
  - Intensifier detection ("very", "extremely", "rất", "cực") → 1.5x intensity boost
  - 7 emotion tag categories: frustration, satisfaction, confusion, excitement, anxiety, relief, disappointment
  - `FELT` synapses created from anchor → emotion STATE neurons during encoding
  - Shared emotion neurons reused across fibers (find-or-create pattern)
  - Valence and intensity stored in synapse and fiber metadata
- **Emotional Resonance Scoring** — `emotional_resonance` component in `ScoreBreakdown`
  - Memories with FELT synapses get up to +0.1 retrieval score boost
  - Intensity-proportional: higher emotion → stronger retrieval signal
- **Emotional Decay Modulation** — FELT/EVOKES synapses decay slower
  - High-intensity emotions: decay^0.5 (much slower, emotional persistence)
  - Low-intensity emotions: decay^0.8 (slightly slower than normal)
  - Models biological trauma persistence and reward reinforcement
- **BrainConfig extension** — `emotional_decay_factor`, `emotional_weight_scale`

### Changed

- `ScoreBreakdown` now includes `emotional_resonance` field (0.0-0.1 range)
- Encoder pipeline now includes step 6a (sentiment extraction) between entity co-occurrence and relation extraction
- Tests: 950 passed (up from 908)

## [0.15.0] - 2026-02-08

### Added

- **Associative Inference Engine** — Pure-logic module that synthesizes co-activation patterns into persistent synapses
  - `compute_inferred_weight()` — Linear scaling capped at configurable max weight
  - `identify_candidates()` — Filters by threshold, deduplicates existing pairs, sorts by count
  - `create_inferred_synapse()` — Creates CO_OCCURS synapses with `_inferred: True` metadata
  - `generate_associative_tags()` — BFS clustering of co-activated neurons → named tags from content
- **Co-Activation Persistence** — Schema v8→v9
  - `co_activation_events` table storing individual events with canonical pair ordering (a < b)
  - `record_co_activation()`, `get_co_activation_counts()`, `prune_co_activations()` in storage layer
  - SQLite mixin (`SQLiteCoActivationMixin`) and InMemory implementation
  - Automatic persistence during retrieval via deferred write queue
- **INFER Consolidation Strategy** — New strategy in `ConsolidationEngine`
  - Creates CO_OCCURS synapses for neuron pairs with high co-activation counts
  - Reinforces existing synapses for pairs that already have connections
  - Generates associative tags from inference clusters
  - Prunes old co-activation events outside the time window
  - Dry-run support for previewing inference results
- **Enhanced PRUNE** — Inferred synapses with `reinforced_count < 2` get 2× accelerated decay
- **Tag Normalizer** — Synonym mapping + SimHash fuzzy matching + drift detection
  - ~25 canonical synonym groups (frontend, backend, database, auth, config, deploy, testing, etc.)
  - SimHash near-match with threshold=6 for short tag strings
  - `detect_drift()` reports multiple variants mapping to the same canonical
  - Integrated at encoder ingestion time — both auto-tags and agent-tags normalized
- **BrainConfig extension** — `co_activation_threshold`, `co_activation_window_days`, `max_inferences_per_run`
- **Memory Layer Unification Docs** — Guide for NM as a unification layer for external memory systems
  - Architecture diagram and pipeline walkthrough
  - Quick-start examples for all 6 adapters (ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex, AWF)
  - Custom adapter guide, incremental sync, provenance tracking

### Changed

- SQLite schema version bumped from 8 to 9
- Encoder normalizes all tags (auto + agent) via TagNormalizer before storage
- Consolidation report now includes `synapses_inferred` and `co_activations_pruned`
- Tests: 908 passed (up from 838)

## [0.14.0] - 2026-02-08

### Added

- **Relation Extraction Engine** — Auto-create causal, comparative, and sequential synapses from text
  - `RelationExtractor` with regex patterns for 3 relation families (EN + VI)
  - Causal: "because", "caused by", "due to", "therefore", "vì", "nên", "do đó" → `CAUSED_BY`, `LEADS_TO`
  - Comparative: "similar to", "better than", "unlike", "khác với" → `SIMILAR_TO`, `CONTRADICTS`
  - Sequential: "then", "after", "before", "sau khi", "trước khi" → `BEFORE`, `AFTER`
  - `RelationCandidate` frozen dataclass with confidence scoring (0.65–0.85 range)
  - Zero LLM — pure regex pattern matching
- **Tag Origin Tracking** (Expert E4 feedback)
  - `Fiber.auto_tags` — content-derived tags from entity/keyword extraction
  - `Fiber.agent_tags` — user-provided tags
  - `Fiber.tags` property — backward-compatible union of both sets
  - Storage: JSON format `{"auto": [...], "agent": [...]}`
- **Confirmatory Weight Boost** (Hebbian tag confirmation, Expert E4)
  - Agent tag matching auto-tag → +0.1 anchor synapse weight boost (capped at 1.0)
  - Divergent agent tags → new `RELATED_TO` synapse with weight 0.3 (provisional)
- **Auto Memory Type Inference** — `suggest_memory_type()` integrated into encoder as fallback

### Fixed

- **Event loop lifecycle** — `asyncio.run()` replaced with `run_async()` helper for proper aiosqlite cleanup
- **"Event loop is closed" noise** — aiosqlite connections close before event loop teardown

### Changed

- SQLite schema version 7 → 8 (auto_tags/agent_tags columns)
- Encoder pipeline gains steps 6b (relation extraction) and 6c (confirmatory boost)
- Tests: 838 passed (up from 776)

---

## [0.13.0] - 2026-02-07

### Added

- **Cognitive Runtime Phases 1–4** (v0.10.0 → v0.13.0 shipped as single release)
  - **Phase 1 (Hebbian Learning Rule)** — Formal novelty adaptation, natural weight saturation, competitive normalization, anti-Hebbian update for conflict resolution
  - **Phase 2a (Activation Stabilization)** — Iterative dampening, multi-neuron answer reconstruction with 3 strategies (single, fiber-summary, multi-neuron)
  - **Phase 2b (Memory Maturation)** — STM → Working → Episodic → Semantic lifecycle with stage-aware decay multipliers, spacing effect for semantic promotion, pattern extraction from episodic clusters
  - **Phase 3 (Conflict Detection)** — Real-time conflict detection at encode time via regex predicate extraction, dispute resolution with anti-Hebbian confidence reduction, `CONTRADICTS` synapse type, retrieval deprioritization of disputed neurons
  - **Phase 4 (Evaluation Benchmarks)** — 30 ground-truth memories, 25 queries across 5 categories, standard IR metrics (P@K, R@K, MRR, NDCG@K), naive keyword baseline, long-horizon coherence test framework
- Schema migration v5 → v7

### Changed

- Tests: 776 passed (141 new, up from 635)

---

## [0.9.5] - 2026-02-06

### Added

- **Type-Aware Decay** — Facts persist (0.02/day), TODOs expire fast (0.15/day), configurable per memory type
- **Score Breakdown** — Expose confidence components in retrieval results and MCP response
- **SimHash Deduplication** — 64-bit locality-sensitive hashing for near-duplicate detection in encoder and auto-capture
- **Point-in-Time Temporal Queries** — `valid_at` parameter filters fibers by time window
- Schema migration v4 → v5 (content_hash column)

### Changed

- Tests: 613 passed (up from 584)

---

## [0.9.3] - 2026-02-06

### Added

- **Eternal Context System** — 3-tier auto-save (critical/session/context) with file-based JSON persistence
  - `BrainPersistence` for `~/.neuralmemory/eternal/<brain_id>/`
  - `TriggerEngine` with auto-save on decisions, errors, milestones, checkpoints
  - `nmem_eternal` MCP tool (status/save/load/compact)
  - `nmem_recap` MCP tool (level 1-3, topic filtering)
  - VS Code commands: recap, recapTopic, eternalSave, eternalStatus
- **External Adapters** — Cognee, Graphiti, LlamaIndex adapters for symbiotic integration
  - Adapter registry (6 adapters total), MCP tool schema, server kwargs mapping
  - VS Code import command with source picker UI

### Fixed

- **Security hardening** — `max_tokens` clamped to 10000, env var preference for API keys
- **CLI refactor** — `remember()` 157→48 lines, `recall()` 121→50 lines, `brain_health()` 131→35 lines
- **MCP server** — Split into handler mixins, eliminate bare excepts

### Changed

- Tests: 584 passed (up from 546)

---

## [0.9.1] - 2026-02-06

### Changed

- VS Code extension synced with codebase indexing features

---

## [0.9.0] - 2026-02-06

### Added

- **Codebase Indexing** — Index Python codebases into neural graph for code-aware recall
  - `nmem_index` MCP tool, `nmem index` CLI
  - `PythonExtractor` (stdlib ast), `CodebaseEncoder`, `GitContext` utility
  - Branch-aware sessions auto-detect git branch/commit/repo
- **Smart Auto-Capture** — Brain grows naturally through MCP usage
  - Auto-capture insights, decisions, errors from recall queries (passive learning)
  - `INSIGHT_PATTERNS` for EN + VI "aha moments"
  - Passive capture on `nmem_recall` (≥50 char queries, confidence 0.8)
- **Enhanced Stats** — Hot neurons, DB size, daily activity, synapse stats by type, neuron type breakdown
- **Consolidation Engine** — Prune (dead synapses + orphan neurons), merge (Jaccard similarity), summarize (tag clusters)
  - `nmem consolidate` CLI with `--strategy` and `--dry-run`
  - POST `/brain/{id}/consolidate` API endpoint
- **Conflict Resolution** — Pure `merge_snapshots()` with 4-phase algorithm and provenance tracking
- **Extraction Intelligence** — Parallel activation (~3x speedup), scored intent, weighted keywords, code entity detection
- **Intelligence Upgrade** — Multi-factor confidence scoring, frequency-boosted activation, batch neuron fetch

### Fixed

- **Server storage** — Replace InMemoryStorage with `get_shared_storage()` (data persists across restarts)
- **Brain name fallback** — `find_brain_by_name()` when `X-Brain-ID` is name, not UUID

### Changed

- Schema migration v3 → v4 (fiber_neurons junction table)
- API versioned routes (`/api/v1/`) with backward-compatible legacy routes
- Tests: 503 passed (up from 431)

---

## [0.8.0] - 2026-02-05

### Added

- **Hebbian Plasticity** — Co-activated neurons auto-strengthen synaptic connections during retrieval
  - `hebbian_delta`, `hebbian_threshold`, `hebbian_initial_weight` in BrainConfig
  - `_strengthen_co_activated` hook in ReflexPipeline after fiber conduction
  - `consolidate()` method in DecayManager for boosting high-frequency fibers
- **FTS5 Full-Text Search** — BM25 ranked retrieval with Porter stemming for neuron content
  - `neurons_fts` virtual table with auto-sync triggers
  - Multi-word implicit AND search, graceful LIKE fallback
  - Schema migration v2 → v3

### Changed

- Tests: 431 passed (up from 413)

---

## [0.7.2] - 2026-02-05

### Fixed

- **Schema migration for old databases** — Existing `default.db` from v1 missing `conductivity`, `pathway`, and `last_conducted` columns in `fibers` table, causing `nmem remember` to crash with `OperationalError`. Now auto-migrates on startup via `ALTER TABLE ADD COLUMN`.
- Added migration framework (`MIGRATIONS` dict + `run_migrations()`) for future schema upgrades

## [0.7.1] - 2026-02-05

### Added

- **Zero-config `nmem init`** — One command sets up everything:
  - Creates `~/.neuralmemory/config.toml` and default brain
  - Auto-configures MCP for Claude Code (`~/.claude/mcp_servers.json`)
  - Auto-configures MCP for Cursor (`~/.cursor/mcp.json`)
  - Safe JSON merging (never overwrites existing MCP entries)
  - `--skip-mcp` flag to opt out of auto-configuration
  - Formatted summary output with [OK]/[--]/[!!] status icons

## [0.7.0] - 2026-02-05

### Added

- **VS Code Extension** (`vscode-extension/`) — Visual brain explorer and memory management
  - Memory tree view in activity bar sidebar with neurons grouped by type
  - Interactive graph explorer with Cytoscape.js force-directed layout
  - Encode commands: selected text or typed input as memories
  - Recall workflow with depth selection (Instant, Context, Habit, Deep)
  - CodeLens integration with memory counts on functions/classes
  - Comment trigger detection (`remember:`, `note:`, `decision:`, `todo:`)
  - Brain management via status bar and command palette
  - Real-time WebSocket sync for tree, graph, and status bar
  - Configurable settings (server URL, Python path, graph node limit)
  - Status bar with live brain stats (neurons, synapses, fibers)
  - Extension icons, README, CHANGELOG, and VSIX packaging

### Changed

- **Enforced 500-line file limit** — Split 8 oversized files into modular sub-modules
  - `sqlite_store.py` (1659 lines) → 9 files: schema, row mappers, neurons, synapses, fibers, typed memories, projects, brain ops, core store
  - `memory_store.py` (906 lines) → 3 files: brain ops, collections, core store
  - `activation.py` (672 lines) → 2 files: spreading activation + reflex activation
  - `shared_store.py` (650 lines) → 3 files: mappers, fiber/brain mixin, core store
  - `retrieval.py` (639 lines) → 3 files: types, context formatting, pipeline
  - `mcp/server.py` (694 lines) → 3 files: tool schemas, auto-capture, server
  - `entities.py` (547 lines) → 2 files: entities + keywords
  - `GraphPanel.ts` (771 lines) → 2 files: template + panel controller
- All source files now under 500 lines per CLAUDE.md coding standards
- Backward-compatible re-exports maintain existing import paths

## [0.6.0] - 2026-02-05

### Added

- **Reflex-Based Retrieval** - True neural reflex activation
  - `ReflexActivation` class with trail decay along fiber pathways
  - `CoActivation` dataclass for Hebbian binding ("neurons that fire together wire together")
  - Time-first anchor finding - temporal context constrains all other signals
  - Co-activation intersection for neurons activated by multiple anchors
- **Fiber as Signal Pathway**
  - `pathway` field - ordered neuron sequence for signal conduction
  - `conductivity` field - signal transmission quality (0.0-1.0)
  - `last_conducted` field - when fiber last conducted a signal
  - `conduct()` method - update fiber after signal transmission
  - `with_conductivity()` method - immutable conductivity update
  - Pathway helper methods: `pathway_length`, `pathway_position()`, `is_in_pathway()`
- **Trail Decay Formula** - `level * (1 - decay) * synapse.weight * fiber.conductivity * time_factor`
- **ReflexPipeline Updates**
  - `use_reflex` parameter to toggle between reflex and classic activation
  - `_find_anchors_time_first()` for time-prioritized anchor finding
  - `_reflex_query()` for fiber-based activation with co-binding
  - `co_activations` field in `RetrievalResult`

### Changed

- SQLite schema version bumped to 2 (added fiber pathway columns)
- Fiber model now includes signal pathway fields
- `RetrievalResult` includes co-activation information

## [0.5.0] - 2026-02-05

### Added

- **Documentation Site** - MkDocs with Material theme
  - Getting started guides (installation, quickstart, CLI reference)
  - Concept documentation (how it works, neurons, synapses, memory types)
  - Integration guides (Claude, Cursor, Windsurf, Aider)
  - API reference (Python, Server)
  - Architecture documentation
  - Contributing guide
- GitHub Actions workflow for automatic docs deployment
- Interactive logo and assets

### Changed

- Documentation URL now points to GitHub Pages
- Reorganized docs structure for better navigation

## [0.4.0] - 2026-02-05

### Added

- **Web UI Visualization** - Interactive brain graph at `/ui` endpoint
  - vis.js-based network visualization
  - Color-coded nodes by neuron type
  - Click nodes to see details
  - Statistics display
- **API endpoint** `/api/graph` for visualization data
- Fiber information in graph response

### Changed

- Server now includes visualization endpoints by default
- Updated dependencies

## [0.3.0] - 2026-02-05

### Added

- **Scheduled Memory Decay** - Ebbinghaus forgetting curve implementation
  - `DecayManager` class in `engine/lifecycle.py`
  - Configurable decay rate and prune threshold
  - Dry-run mode for previewing changes
- **CLI `decay` command** - Apply decay with `nmem decay --dry-run`
- `ReinforcementManager` for strengthening accessed paths
- `get_all_neuron_states()` method in storage backends
- `get_all_synapses()` method in storage backends

### Changed

- Improved neuron state management
- Better activation level tracking

## [0.2.0] - 2026-02-05

### Added

- **MCP System Prompt** - AI instructions in `mcp/prompt.py`
  - Full and compact prompt versions
  - Resource URIs for prompts
- **Auto-capture** - `nmem_auto` tool with "process" action
  - One-call memory detection and saving
  - Configurable confidence threshold
- **MCP Resources** - `neuralmemory://prompt/system` and `neuralmemory://prompt/compact`
- CLI commands: `prompt`, `mcp-config`

### Changed

- MCP server now exposes 6 tools (was 5)
- Improved auto-detection of memory types

## [0.1.0] - 2026-02-05

### Added

- **Core Data Structures**
  - `Neuron`, `NeuronType`, `NeuronState`
  - `Synapse`, `SynapseType`, `Direction`
  - `Fiber` - Memory clusters
  - `Brain`, `BrainConfig` - Container and configuration
  - `TypedMemory`, `MemoryType`, `Priority`
  - `Project` - Project scoping

- **Storage Backends**
  - `InMemoryStorage` - NetworkX-based for testing
  - `SQLiteStorage` - Persistent single-user storage
  - `SharedStorage` - HTTP client for remote server

- **Engine**
  - `MemoryEncoder` - Text to neural structure
  - `ReflexPipeline` - Query with spreading activation
  - `SpreadingActivation` - Core retrieval algorithm
  - `DepthLevel` - Retrieval depth control

- **Extraction**
  - `QueryParser` - Query decomposition
  - `QueryRouter` - Intent and depth detection
  - `TemporalExtractor` - Time reference parsing (EN + VI)

- **CLI Commands**
  - `remember`, `recall`, `todo`, `context`, `list`
  - `stats`, `check`, `cleanup`
  - `brain list/create/use/export/import/delete/health`
  - `project create/list/show/delete/extend`
  - `shared enable/disable/status/test/sync`
  - `serve`, `mcp`, `export`, `import`

- **Server**
  - FastAPI-based REST API
  - Memory and brain endpoints
  - WebSocket sync support
  - CORS configuration

- **MCP Server**
  - `nmem_remember` - Store memories
  - `nmem_recall` - Query memories
  - `nmem_context` - Get recent context
  - `nmem_todo` - Quick TODO
  - `nmem_stats` - Brain statistics

- **Features**
  - Multi-language support (English + Vietnamese)
  - Memory types with auto-expiry
  - Priority system (0-10)
  - Sensitive content detection
  - Brain export/import
  - Real-time sharing

## [Unreleased]

### Planned

- Neo4j storage backend
- Semantic search with embeddings
- Memory compression for old fibers
- Admin dashboard
- Brain marketplace

---

[1.7.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v1.7.0
[1.6.1]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v1.6.1
[1.6.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v1.6.0
[1.5.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v1.5.0
[1.0.2]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v1.0.2
[1.0.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v1.0.0
[0.20.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.20.0
[0.19.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.19.0
[0.17.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.17.0
[0.16.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.16.0
[0.15.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.15.0
[0.14.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.14.0
[0.13.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.13.0
[0.9.5]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.9.5
[0.9.3]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.9.3
[0.9.1]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.9.1
[0.9.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.9.0
[0.8.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.8.0
[0.7.2]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.7.2
[0.7.1]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.7.1
[0.7.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.7.0
[0.6.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.6.0
[0.5.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.5.0
[0.4.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.4.0
[0.3.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.3.0
[0.2.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.2.0
[0.1.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.1.0
