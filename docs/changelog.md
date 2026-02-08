# Changelog

All notable changes to NeuralMemory are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.19.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.19.0
[0.17.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.17.0
[0.16.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.16.0
[0.15.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.15.0
[0.7.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.7.0
[0.6.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.6.0
[0.5.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.5.0
[0.4.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.4.0
[0.3.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.3.0
[0.2.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.2.0
[0.1.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.1.0
