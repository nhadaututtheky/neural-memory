# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
