# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.52.2] — 2026-04-20

### Improved — DREAM Hubs Now Consumed by Retrieval

Closes the Section 9 integration gap **DREAM hubs + graph density scaling** — hub synapses were being *written* by consolidation but never *read* back by retrieval.

- **Graph density excludes DREAM hubs by default for strategy selection.** `get_graph_density()` grows an `exclude_hubs: bool = False` parameter on both the SQL mixin (`storage/sql/mixins/calibration.py`) and the legacy SQLite backend (`storage/sqlite_calibration.py`). When True, it filters out synapses whose metadata contains `_hub=True` via `json_extract(metadata, '$._hub') IS NULL` / `metadata->>'_hub' IS NULL`. `retrieval._auto_select_strategy()` now calls with `exclude_hubs=True` so DREAM's synthesized hub links don't inflate density and trick the engine into picking PPR on graphs that are organically sparse.
- **PPR dampens hub edges during push.** `PPRActivation` multiplies the effective weight of `_hub=True` synapses by `BrainConfig.hub_edge_dampening` (default `0.5`) when building the neighbor cache. Hub edges still carry activation — they just can't hijack random walks at the expense of genuine edges. Setting the config to `1.0` disables the dampening for users who want the pre-v4.52.2 behavior.
- **New config field.** `BrainConfig.hub_edge_dampening: float = 0.5`. Documented inline.

### Docs

- `.rune/FEATURE_REGISTRY.md` Section 2c (DREAM hub extraction) and Section 9 (DREAM hubs + graph density scaling) updated to reflect the fix — the gap moves from OPEN → FIXED v4.52.2.

### Tests

- `tests/unit/test_v4_52_2_hub_aware_retrieval.py` — 8 tests: density computation with/without hub exclusion on a real SQLite brain, `_auto_select_strategy` passes `exclude_hubs=True`, PPR hub dampening (hub target gets less activation than plain target at equal base weight), dampening disabled when factor=1.0, default config value.

## [4.52.1] — 2026-04-20

### Improved — Activation Decay Integrated into Consolidation

- **`DECAY` is now a first-class consolidation strategy (Tier 0)**. Previously the Ebbinghaus decay pass only ran on the scheduled 12h cycle, so every consolidation between those cycles worked off stale activation + synapse weights. Old memories kept their full activation and crowded fresh ones out of recall. `ConsolidationEngine` now runs `DecayManager.apply_decay()` as a dedicated first tier before `PRUNE` — so PRUNE sees the actually-decayed activation and can drop items below its threshold on the same run.
- **Safe by construction.** DECAY sits in its own frozenset tier (before the PRUNE/LEARN_HABITS/DEDUP tier) so execution order is explicit, not frozenset-hash-dependent. `min_age_days` in `DecayManager` still guards against double-decaying recently-touched memories. `dry_run=True` propagates correctly — the report records stats without persisting changes. Failures from the decay pass are logged and swallowed (`logger.warning`) so a storage backend issue cannot take consolidation down with it.
- **Report surface.** `ConsolidationReport.extra["decay"]` carries `{neurons_processed, neurons_decayed, synapses_processed, synapses_decayed, duration_ms}` so callers (dashboard, MCP clients, pre-ship checks) can inspect what the decay pass did without a second round-trip.

### Docs

- `.rune/FEATURE_REGISTRY.md` Section 10 updates: freshness weight tweaks marked DONE (already at 15% / 0.15 default on `BrainConfig`) — the "stale" audit from the v4.52.0 review found these shipped separately. Decay ↔ consolidation gap moves from OPEN → FIXED. Context compiler keyword boost marked DONE (already case-normalized).

### Tests

- `tests/unit/test_v4_52_1_decay_in_consolidation.py` — 7 tests covering the DECAY enum, tier ordering (DECAY < PRUNE), dispatch wiring, report surface (DECAY alone + ALL), dry_run propagation, and non-fatal failure handling.

## [4.52.0] — 2026-04-20

### Improved — 3 Cross-Feature Wirings

Closes three integration gaps flagged in Section 9 of `.rune/FEATURE_REGISTRY.md`. Each is small in LOC but meaningful: features that already existed now actually talk to each other.

- **Dynamic abstraction → stratum MMR** (Section 2c/Section 2 integration). `_apply_mmr_diversity` now tracks `abstraction_counts` alongside `schema_counts`. Fibers anchored on a CONCEPT neuron with `_abstraction_induced=True`, or carrying `_abstract_neuron_id` from MERGE consolidation, are capped per-cluster using the same `max_per_stratum` budget. Prevents a single "super-abstract" from dominating top-K results.
- **Vietnamese keyword extraction → query_expander** (Section 2b/Section 2 integration). `expand_terms()` now accepts a `language` parameter. When set to `"vi"` (or auto-detected via Vietnamese diacritics), multi-word phrases (3+ tokens) are run through pyvi's `ViTokenizer` to extract compound tokens — so "học sinh giỏi nhất" now surfaces "học_sinh" as an expansion candidate. Graceful no-op when pyvi is not installed. Wired through `stimulus.language` in `RecallPipeline`.
- **Abstraction → priming** (Section 2e integration). `prime_from_topics` and `prime_from_habits` now apply a +25% boost (`ABSTRACTION_BOOST_MULT = 1.25`) to neurons whose metadata carries `_abstraction_induced=True`. Concept-level summaries surface before raw episodes in primed recall rounds.

### Tests

- `tests/unit/test_v4_52_wirings.py` — 9 tests pinning each wiring independently so future edits don't silently regress the integrations.

### Registry

- `.rune/FEATURE_REGISTRY.md` Section 9 updates: the three gaps above move from OPEN → FIXED. Three other gaps remain (DREAM hubs + graph density, auto-capture + FastAPI, agent ID + provenance) — queued for future versions.

## [4.51.4] — 2026-04-19

### Fixed

- **Issue #132 — `nmem recall --limit` restored** (Bé Mi): `recall` now accepts `--limit` / `-l N` again as an approximate cap. Internally mapped to `max_tokens = max(100, N * 200)` and truncates the displayed `fibers_matched` list to `N`. Default `None` preserves previous behaviour when the flag is omitted.
- **Issue #132 — pyvi warning leak**: `_tokenize_vietnamese()` and the auto-capture pyvi probe now wrap the import in `warnings.simplefilter("ignore")`. The previous `DeprecationWarning` filter missed `numpy.VisibleDeprecationWarning`, which is a `UserWarning` subclass, so pyvi's first tokenization still emitted a warning. Broadened to silence all warnings during the pyvi import path.
- **Issue #132 — doctor schema-migration hint**: the auto-fix message for "Schema version mismatch" no longer claims a `--fix` flag exists. Now reads: `"Auto-migrates on next read/write — run any command (e.g. 'nmem recall \"test\"' or 'nmem doctor --fix') to trigger now"`.

### Improved

- **Issue #132 — doctor tiered output**: `nmem doctor` now groups findings into three priority tiers so users can distinguish "broken" from "missing optional":
  - `CORE` — schema, Python version, brain database, permissions (must be green)
  - `RECOMMENDED` — hooks, MCP server (should be green for best UX)
  - `OPTIONAL` — Pro features, hub status (informational)
  Each tier renders under its own header. Exit code stays driven by core-issue count only.

### Tests

- `tests/unit/test_recall_limit.py` — regression test confirming `recall()` exposes a `--limit` parameter with `None` default and `int | None` annotation.
- `tests/unit/test_vietnamese_keywords.py::test_tokenize_does_not_leak_pyvi_warnings` — asserts no pyvi/numpy warnings escape `_tokenize_vietnamese()` (skips if pyvi not installed).
- `tests/unit/test_doctor_enhanced.py` — new `TestPriorityTiers` + `TestSchemaMigrationHint` classes covering tier assignments and the new migration message.

## [4.51.3] — 2026-04-19

### Changed

- **Storage ABC slim-down** — split `NeuralStorage` (1900-LOC monolithic ABC) into a layered hierarchy without touching any caller or backend:
  - `CoreStorage(ABC)` — 29 genuinely abstract methods covering CRUD, graph traversal, fibers, brain meta, lifecycle, stats. `_get_brain_id()` now ships with a concrete default (reads `self.brain_id`, raises `ValueError` if unset). A new backend only needs to fill these 29 slots to be usable, and ABC enforces the contract at instantiation time.
  - `ExtendedStorage` (plain class) — optional domains (typed memory, alerts, drift, versioning, sync hub, bulk enumeration, ghost tracking, etc.) with every method defaulting to `raise NotImplementedError`. Backends override piecewise.
  - `NeuralStorage(CoreStorage, ExtendedStorage)` — canonical union used throughout the codebase; existing inheritance chains (`SQLiteStorage`, `SQLStorage`, `PostgreSQLStorage`, `InMemoryStorage`, `SharedStorage`) are unchanged.
- Tightened contract after review: `set_brain` is now a true `@abstractmethod` (previously a stub); `get_all_neuron_states`, `get_all_synapses`, `find_brain_by_name`, and `batch_update_ghost_shown` moved from CoreStorage stubs to `ExtendedStorage` (truthful tier — `SharedStorage` and similar proxy backends genuinely do not support them).

## [4.51.2] — 2026-04-19

### Added

- **Storage Evolution Phase 3 — Consolidation + Abstraction Upgrade**:
  - **Causal prune guard**: `PRUNE` strategy now skips `CAUSED_BY` / `LEADS_TO` / `ENABLES` / `PREVENTS` synapses unless they carry `_inferred=True`. Causal knowledge survives decay sweeps; auto-inferred causal noise remains prunable.
  - **Dynamic abstraction induction**: new `engine.abstraction.induce_abstraction(cluster)` condenses a cluster of episodic neurons into one CONCEPT neuron (abstraction level 2) using stopword-filtered term-frequency extraction. Content template: `"[N] memories about [TOPIC]: [TERMS]. Key: [exemplar]"`. Exemplar chosen by highest `goal_priority` (ties by recency). Metadata traces `_abstraction_source_ids`, `_abstraction_terms`, `_abstraction_exemplar_id`, `_abstraction_induced=True`.
  - **MERGE strategy wiring**: large summaries (cluster size ≥ `abstraction_cluster_min_size`, default 5) now persist the abstract neuron and wire `IS_A` links (weight 0.6) from each cluster exemplar → abstract. Gated by `ConsolidationConfig.enable_dynamic_abstraction` (default True). `ConsolidationReport.concepts_created` surfaces the count.
  - **DREAM hub extraction**: post-delegation pass identifies neurons participating in ≥3 consolidation events, creates `RELATED_TO` synapses between the top two hubs with `_hub=True` / `_semantic_discovery=True` metadata. Feeds `report.patterns_extracted`.
  - **REPLAY semantic census**: queries all `CONCEPT` neurons (limit 500), counts those with `_abstraction_induced=True`, stores snapshot in `report.extra`.

### Tests

- `tests/unit/test_storage_evolution_phase3.py` — 14 tests covering causal prune guard set contents, inferred-causal exclusion, manual-causal protection, `induce_abstraction` template + metadata + stopword/short-token filtering + exemplar priority + content truncation, and MERGE config surfaces.

## [4.51.1] — 2026-04-19

### Fixed

- **CI build unblocked**: `.gitignore` rule `lib/` was over-greedy — it matched `dashboard/src/lib/`, preventing `neuron-colors.ts` (imported by Living Brain 3D + NetworkGraph) from being committed. Tightened to `/lib/` so only root Python build dir is ignored.
- **Dashboard TypeScript build**: replaced `ElementType` with Phosphor's `Icon` type in 5 files (Sidebar, InsightsPage, OverviewPage, QuickActionsCard, TimelinePage) — React 19 types collapsed `ElementType` JSX props to `never`, breaking `className` assignment.
- **BrainCanvas OrbitControls**: wrapped `invalidate` in an arrow so its `(frames?: number)` signature no longer clashes with OrbitControls' `onChange(e?: Event)` contract.

## [4.51.0] — 2026-04-18

### Added

- **Living Brain 3D (Pro)**: new dashboard visualization — interactive 3D brain graph where neurons sit inside a translucent brain shell, clustered by cortical zone. Built on `@react-three/fiber` + `@react-three/drei` + `d3-force-3d`. Routes: `/living-brain`, with `ModeToggle` to switch between 2D Sigma and 3D.
- **Interactivity**: click-to-inspect `NodeDetailPanel` (neighbors, connections, ID), keyboard navigation (`ArrowRight`/`Left`/`Up`/`Down` walk the graph via `useKeyboardNav`), `?focus=<id>` deep-link two-way binding (`useFocusDeepLink`), hover + selection highlight with proximity lerp.
- **Activation stream**: client-side simulation of "living" brain pulses — delta pulses on layout change + ambient degree-weighted pulses every ~2.5s (±25% jitter). Pulses ride the r3f `frameloop="demand"` via kick-tick invalidation. Reduced-motion users see a still brain.
- **Stats HUD**: bottom-left neuron / synapse / active / pulses count with `useCountUp` ease-out-cubic animation, tabular-nums monospace.
- **Pro gate**: free users see a Pro upsell card; `useLivingBrain` + `useActivationStream` never mount for non-Pro users (no `/api/graph` fetch, no background timers).
- **Share PNG**: `ShareBrain` exports the canvas as a watermarked PNG (1920px max-edge cap, "Neural Memory" watermark, `preserveDrawingBuffer: true` + imperative `gl.render` before `toBlob` so the buffer isn't stale).
- **Settings drawer**: toggles for effects / brain shell / activation pulse / labels. Zustand `persist` with `version: 1` + `localStorage` key `nm.livingBrain.settings`. Click-outside + Esc dismissal, focus return to trigger.
- **i18n**: full EN + VI parity for all Living Brain strings (stats, settings, share, upsell, error).

### Improved

- **P3 review fixes** (interactivity phase): SynapseEdges geometry disposal via `key={edges.length}`, dead `?focus=` URL cleanup when layout doesn't know the id, dead-selection clear on layout swap, `onPointerLeave` belt-and-braces for hover clear, focus restore on detail panel close, sphere tessellation bumped to `[1, 16, 16]`.
- **P4 review fixes** (live phase): Pro gate leak plugged (hooks only mount for Pro), disable-pulse repaints cleared state via kick-tick, share export forces a fresh render via `RendererRegistrar` before reading pixels, `revokeObjectURL` deferred to avoid Firefox/Safari download cancel, `useCountUp` interrupt-safe (starts new tween from live display, not stale start), settings drawer focus management, `AutoRotate` gated by effects setting so the demand frameloop truly idles.

### Tests

- No new Python tests (pure frontend feature). Dashboard chunk 1.03 MB / 279 kB gz, lazy-loaded so non-Pro pages stay trim.

## [4.50.0] — 2026-04-18

### Added

- **Sparse Selective Restore (SSC-lite)**: warm-start recall now ranks cached neurons by query cosine similarity and keeps only the top-K (K=20) most relevant. Stale warm activations no longer pollute recall.
- **Cache invalidation**: new `InvalidationTracker` + `apply_invalidation` pipeline. Partial invalidation drops only dirty neurons; full invalidation triggers on consolidation or ≥25% dirty ratio; `detect_staleness` compares brain hashes.
- **`cache/selector.py`**: reusable cosine-ranking utility with activation-ranked fallback, bounded storage concurrency (semaphore=20), dimension-mismatch warning.
- **`cache/invalidation.py`**: event-driven change accumulator (neuron_add, neuron_delete, synapse_change, consolidation) + immutable `remove_neurons` transform.

### Improved

- **`ActivationCacheManager.get_warm_activations_selective`**: bounds-clamped API (`top_k` to `[1, max_entries]`, `min_similarity` to `[-1.0, 1.0]`) per project convention.
- **Warm-start wiring**: `recall_handler` now uses SSC-lite instead of returning all cached neurons; exception path logs instead of silently swallowing.
- **Encapsulation**: `_replace_cache()` method added to manager — invalidation module no longer writes `manager._loaded_cache` directly.

### Tests

- 33 new Phase 3 tests covering selector ranking, fallbacks, dimension mismatch, invalidation tracker, partial/full invalidation, hash-based staleness detection, bounds clamping, and the integration contract (`ReflexPipeline._embedding_provider` exists).
- 61 total activation-cache tests pass (28 Phase 1-2 + 33 Phase 3).

## [4.49.0] — 2026-04-15

### Added

- **Negation conflict detection**: detects contradictions like "we use Redis" vs "we do NOT use Redis" — new `NEGATION_CONFLICT` type with 11 negation patterns (not, never, stopped, no longer, removed, disabled, dropped, deprecated, don't, won't, no)
- **Temporal supersession**: conflicts are now classified as `SUPERSEDED` (>24h gap), `SAME_SESSION_CORRECTION` (<1h), or `TRUE_CONTRADICTION` — superseded conflicts auto-resolve as keep_new
- **Entity matching**: tech term extraction via CamelCase/kebab-case/acronym regex + alias normalization (postgres→postgresql, k8s→kubernetes, mongo→mongodb)
- **Multi-factor tier promotion** (v4.48): composite score `0.4*recency + 0.3*frequency + 0.2*importance + 0.1*causal` replaces single access_frequency threshold
- **Power-law memory decay** (v4.48): `S(t) = (t+1)^(-b)` for frequently accessed memories, hyperbolic fallback for cold-start
- **SM-2 spaced repetition** (v4.48): ease_factor (1.3-3.0) adjusts per-item difficulty, graduated failure (drop 1-2 boxes vs hard reset)

### Improved

- **Conflict MCP responses**: `nmem_conflicts list` and `check` now include `temporal_classification` field
- **CONTRADICTS synapse metadata**: includes temporal classification for audit trail
- **Subject matching**: now normalizes tech aliases before comparison (PostgreSQL = Postgres = pg)

### Tests

- 16 new conflict tests: negation (5), temporal (5), entity matching (4), auto-resolve supersession (2)
- 59 total conflict detection tests pass

## [4.48.0] — 2026-04-15

### Added

- **Explainable recall**: activation paths now surfaced to MCP via `include_paths` parameter — hop-by-hop gain scores visible in recall responses
- **FalkorDB removed**: ~3,900 LOC deleted — consolidates on SQLite (default) + PostgreSQL (opt-in) + InfinityDB (Pro)

### Improved

- **CI compatibility**: all GitHub Actions bumped to Node.js 24 compatible versions (checkout v5, setup-python v6, etc.)
- **Schema v39**: migration adds `ease_factor` column to review_schedules table

## [4.47.0] — 2026-04-15

### Added

- **Compact response mode**: `remember` and `recall` default to `compact=true`, returning only essential fields (saves 200-800 tokens per call). Set `compact=false` for full metadata
- **Empty brain early exit**: `recall` skips the full pipeline when brain has 0 neurons — eliminates 8+ async ops for new users
- **SimHash always-on dedup** (v4.46.1): zero-cost SimHash dedup runs on every `remember`, even when full dedup is disabled
- **Stem-based keyword matching** (v4.46.1): cross-inflection matching in context compiler ("optimizing" ↔ "optimization")

### Improved

- **`clean_for_prompt` default changed**: `true` (was `false`) — recall responses skip section headers and type tags by default
- **Activation threshold raised**: `0.2` → `0.3` across all backends and presets — reduces noise, fewer weak activations waste tokens
- **Composite score rebalanced** (v4.46.1): activation 25% + priority 25% + frequency 20% + conductivity 15% + freshness 15%
- **Freshness weight enabled** (v4.46.1): default `0.15` (was `0.0`) — stale memories naturally deprioritized
- **BALANCED preset activation threshold**: `0.2` → `0.3` to match new defaults

### Docs

- **Tool count corrected**: 55/56 → 59 across README, quickstart, npm-package
- **Neo4j → FalkorDB**: updated outdated storage references in installation.md and FAQ.md
- **Init contradiction resolved**: removed misleading `nmem init --full` from README (MCP auto-initializes)
- **Troubleshooting decision tree**: added to FAQ.md for common issues
- **"First 5 Minutes" onboarding**: added realistic MCP workflow to quickstart.md

### Tests

- 6717 unit tests pass, 0 failures

## [4.46.0] — 2026-04-14

### Added

- **Brain Store delete**: new `delete` action in `nmem_store` MCP tool — preview what would be deleted, then confirm to permanently remove a brain and all associated data
- **PostgreSQL storage switch (CLI)**: `nmem storage switch postgres` validates config and tests connection before switching
- **PostgreSQL storage switch (Dashboard)**: extended storage status, backend switch, and migration API endpoints to support postgres
- **Connection test endpoint**: `POST /api/dashboard/storage/test-connection` for testing PostgreSQL connectivity from the dashboard
- **PostgreSQL migration**: dashboard migration pipeline now supports `to_postgres` direction via export/import snapshot

### Fixed

- **Brain delete orphans**: `clear()` now deletes from all 12 brain-scoped tables (was missing `brain_versions`, `sync_states`, `alerts`, `change_log`, `devices`, `merkle_hashes`)
- **Brain delete atomicity**: `clear()` wrapped in a transaction — partial failures now roll back cleanly
- **Migration source selection**: migration task now reads current backend from config instead of inferring from direction (fixes postgres→sqlite path)
- **Credential leak prevention**: PostgreSQL connection errors in HTTP responses sanitized — no password exposure, details logged server-side only

## [4.45.2] — 2026-04-13

### Fixed

- **SQLite concurrency**: `busy_timeout` increased 5s→30s across all connection types (store, dialect, read pool) to handle multi-process contention
- **MCP tool retry**: automatic retry with exponential backoff (3 attempts) on "database is locked" errors during tool calls
- **Consolidation resilience**: gracefully skip fiber merges on FK violations instead of crashing the entire consolidation run
- **Brain import**: support `.brain` package format with auto-detection, robust handling of missing fields (`brain_id`, `exported_at`, `version`)
- **Consolidation scheduler**: lambda now correctly binds `MaintenanceConfig` instead of calling with no args
- **Dashboard store UI**: improved text contrast — `muted-foreground` was too dim in dark mode

### Removed

- **Companion setup guide**: removed unrelated Vibe Companion docs from Neural Memory documentation

## [4.45.1] — 2026-04-12

### Fixed

- **Brain Store import**: `validate_brain_package()` no longer requires `content_hash` in manifest — registry-hosted community brains were being rejected because hash is only computed at export time
- **MCP docs**: regenerated for `nmem_reflex` tool (CI docs freshness gate)

## [4.45.0] — 2026-04-12

### Added

- **Reflex Arc**: always-on neurons pinned via `_reflex` metadata flag, injected into every recall context before spreading activation. SimHash conflict detection (hamming ≤ 10) auto-supersedes old reflexes via SUPERSEDES synapse. MCP tool `nmem_reflex` (pin/unpin/list), max 20 per brain, `exclude_reflexes` param on `nmem_recall`
- **Thought Chains**: `include_paths` param on `nmem_recall` exposes existing activation paths — returns top-5 neurons with activation scores showing how each result was reached. Zero new modules, reuses `ActivationResult` data
- **Déjà Vu**: scar tissue detection extending `PredictionErrorStep`. Finds SimHash-similar neurons (hamming ≤ 12) participating in causal chains (`caused_by`/`leads_to`/`resolved_by` synapses). Warnings surfaced in `nmem_remember` response as `deja_vu_warnings`

### Tests

- 42 new tests: reflex arc (25), reflex MCP (12), déjà vu detection (7), déjà vu pipeline integration (2), thought chains (covered via existing recall tests)
- MCP tool count: 58 → 59 (`nmem_reflex`)

## [4.44.0] — 2026-04-11

### Added

- **Engine**: abstraction constraints — level-gated spreading activation prevents low-abstraction nodes from activating high-abstraction concepts
- **Engine**: context compiler — cross-fiber deduplication, merge, and query re-scoring for tighter recall windows
- **Engine**: hybrid retrieval fusion — tri-modal scoring combining graph spreading activation, semantic (vector), and lexical (BM25) signals
- **Engine**: depth gap features — ACL per-neuron access control, confidence scores, preference/temporal/role query filters, scheduler, goal hierarchy
- **Embedding**: hybrid vector retrieval for SQLite — HNSW sidecar index, `VectorSearchMixin`, `EmbeddingStep` pipeline integration
- **Benchmark**: LongMemEval benchmark suite — evaluation scripts for long-horizon memory recall quality
- **Sync Hub**: teams schema, queries, routes, and types for multi-user organization support

### Distribution

- **OpenClaw**: bumped plugin to 1.16.1
- **npm MCP**: added `publish-npm-mcp` CI job for automated npm wrapper publishing

## [4.43.0] — 2026-04-09

### Improved

- **Goal proximity × prediction error compound** — memories that are both surprising (high prediction error) and near active goals now receive an amplified boost (`1 + surprise * 0.3`), making unexpected goal-relevant information surface more strongly
- **Causal auto-inclusion dedup** — `gather_causal_context()` now excludes neuron IDs already present in matched fibers, preventing duplicate content when temporal binding and causal tracing surface the same neurons
- **Schema-cluster diversity in MMR** — when schema assimilation is enabled, the greedy MMR loop now caps fibers per schema cluster (same cap as lifecycle strata), preventing a single knowledge schema from dominating recall results
- **Interference goal tiebreaker** — `resolve_interference()` accepts `goal_neuron_ids`; goal-relevant neurons receive halved weight decay during retroactive interference resolution (0.975 vs 0.95), preserving goal-relevant memories under competition

### Fixed

- **4 integration debts resolved** — all cross-feature gaps from Section 9 of FEATURE_REGISTRY.md now marked FIXED (goal×prediction, causal dedup, schema→MMR, interference+goal)

## [4.42.0] — 2026-04-09

### Added

- **Causal auto-inclusion** — recalled memories with CAUSED_BY/LEADS_TO synapses automatically include their causal chain as supporting context. Traces both causes and effects (max 2 hops), deduplicates across fibers, and respects a 20% token budget cap
- **Anti-redundancy attention set** — session-scoped tracking of surfaced fiber IDs (FIFO@500 with O(1) lookup). Previously surfaced fibers receive a 0.3x multiplicative penalty on repeat queries, preventing the same memories from dominating every recall
- **Cascade staleness propagation** — when a fact is superseded via conflict resolution, downstream causal neurons and their fibers are automatically marked stale via BFS through LEADS_TO synapses (weight-gated, max 3 hops). Respects `cascade_staleness_enabled` config flag
- **Stratum-aware MMR diversity** — recall results now span multiple lifecycle stages (episodic/consolidating/semantic/archival) with a 40% cap per stratum, preventing any single memory layer from dominating results
- **Temporal neighborhood queries** — `query_temporal_neighborhood(fiber_id, window_hours)` finds chronologically adjacent memories within a configurable time window

### Improved

- **Session state eager creation** — `get_or_create()` used instead of `get()` for session state in retrieval pipeline, ensuring anti-redundancy works from the very first query
- **Familiarity path records surfaced fibers** — early-exit via familiarity fallback now properly records matched fibers in the attention set
- **Causal supplement dedup** — `format_causal_supplement()` deduplicates neurons across chains to prevent repeated content in output

### Tests

- 36 new tests: causal inclusion (8), anti-redundancy attention set (7), causal recall integration (8), cascade invalidation (3), stratum MMR config (5), temporal neighborhood (5)

## [4.41.0] — 2026-04-09

### Added

- **Goal-directed recall** — prefrontal cortex-style top-down attention modulation. Agents can declare active goals (`nmem_goal`), and memories topologically close to those goals get a BFS proximity boost during retrieval. Recall becomes relevance-based (proximity to goal) instead of just similarity-based
- **Session intent declaration** — `nmem_session(action="set", intent="...")` seeds topic EMA with strong 0.6 alpha boost, priming all subsequent recalls toward the declared intent. Pending intent propagates to new sessions automatically
- **New MCP tool: `nmem_goal`** — create, list, activate, pause, complete goals with priority 1-10. Goals are metadata-backed on existing INTENT neurons (zero schema migration)

### Improved

- **Goal proximity in all recall paths** — goal scoring applied consistently across main pipeline + familiarity fallback strategies A and B
- **BFS capped at 10 goals** to prevent unbounded graph traversal in large brains
- **EMA seed capped at 1.0** to prevent double-seeding from repeated intent declarations

### Tests

- 17 new tests for BFS proximity scoring (linear chains, branching, cycles, multi-goal), goal neuron helpers, intent seeding, and edge cases

## [4.40.0] — 2026-04-08

### Added

- **Grounded neurons** — canonical truth anchors inspired by hippocampal reference frames. Grounded neurons resist decay in lifecycle, auto-win conflicts in auto-resolve (Rule 0), and skip conflict resolution entirely. Metadata-backed (`_grounded`, `_confidence`) for zero-migration compatibility across all storage backends
- **Pin-to-ground wiring** — `nmem_pin` now automatically grounds anchor neurons when pinning (sets `grounded=True`, `confidence=1.0`) and ungrounds when unpinning

### Tests

- 10 new tests for grounded neuron model properties, conflict auto-resolve Rule 0, and pin grounding behavior

## [4.39.0] — 2026-04-08

### Added

- **Embedding enabled by default** — `embedding_enabled` now defaults to `True`, enabling semantic similarity search, hybrid recall, and semantic discovery out of the box. Graceful fallback if sentence-transformers is not installed

## [4.38.0] — 2026-04-08

### Added

- **Session cortical columns** — cortical column episodic binding (Mountcastle 1957): encoding now creates "column" fibers that aggregate all neurons from a session into a single searchable unit. Column fibers get 1.3x score boost in retrieval and enable a new familiarity Strategy C (summary keyword search) when both activation-based and keyword-based recall fail. Gated behind `session_columns_enabled` config flag (default: True)

### Tests

- 11 new tests for column fiber creation, config flag, score boost, summary keyword matching, and truncation

## [4.37.0] — 2026-04-08

### Added

- **Adaptive density scaling** — homeostatic synaptic scaling (Turrigiano 2008) for large graphs: density-aware stabilization noise floor, entropy-normalized sufficiency gate, and adaptive lateral inhibition K. All gated behind `graph_density_scaling_enabled` config flag (default: True)

### Tests

- 12 new tests for density-aware stabilization, entropy normalization, and adaptive K scaling

## [4.36.0] — 2026-04-08

### Added

- **Familiarity fallback recall** — dual-process theory (Yonelinas 1994): when recollection fails (sufficiency gate INSUFFICIENT), NM now tries familiarity-based recall with relaxed activation thresholds before returning empty. Two strategies: (A) halved activation threshold for weak signals, (B) broader keyword-based anchor search for no_anchors gate. Results tagged `synthesis_method="familiarity"` with confidence capped at 0.4
- **Codex CLI support** — `nmem setup rules --ide codex` generates `codex.md` with hook-like instructions for OpenAI Codex CLI, ensuring persistent memory usage across sessions (closes #127)

### Improved

- **Sufficiency gate tuning** — `unstable_noise` threshold tightened (0.3 → 0.2) and `ambiguous_spread` entropy base raised (3.0 → 4.0) to reduce false-INSUFFICIENT decisions that caused empty retrieval

### Tests

- 11 new tests: familiarity config defaults, gate threshold changes, integration tests for fallback behavior

## [4.35.0] — 2026-04-07

### Added

- **Tool tier recommendation** — `nmem_stats` now suggests switching to `minimal` (4 tools) or `standard` (9 tools) tier when usage data shows fewer tools are needed, saving 60-80% context tokens
- **OpenClaw plugin display name** — fixed plugin name from "NeuralMemory" to "Neural Memory" for proper ClawHub listing display

### Fixed

- **CI: mypy errors** — `text_index.py` return type, `retrieval.py` attr-defined for batch method
- **CI: embedding anchor test** — removed auto-created `knn_search` mock attribute that broke fallback path
- **CI: docs freshness** — regenerated `mcp-tools.md`

### Tests

- 9 new tool tier hint tests covering all recommendation paths

## [4.34.0] — 2026-04-07

### Added

- **Consolidation Phase 3-4 Quality Improvements** — 5 new config knobs for fine-tuning consolidation behavior:
  - **Semantic prune protection** — neurons in semantic-stage fibers use halved prune threshold (`prune_semantic_factor=0.5`), preserving mature knowledge
  - **Bridge weight floor** — sole-connection synapses protected above configurable floor (`bridge_weight_floor=0.01`), preventing graph fragmentation
  - **Expanded surface regen triggers** — regeneration now fires on `fibers_compressed`, `lifecycle_states_updated`, and configurable `surface_regen_prune_threshold`
  - **Maturation fast-track** — memories with 10+ rehearsals across 3+ time windows advance episodic→semantic in 1 day instead of 3 (`maturation_fast_track_rehearsals`, `maturation_fast_track_time_days`)
  - **Config validation** — `ConsolidationConfig.__post_init__` clamps degenerate values (negative factors, zero rehearsals)
- **Robust brain import** — enum fallbacks and per-entity error handling for `.brain` file imports

### Fixed

- **Version regex false positives** — version pattern no longer matches IPv4/IPv6 prefixes (e.g., `IPv4.0`)
- **Pro feature name mismatch** — unified feature names between pay-hub and sync-hub (canonical registry in `pay-hub/src/lib/features.ts`)
- **brain_ops.py logger scope** — `_log` moved to module level, fixing undefined name errors in import methods

### Tests

- 35 new tests (22 Phase 3, 13 Phase 4) covering semantic protection, bridge floors, surface triggers, version regex, fast-track maturation, config validation

## [4.33.0] — 2026-04-05

### Added

- **SimHash Pre-filter** — opt-in locality-sensitive hashing gate before spreading activation. Set `simhash_prefilter_threshold` (1-64) in BrainConfig to exclude distant neurons by Hamming distance, reducing candidate sets on large brains. Legacy neurons (hash=0) are never excluded. Empty/whitespace queries skip the filter entirely.
- **Time-Travel Queries** — `as_of` parameter on `nmem_recall` filters neurons and fibers by `created_at <= as_of`, reconstructing historical memory state. `reconstruct_stage()` utility walks back maturation stages using `stage_entered_at` timestamps.

### Fixed

- **Time-travel anchor filtering** — `created_before` now passed to all `find_neurons()` calls in anchor search (time, entity, keyword, fuzzy), not just fiber search
- **Storage backend compat** — all backends (memory, postgres, falkordb, sql, shared, pro) accept `created_before` parameter for signature compatibility

### Tests

- 21 new tests (13 SimHash, 8 time-travel), 6170 total passed

## [4.32.1] — 2026-04-05

### Fixed

- **Double-decay on inferred CO_OCCURS** — inferred synapses were hit by both `_inferred` (0.5×) AND CO_OCCURS (0.33×) decay, evaporating at 0.165× per consolidation. Now only one decay path applies.
- **Dedup ALIAS pruning** — dedup ALIAS synapses (`_dedup=True`, `reinforced_count=0`) were pruned by the new type-aware decay, breaking dedup chains. Now excluded from accelerated decay.
- **Truncation metadata leak** — `_content_truncated` metadata no longer leaks onto dedup alias neurons
- **Ephemeral quality warning** — quality warning for long content now skipped for ephemeral memories

## [4.32.0] — 2026-04-05

### Added

- **Phase F: Quality Deep Dive** — comprehensive recall quality improvements across 4 sub-features:

- **F1: Synapse Quality** — causal/semantic synapses now form reliably in mature brains:
  - Existing neurons tracked in `PipelineContext` so `RelationExtractionStep` can match spans against previously encoded entities/concepts
  - Word-overlap matching in `_match_span_to_neuron()` with Jaccard-like scoring (falls back from exact → substring → word overlap)
  - CO_OCCURS synapse initial weight reduced 0.5 → 0.3 to reduce noise in spread activation
  - Anchor content truncation at 500 chars with sentence-boundary detection
  - Quality warning on `nmem_remember` when content exceeds 500 chars

- **F2: Fiber Precision** — faster maturation for agent brains, aggressive noise pruning:
  - EPISODIC→SEMANTIC time gate relaxed: 7 days → 3 days, 3 distinct days → 2
  - Agent rehearsal path relaxed: 15 rehearsals → 5, 5 windows → 3
  - Type-aware pruning: CO_OCCURS/ALIAS synapses decay 3× faster when reinforced < 3 times
  - Associative inference weights reduced: initial 0.3→0.2, max 0.8→0.5

- **F4: Role-Based Spread Activation** — synapse roles now modulate activation traversal:
  - SEQUENTIAL synapses (CAUSED_BY, LEADS_TO) boost 1.3×
  - REINFORCEMENT synapses boost 1.2×
  - LATERAL synapses (CO_OCCURS, RELATED_TO) dampen 0.85×
  - PASSIVE synapses (ALIAS) skip entirely (0.0×)

### Improved

- **Health diagnostics** — tightened warning thresholds so grade D brains no longer say "healthy"
- **Brain registry** — Hub-first fetch, proper URL import, export download improvements

### Tests

- 24 new tests: 13 synapse quality, 3 relation encoding, 6 memory stages, 2 activation
- 4 existing tests updated for new thresholds
- All 93 Phase F tests passing

## [4.31.0] — 2026-04-05

### Added

- **Community Brain Publishing** — publish brains to the community store directly from Dashboard or MCP:
  - GitHub-based distribution: brain packages stored in `nhadaututtheky/brain-store` repo (unlimited free storage)
  - Hub thin proxy: creates GitHub PRs for publish, stores only ratings in D1 (minimal storage)
  - GitHub Actions: validate brain packages (format + security scan) and auto-merge safe PRs
  - Post-merge Action rebuilds `index.json` catalog automatically
  - `BrainRegistryClient` upgraded: Hub API primary (includes ratings), GitHub raw fallback
  - FastAPI `POST /api/dashboard/store/publish` endpoint with security scan gate
  - MCP `nmem_store action="publish"` for agent-driven publishing
  - ExportDialog: "Download .brain" + "Publish to Community" buttons with success confirmation
  - Full EN + VI translations for publish flow

### Fixed

- **Hub security scan bypass** — packages without `scan_summary` are now rejected (previously passed through)
- **Hub SSRF prevention** — `ghApi` no longer accepts arbitrary URLs, always prepends GitHub API base
- **Hub orphan branch cleanup** — failed PR creation now deletes the orphan branch
- **Hub offset/limit NaN** — non-numeric query params now fall back to defaults
- **MCP export path restriction** — `output_path` now restricted to home directory or CWD
- **Ratings GET validation** — `GET /ratings/:name` now validates brain name format
- **Tool tier count** — updated 55→56 after adding `nmem_store`

## [4.30.1] — 2026-04-05

### Fixed

- **SSRF prevention in brain registry** — HTTPS-only scheme validation, private/loopback IP blocking, body-size limit on actual bytes (not Content-Length header)
- **Import brain UUID generation** — imported brains now get a proper UUID instead of empty string, which caused `import_brain()` to fail silently
- **Brain Store page layout** — fixed content overlapping sidebar (missing padding)
- **BrainPreviewDialog null safety** — optional chaining on `scan_result.findings` and `content_hash`
- **Error message sanitization** — removed brain names from 404 responses, added path validation on registry preview

### Added

- **Upload .brain file button** — direct file import from Store page header with toast feedback
- **Export brain dialog** — export active brain from Overview page with metadata form and auto-download

## [4.30.0] — 2026-04-05

### Added

- **Brain Store (Marketplace)** — community brain marketplace for sharing and importing curated knowledge brains:
  - `.brain` package format v1.0 — single JSON file with manifest + BrainSnapshot, human-inspectable
  - Three-gate security model — export, import, and registry gates all use `brain_scanner.py` with Unicode bypass protection (NFKC normalization + zero-width char stripping)
  - Registry client — GitHub-based distribution via `raw.githubusercontent.com`, 5-min TTL in-memory cache with stale fallback
  - MCP tool `nmem_store` — browse, preview, import, and export community brains (action-based schema)
  - FastAPI endpoints — `/api/dashboard/store/registry`, `/preview`, `/import`, `/export`, `/rate`, `/import-remote`
  - Dashboard Store page — browse grid with search, category filters, sort options, preview dialog with security scan display, one-click import, 5-star rating
  - Export dialog — export active brain from Overview page with metadata form, auto-download as `.brain` file
  - Size tiers — Micro (<100KB), Small (<1MB), Medium (<10MB), Reject (>10MB)
  - Rating system — in-memory bounded (1000 packages × 100 ratings), average + count tracking
  - i18n — full English and Vietnamese translations (35+ keys)
- **`tag_mode` parameter for recall** — filter by tags using AND (all must match, default) or OR (any tag matches). Available in REST API, MCP `nmem_recall`, and cross-brain recall. Backward-compatible: default behavior unchanged.

### Fixed

- **FalkorDB `find_fibers` missing tag filter** — `find_fibers()` accepted `tags` parameter but never filtered by it (only `find_fibers_batch` did). Now both methods filter correctly.

## [4.29.0] — 2026-04-04

### Added

- **A7 Recall Intelligence**: Causal synapse roles (CAUSED_BY, RESOLVED_BY, EVIDENCE_FOR/AGAINST, etc.) with spreading activation integration. Fixes auto-supersede errors — storing a fix creates RESOLVED_BY synapse and demotes error activation by >=50%. Outcome learning tracks prediction accuracy.
- **A8 Agent Intelligence — Phase 1 (Precision Recall)**: Context optimizer sharpens recall by injecting session topic context, deduplicating fiber results, and capping token budgets. Retrieval penalizes stale version references (-20%).
- **A8 Phase 2 (Proactive Context)**: Surface-based topic injection at session start — up to 9 topic memories from 3 clusters. Habit-aware context with time-of-day and day-of-week patterns.
- **A8 Phase 3 (Auto-Save Intelligence)**: Quality scoring (0-10) for every nmem_remember — specificity, structure, brevity bonuses with wall-of-text penalty. Auto-classification confidence scoring. Enhanced importance scoring with security keywords, CVE detection, error traces.
- **A8 Phase 4 (Aggressive Consolidation)**: SimHash semantic merge (content-similar fibers merged via Union-Find). Stale version detection (>=2 major behind flagged). Access-based demotion (30d cold, 90d prune candidate, pinned exempt). Summary fibers for 5+ merged groups (1.1x retrieval bonus). Surface regeneration after structural changes.
- **Code-semantic encoding**: 7 new synapse types (IMPORTS, CALLS, DEPENDS_ON, INHERITS, IMPLEMENTS, DEFINED_IN, RAISES). Compound identifier keyword extraction (PascalCase, camelCase, snake_case splitting).

### Improved

- **Agent instructions rewritten**: SYSTEM_PROMPT, COMPACT_PROMPT, MCP_INSTRUCTIONS, and SKILL.md restructured for "read less, understand more" — scannable sections, A7/A8 smart behaviors documented, tool decision matrix.
- **Tighter regex patterns**: `_ERROR_TRACE_RE` requires `Error:` or `Traceback (most recent` pattern. `_VERSION_PATTERN` requires `v` prefix to avoid IP/date false positives. `_FILE_PATH_PATTERN` excludes abbreviations.

### Tests

- ~100 new tests: causal recall (22), precision recall (15), proactive context (12), auto-save intelligence (17), aggressive consolidation (23), code-semantic encoding (15+)

## [4.28.0] — 2026-04-03

### Fixed

- **Pro deps now optional** (#125): numpy, hnswlib, msgpack moved to `[pro]` extra. Free users on Python 3.14+ were blocked by hnswlib C++ build failure. Now `pip install neural-memory` works everywhere; `pip install neural-memory[pro]` adds Pro deps.
- **CLI activate auto-installs**: after activating a Pro key, CLI detects missing deps and offers to install them automatically.
- **Dashboard Pro deps banner**: Settings page shows yellow warning with install command when Pro is activated but deps are missing.
- **ClawHub display name**: fixed ".Claude Plugin" → "Neural Memory" via `--name` flag on publish.
- **CI test fix**: mock `detect_project_root` in surface path test to prevent `.neuralmemory/` dir interference on CI runner.

### Added

- **InfinityDB mixin refactor**: split InfinityDBStorage into 3 mixins (typed, sync, extras) for maintainability, with dedicated test files.
- **Pay-hub license grant**: new `/admin/license/grant` endpoint for direct D1 key insertion; `/admin/license/sync` now auto-inserts into D1.
- **GitHub stars + download badges** on README.
- **Distribution**: published to MCP Registry, submitted to awesome-mcp-servers lists.

## [4.27.1] — 2026-04-02

### Changed

- **Pro deps bundled**: numpy, hnswlib, msgpack moved from optional `[pro]` extra to main dependencies. One install, zero friction: `pip install neural-memory` → activate key → done.

## [4.27.0] — 2026-04-02

### Changed

- **Pro merge**: Pro features (InfinityDB, cone queries, directional compression, smart merge) bundled in main package. No separate `neural-memory-pro` needed.
  - Activate with key: `nmem pro activate <KEY>`
  - Plugin system preserved for third-party extensions

### Added

- 301 Pro tests migrated to `tests/unit/pro/` (auto-skip when Pro deps missing)

## [4.26.0] — 2026-04-02

### Added

- **Tier analytics** (B5 Phase 4): MCP `nmem_tier action="analytics"` returns memory type x tier breakdown, 7d/30d velocity metrics (promoted/demoted/archived), and recent tier changes (capped at 50 events).
  - REST API: `GET /api/dashboard/tier-analytics` (breakdown + velocity), `GET /api/dashboard/tier-history?limit=20&offset=0` (paginated events)
  - Dashboard: Tier Analytics page with velocity KPI cards, grouped bar chart (recharts), and recent changes table
  - `_classify_change()` helper classifies tier transitions as promoted/demoted/archived

### Improved

- Brain Quality C4 (Agent Visualization) marked complete — `nmem_visualize` tool fully shipped with Vega-Lite, markdown table, and ASCII chart formats
- B5 Smart Tiers track fully complete (4/4 phases: Auto-Tier, Decision Intelligence, Domain Boundaries, Tier Analytics)

### Tests

- 11 new tier analytics tests (classify change, breakdown by type, velocity windows, recent changes)

## [4.25.0] — 2026-04-01

### Added

- **Domain boundaries** (B5 Phase 3): Scope boundary memories to specific domains (e.g. `financial`, `security`, `code-review`). Boundaries without a domain remain global. Uses existing tag system with `domain:` prefix convention — zero schema migration.
  - `nmem_remember(content="...", type="boundary", domain="financial")` → auto-adds `domain:financial` tag
  - `nmem_recall(domain="financial")` → HOT context injection filters boundaries by domain, keeps global (unscoped) boundaries
  - `nmem_boundaries` tool — list boundaries grouped by domain, list unique domains with counts
- **Brain milestone tool** (`nmem_milestone`): Track neuron-count achievements and generate growth reports

### Improved

- MCP tool count: 53 → 55 (added `nmem_boundaries`, `nmem_milestone`)

### Tests

- 15 new domain boundary tests (tag creation, filtered context, boundaries tool, remember injection)
- Updated tool count assertions across test suite

## [4.24.0] — 2026-03-31

### Added

- **Auto-tier engine** (B5 Phase 1, Pro): Automatic WARM→HOT promotion, HOT→WARM demotion, WARM→COLD archival based on access patterns. Protection for BOUNDARY types and pinned fibers. Oscillation prevention. `nmem_tier` MCP tool with status/evaluate/apply/history/config actions.
- **Decision intelligence** (B5 Phase 2): Extract structured decision components (chosen, alternatives, reasoning, confidence) from DECISION-type memories. Detect overlapping prior decisions, classify relationships (confirms/contradicts/evolves), create EVOLVES_FROM synapses, boost recall scores for domain-relevant decisions.
- **Dashboard Phosphor icons**: Migrated all 19 component files from Lucide to Phosphor Icons (`@phosphor-icons/react`). Added Playwright E2E smoke tests (8 tests).

### Fixed

- **CLI ignores `NMEM_BRAIN` env var** (#123): CLI `get_storage()` now respects `NMEM_BRAIN`/`NEURALMEMORY_BRAIN` env vars. Priority: explicit arg > env var > config file. `brain list` shows effective brain from env var.
- **Handler monolith split**: Split `tool_handlers.py` (2030 LOC) into 7 domain-specific handler modules. Fixed circular imports, removed duplicate utility functions.
- **Input firewall hardening**: Added bounds validation, type checks, and range clamping across handler modules (lifecycle, provenance, evolution, stats).

### Improved

- Auto-tier config: `cold_archive_days` invariant (must be ≥ `demote_inactive_days`), Pro gate in consolidation engine
- MemoryTier constants used consistently (no string literals in handlers)
- MCP tool count: 52 → 53

### Tests

- 50+ new tests across tier engine, decision intelligence, brain isolation, E2E smoke tests
- Fixed stale `ReflexPipeline` patch targets and MagicMock config attrs in test fixtures

## [4.23.4] — 2026-03-30

### Fixed

- **macOS SSL cert failures** (#120): Added `ssl_helper.py` with certifi-based SSL context, patched all 11 aiohttp session locations
- **`nmem init --full` hang** (#121): Added `--skip-embeddings` flag and non-interactive terminal guard to prevent hang in pipes/CI
- **`find_spec` crash** (#122): Handle `ImportError` from namespace packages (e.g. `google-cloud-storage`) in `_is_module_available`

### Tests

- 11 new tests: `_is_module_available` edge cases (6), SSL helper (4), skip-embeddings (1)

## [4.23.3] — 2026-03-30

### Improved

- **Landing page**: Added "Install Free" CTA, quickstart guide for new users, post-purchase activation steps
- **ClawHub**: Fixed display name from ".Claude Plugin" to "Neural Memory", published v4.23.3
- **CLI docs**: Regenerated CLI reference for storage commands

## [4.23.2] — 2026-03-30

### Added

- **Pro upgrade URL**: Free users see `upgrade_url` in MCP stats, CLI status, and dashboard license API — agents and UI can guide users to purchase page
- **CLI license info**: `nmem shared status` now shows license tier and upgrade link for free users

## [4.23.1] — 2026-03-30

### Fixed

- **Dashboard live reload**: License, storage status, and backend switch endpoints now reload config from disk — CLI changes (Pro activation, backend switch) reflected without server restart
- **Dashboard activation**: `/license/activate` now uses pay-hub directly (no sync config required), matches MCP + CLI behavior
- **Dashboard activation**: Adds `next_step` InfinityDB guidance hint when activating on SQLite

## [4.23.0] — 2026-03-30

### Added

- **Storage visibility**: `nmem_stats` now shows `storage_backend`, `pro_installed`, `is_pro` fields
- **Storage CLI**: `nmem storage status` — shows backend, Pro status, data file existence + sizes
- **Storage CLI**: `nmem storage switch <sqlite|infinitydb>` — switch with Pro/data guards
- **Migration**: `nmem migrate infinitydb` — SQLite → InfinityDB via export/import (Pro required)
- **Activation guidance**: Pro activation (MCP + CLI) now shows next_step hint to InfinityDB
- **Stats hint**: When Pro active but on SQLite, suggests InfinityDB upgrade path

## [4.22.2] — 2026-03-30

### Fixed

- **Pro activation**: Decoupled license activation from sync config — no longer requires hub_url + api_key
- **Pro activation**: Both MCP tool and CLI now call pay-hub directly with just the license key
- **Config**: ISO datetime sanitizer now accepts space-separated timestamps (pay-hub format)
- **Pro activation**: `activated_at` now populated with actual activation timestamp

## [4.22.1] — 2026-03-30

### Fixed

- **L4**: `with_priority()` now preserves `trust_score` and `source` fields (pre-existing bug)
- **L2**: HOT tier injection catches all exceptions, not just TypeError/AttributeError
- **M1**: Boundary auto-promote in `_edit` moved before tier assignment — eliminates dead code path
- **M2**: Tier distribution counts use `count_typed_memories()` SQL COUNT — no 1000-row display cap
- **L1**: Schema v38 migration promotes pre-v37 BOUNDARY memories from default "warm" to "hot"
- **L3**: Tier param normalized to lowercase in `nmem_remember`, `nmem_recall`, and `nmem_edit`

## [4.22.0] — 2026-03-29

### Added

- **A6 Tiered Memory Loading** — HOT/WARM/COLD tier system for context priority and decay behavior
  - **HOT**: Always injected into context, 0.5× decay rate, activation floor at 0.5 — memories never fade below half strength
  - **WARM**: Default tier, standard semantic-match retrieval, normal decay
  - **COLD**: Excluded from auto-context, 2× decay rate — archive-grade memories accessible only via explicit recall
  - **BOUNDARY safety invariant**: `MemoryType.BOUNDARY` memories always auto-promote to HOT tier (enforced in create, edit, pin, and decay)
- **Tier parameter** on `nmem_remember`, `nmem_edit`, `nmem_pin`, and `nmem_train` tools
- **Tier filter** on `nmem_recall` — filter recall results by specific tier
- **Schema v37** — `tier` column on `typed_memories` table with index
- **Dashboard**: Storage page with TierDistribution card (progress bars: red HOT, amber WARM, blue COLD)
- **`MAX_HOT_CONTEXT_MEMORIES`** constant (50) caps auto-injected HOT memories per recall

### Improved

- **Context optimizer** — HOT tier gets +0.3 score boost, COLD excluded by default (`exclude_cold=True`)
- **Lifecycle decay** — tier-aware decay with per-tier multipliers and floors, batched fiber lookups
- **Recall handler** — combined trust + tier filtering into single loop, HOT memories always injected regardless of `fresh_only`

### Tests

- 42 new unit tests across 4 phase files (`test_tiered_memory_phase1-4.py`)
- Covers: schema migration, tier constants, decay math, context optimizer, recall filter, lifecycle integration, dashboard API, tier stats

## [4.21.1] — 2026-03-28

### Fixed

- **Multilingual neuro engine** — arousal detection + prediction error reversal now support Vietnamese via pattern registries, with language-agnostic fallback for all other languages (closes #116, #119)
- **Auto-ingest noise stripping** — input firewall strips NM context headers, neuron-type bullets, and metadata wrappers before re-encoding, preventing self-referential memory pollution (closes #118)
- **OpenClaw hook migration** — migrated all legacy hooks to current API (`before_prompt_build`, `before_compaction`, `before_reset`, `gateway_start`)
- **Gemini SDK import** — updated `google.generativeai` → `google.genai` + default model to `gemini-2.0-flash` (#117)

### Added

- **`clean_for_prompt` recall mode** — new parameter on `nmem_recall` strips section headers and type tags from output, reducing noise when injecting context into prompts
- **Shared `detect_language()`** — deduplicated language detection from arousal + prediction_error into `extraction/parser.py`

### Improved

- **OpenClaw plugin v1.16.0** — auto-context recall uses `clean_for_prompt`, `sanitizeAutoCapture()` strips NM noise + short acknowledgements before re-ingest

## [4.21.0] — 2026-03-26

### Added

- **Neuroscience Engine** — 10 brain-inspired improvements across 4 phases:
  - **Phase 1**: Temporal binding (TEMPORAL synapses between nearby memories) + arousal detection (emotional valence scoring via sentiment/punctuation/caps)
  - **Phase 2**: Prediction error encoding (novelty-based priority boost via SimHash) + retrieval reconsolidation (context drift detection, context anchors for shifted memories)
  - **Phase 3**: Context-dependent retrieval (encoding fingerprint stored per fiber, Jaccard similarity scoring at recall) + hippocampal replay (LTP/LTD synapse strengthening during consolidation) + cognitive chunking (greedy clustering of retrieval results by activation + synapse connectivity)
  - **Phase 4**: Schema assimilation (auto-creates SCHEMA neurons when tag clusters exceed threshold, Piaget assimilate/accommodate) + interference forgetting (SimHash-based retroactive/proactive/fan-effect detection, CONTRADICTS synapses)
- **Post-encode hooks** — schema assimilation + interference detection auto-run after every `encode()` when enabled (non-critical, swallowed on error)
- **Real activation scores** — chunking now uses per-neuron activation levels from retrieval instead of dummy values
- **Paginated tag fetch** — `_find_neurons_by_tags()` helper pages through large brains (1000/page) instead of fixed limit
- 10 new `BrainConfig` fields (all default OFF except `context_retrieval_enabled` and `chunking_enabled`)
- 107 new unit tests across all neuro engine modules

### Fixed

- **`list(int)` bug** in recall_handler chunking — `result.neurons_activated` is int, not iterable
- **`replay_enabled` gate** — `hippocampal_replay()` now checks flag directly (was only checked at dispatcher level)
- **Small brain skip** — post-encode schema hook checks `get_stats()` neuron count before querying

## [4.20.4] — 2026-03-25

### Fixed

- **`_essence_backfill` pagination bug** — used broken cursor-based pagination with `offset=` param that `get_fibers()` doesn't support. Replaced with single-batch fetch (limit=1000) + safety cap of 2000 fibers
- **`_summarize` O(N²) pair explosion** — no cap on candidate pairs or fiber count. Added: cap fibers at 1000 (highest-salience), skip tags shared by >100 fibers, cap pairs at 50K, yield every 1000 pairs
- **Unbounded `get_synapses()` in dream engine** — filtered by `RELATED_TO` type (the only type dream creates), reducing memory footprint
- **`_prune` event loop blocking** — added `asyncio.sleep(0)` yield every 500 synapses in prune loop
- **Dormant neuron selection bias** — `_dream_cycle` always picked first 20 dormant neurons instead of randomizing

### Improved

- **Yield frequency** — cross-cluster enrichment 50→20 iterations, SimHash dedup 100→50 iterations for more responsive timeout cancellation
- **Encryptor cache TTL** — `_get_encryptor()` in retrieval engine now has 5-minute TTL instead of caching forever (picks up config changes mid-session)

## [4.20.3] — 2026-03-25

### Fixed

- **Consolidation CPU hang** — consolidation could run 1+ hours at 100% CPU on large brains. CPU-bound O(N²) loops in `_dedup`, `_merge`, and cross-cluster enrichment blocked the event loop, preventing `asyncio.wait_for` timeout from ever firing
  - Added `asyncio.sleep(0)` yields in all O(N²) loops so timeouts actually work
  - Capped dedup anchors at 2000, merge candidate pairs at 50K
  - Skip overly-shared neurons (>100 fibers) in merge candidate generation
  - Cross-cluster Jaccard loop now yields every 50 iterations
- **Dashboard search overlay** — command palette (Ctrl+K) had z-index stacking issue causing partial dark overlay. Fixed by rendering via `createPortal` to `document.body` with `z-[100]`

## [4.20.2] — 2026-03-25

### Fixed

- **Consolidation timeout** — full consolidation could run for hours on brains with 5K+ neurons/20K+ synapses. Root causes: dream engine O(N²) pair generation from unbounded activated neurons, enrichment O(N²) Jaccard fiber comparison on up to 10K fibers, and no timeout on any strategy
  - Added per-strategy timeout (120s) and total timeout (600s) via `asyncio.wait_for()`
  - Capped dream activated neurons at 500, reduced max pairs from 50K to 5K, max new dream synapses capped at 200
  - Capped enrichment fiber clustering at 1000 highest-salience fibers (was unbounded up to 10K)

### Improved

- **Consolidation progress logging** — each strategy now logs start/finish with duration, making it easy to identify bottlenecks
- **Timeout reporting** — timed-out strategies are listed in the consolidation report (`report.extra["timed_out_strategies"]`)

## [4.20.1] — 2026-03-25

### Fixed

- **Consolidate prune crash** (#113) — `consolidate --strategy prune` crashed with `TypeError: can't subtract offset-naive and offset-aware datetimes`. Added `ensure_naive_utc()` helper to normalize timezone-aware reference times in `synapse.time_decay()`, `consolidation.run()`, and `compression.run()`
- **CLI packaging regression** (#114, #115) — v4.20.0 wheel published to PyPI was missing `cli/commands/` directory, breaking all `nmem` CLI commands. Rebuilt wheel includes all command modules. Added import smoke tests to prevent regression

### Tests

- 7 new tests: timezone-aware decay (2), `ensure_naive_utc` helper (3), package integrity smoke (2)

## [4.20.0] — 2026-03-23

### Added

- **Ctrl+K Command Palette** — dashboard-wide search: navigate pages, search fibers by summary, search neurons by content (debounced). Pro upsell hints for semantic search and cross-brain search
- **Mindmap fiber names** — fiber list and root node now show human-readable summaries instead of UUIDs

### Fixed

- **activate.ts security** — license key moved from query param to POST body, upstream error messages no longer forwarded, strict regex validation (`nm_pro_*`/`nm_team_*`), tier whitelist validation before D1 write, removed dead code
- **Stale `type: ignore`** in `file_watcher.py` — removed unused mypy suppression

## [4.19.0] — 2026-03-22

### Added

- **Fidelity layers** — memories decay through 4 levels (FULL → SUMMARY → ESSENCE → GHOST) based on activation, importance, and time. Budget pressure shifts thresholds upward, automatically compressing aged memories to save tokens
- **Extractive essence engine** — sentence-level scoring using entity density and position bias. No LLM required, generates single-sentence distillations (max 150 chars)
- **LLM essence generator** — optional abstractive essence via configured provider with cost guard (skips LLM for priority < 3). Factory pattern with `extractive` (default) and `llm` strategies
- **Ghost recall** — faded memories render as `[~] tags | age | links | recall:fiber:{id}`. Users can restore full content via the recall key
- **Ghost visibility boost** — fibers shown as ghosts within 24h get +0.1 fidelity score, preventing repeated ghost cycling
- **Budget-aware context assembly** — `optimize_context()` now scores each fiber's fidelity, renders at appropriate level, and tracks fidelity stats (full/summary/essence/ghost counts)
- **`include_ghosts` parameter** on `nmem_context` — controls ghost section visibility in context output
- **Schema v33→35** — `essence` column on fibers (v34), `last_ghost_shown_at` column (v35)
- **BrainConfig fidelity fields** — `fidelity_enabled`, `fidelity_full_threshold`, `fidelity_summary_threshold`, `fidelity_essence_threshold`, `decay_floor`, `essence_generator`
- **Consolidation essence backfill** — cursor-based pagination for existing fibers without essence

### Fixed

- **13 pre-existing mypy errors** — Anthropic SDK union type narrowing in `llm_judge.py`, tags `set`/`list` type mismatch in recall handler
- **Doctor check count** — updated assertions for `_check_config_freshness` addition (11→12 checks)

### Tests

- 73 new fidelity tests across 4 phases (essence extraction, fidelity scoring, ghost rendering, generators)

## [4.18.1] — 2026-03-21

### Added

- **`nmem lifecycle` CLI** — manage memory lifecycle states from CLI: `status` (distribution), `freeze` (prevent compression), `thaw` (resume lifecycle), `recover` (rehydrate compressed memory). Mirrors MCP `nmem_lifecycle` tool (#97)
- **Config freshness check** — `nmem doctor` now detects missing config sections from newer versions. `nmem doctor --fix` auto-adds them with defaults (#97)

### Fixed

- **Write gate scope clarification** — changelog now documents that write gate applies to MCP pipeline only; CLI `nmem remember` bypasses it (explicit user intent). Disabled by default (#97)

## [4.18.0] — 2026-03-21

### Added

- **Write gate** — hard quality filter before storage with configurable thresholds (`min_length`, `min_quality_score`, `reject_generic_filler`, `max_content_length`). Applies to MCP pipeline only (auto-capture + `nmem_remember` tool). CLI `nmem remember` bypasses write gate (explicit user intent). Disabled by default (`enabled = false`) — opt-in via `config.toml` `[write_gate]` section
- **Agent identity capture** — MCP `clientInfo.name` auto-injected as `agent:` tag on every memory, enabling per-agent filtering in multi-agent setups
- **Consolidation lock** — atomic file-based lock (`O_CREAT|O_EXCL`) with per-brain isolation and cross-platform PID check (Windows + Unix), prevents concurrent consolidation corruption
- **Sync dedup** — content hash check on neuron import (skip duplicates), fiber anchor match with tag union merge on sync
- **Dead neuron pruning** — auto-prune neurons with `access_frequency=0` older than configurable `prune_dead_neuron_days` (default 14) during consolidation

### Improved

- **Dedup tuning** — simhash threshold 10→7 and max_candidates 10→30 for tighter duplicate detection
- **Recall quality** — configurable recency sigmoid halflife (`recency_halflife_hours`, default 168h), tag-aware scoring with additive boost for matching tags
- **BrainConfig** — 3 new fields: `recency_halflife_hours`, `tag_match_boost`, `prune_dead_neuron_days`
- **Session-end consolidation** — now includes DEDUP strategy alongside MATURE/INFER/ENRICH

### Fixed

- **CRITICAL: TOCTOU race** in consolidation lock — replaced check-then-write with atomic file creation
- **HIGH: Windows PID check** — `os.kill(pid, 0)` doesn't work on Windows; now uses `kernel32.OpenProcess`
- **HIGH: `_auto_capture` bypass** — parameter now popped from args so users cannot override auto-capture quality threshold
- **HIGH: Sync dedup abstraction** — replaced raw `_read_pool` SQL with `has_neuron_by_content_hash()` storage method

### Tests

- 95 new tests across 6 files (test_write_gate, test_dedup_improvements, test_recall_quality, test_multi_agent, test_sync_safety, test_dedup_config updates)
- Total: 4480 passed

## [4.17.0] — 2026-03-21

### Fixed

- **Per-project Knowledge Surface** — `save_surface_text()` always wrote to global `~/.neuralmemory/surfaces/` because `get_surface_path()` only returned project path when the file already existed (chicken-and-egg bug). Now uses `for_write=True` to prefer project-level `<root>/.neuralmemory/surface.nm` when a project root is detected, regardless of whether the file exists yet
- **Stale global surface warning** — logs `INFO` when both project and global surface files coexist, alerting users to the stale global copy

### Tests

- 2 new resolver tests: write-mode project path creation, read-mode global fallback

## [4.16.0] — 2026-03-21

### Improved

- **Agent instruction prompts** — audited and optimized all 10 instruction surfaces
  - Deduplicated SYSTEM_PROMPT cognitive sections (merged 2 → 1, ~50 lines saved)
  - Strengthened OpenClaw `buildToolInstructions()` from 5-line stub to full RECALL/SAVE/EPHEMERAL/COMPACT guide
  - Removed marketing copy from SKILL.md — agents see usage instructions, not feature lists
  - Fixed stale `fresh_only=true` param in `.cursorrules` and `CLAUDE.md` template
  - Added `ephemeral=true` docs to all surfaces (MCP_INSTRUCTIONS, SKILL.md, .cursorrules, CLAUDE.md, plugin.json)
  - Added `compact=true` + `token_budget` mention to all surfaces
  - Added `tags=[...]` to all SYSTEM_PROMPT examples for consistency
  - Removed hardcoded tool count from SKILL.md

## [4.15.0] — 2026-03-21

### Added

- **Ephemeral memories** (#91) — session-scoped scratch notes that auto-expire
  - `nmem_remember(content="temp", ephemeral=true)` — stores with 24h TTL, excluded from consolidation and cloud sync
  - `nmem_recall(query="temp", permanent_only=true)` — filter out ephemeral memories from results
  - `nmem_remember_batch` supports `ephemeral` per item
  - Auto-cleanup of expired ephemeral neurons at session end (`nmem_auto(action="process")`)
  - Schema migration v32→v33: `ephemeral` column + index on neurons table

### Tests

- 14 new tests for ephemeral memories (`test_ephemeral.py`)

## [4.14.0] — 2026-03-21

### Fixed

- **Vietnamese auto-capture quality** (#94) — dramatically reduce low-quality Vietnamese memories
  - Quality gate: reject captures where >60% of words are Vietnamese stop words
  - TODO patterns: require compound forms (`cần phải`, `nhớ là`) — bare `cần`/`phải`/`nên` no longer match
  - Preference patterns: require explicit subject (`tôi`/`mình`/`em`/`anh`) + minimum content length
  - Correction patterns: require minimum 10-char capture content
  - Confidence penalty increased (0.7 → 0.55) for all Vietnamese regex captures
  - Minimum capture length raised (15 → 25 chars) for Vietnamese patterns
  - One-time pyvi missing warning when Vietnamese text detected in auto-capture

### Tests

- 22 new tests for Vietnamese capture quality (`test_vietnamese_capture.py`)
- Updated existing Vietnamese preference test for tighter pattern

## [4.13.0] — 2026-03-20

### Added

- **Memory Lifecycle Engine** — Heat-based compression resistance inspired by TEMM1E
  - Heat score: weighted combination of access recency, frequency, priority (exponential decay)
  - Lifecycle states: ACTIVE → WARM → COOL → COMPRESSED → ARCHIVED
  - Hot memories resist compression by 1 tier; frozen memories never compress
  - Neuron snapshots: recoverable content even after destructive Tier 3-4 compression
  - Access tracking: batch update `last_accessed_at` on every recall
  - `nmem_lifecycle` tool: status/recover/freeze/thaw actions
  - Schema v32: `lifecycle_state`, `frozen`, `last_accessed_at` columns + `neuron_snapshots` table
- **Adaptive Instructions** — Self-improving procedural memory
  - Auto-populate instruction metadata: version, execution_count, success_rate, trigger_patterns
  - `nmem_refine` tool: version instructions with refinement history, add failure modes/triggers
  - `nmem_report_outcome` tool: track execution success/failure, recompute success_rate
  - Recall boost: proven instructions (high success_rate) rank higher via activation bonus
  - Trigger pattern matching: instruction keywords boost relevance when query overlaps
- **Budget-Aware Retrieval** — Token cost management for context-efficient recall
  - Token cost estimator: estimate fiber tokens from content length
  - Greedy value-per-token allocation within context budget
  - `nmem_budget` tool: estimate/analyze/optimize token usage
  - `recall_token_budget` param on `nmem_recall` for opt-in budget-aware formatting
- 4 new MCP tools (47→50): `nmem_lifecycle`, `nmem_refine`, `nmem_report_outcome`, `nmem_budget`
- 133 new tests across 3 test files

## [Unreleased]

### Added

- **Input firewall (Gate 1)** — Security gate blocking garbage/adversarial content from auto-capture pipeline
  - Blocks: oversized content (>10KB), control sequences (`<ctrl*>`, fake role tags), JSON metadata injection, base64/binary blocks, repetitive content, low-entropy data
  - `FirewallResult` dataclass with `blocked`, `reason`, `sanitized` fields
  - Integrated into all 3 auto-capture entry points: stop hook, precompact hook, post-tool passive capture
  - 30 new tests (`test_input_firewall.py`)
- **Stop hook role filtering** — JSONL transcript entries classified by role; tool results skipped, assistant messages filtered by memory markers
- **Embedding semantic dedup** — Removes near-duplicate auto-captures using local embedding cosine similarity (sentence_transformer/ollama only)
- **Compact response mode** — Reduce MCP tool response tokens by 60-80%
  - `compact=true` param on all 46 MCP tools to strip metadata hints and truncate lists
  - `token_budget=N` param for progressive response size enforcement
  - Auto-compact: responses with >20 list items are compacted automatically
  - Content preview: list items show truncated content with `_content_truncated` flag
  - Count-replace: `fibers_matched`, `conflicts`, `expiry_warnings` → count only
  - Long string truncation: `markdown` field capped at 500 chars
  - `ResponseConfig` in config.toml: `compact_mode`, `max_list_items`, `strip_hints`, `content_preview_length`, `auto_compact_threshold`
  - 47 new tests (`test_response_compactor.py`)

### Fixed

- **Memory poisoning prevention** — Garbage content (chat control sequences, fake role injection, 270KB payloads) no longer enters brain through hooks (#94)
- **PreCompact emergency threshold** — Raised from 0.5 to 0.65 to reduce false positive captures
- **fiber.metadata type sync** — `nmem_edit` now syncs type changes into `fiber.metadata` (cherry-picked from PR #85)
- **Compression size guard** — Skip compression when summary is not smaller than original (#92)

## [4.11.0] - 2026-03-17

### Added

- **Diminishing returns gate (v4.0 Phase 5)** — Stop spreading activation early when new hops add insufficient signal
  - `ActivationTrace` dataclass: per-hop tracking of new neurons and activation gain
  - `should_stop_spreading()`: absolute (< min neurons) + relative (gain ratio < threshold) criteria
  - Wired into all 3 activation engines: BFS, PPR, Reflex
  - 4 new `BrainConfig` fields: `diminishing_returns_enabled/threshold/min_neurons/grace_hops`
  - 25 new tests (`test_diminishing_returns.py`)

### Improved

- **Roadmap cleanup** — Removed 45 completed/obsolete plan files, consolidated remaining plans
  - File watcher plan added (3 phases, Issue #66)
  - Brain Quality Track C1+C2 merged
  - v4.0 master plan: all 5 phases complete

### Tests

- 4140 passed, 92 skipped, 1 xfailed

## [4.10.0] - 2026-03-16

### Added

- **Onboarding overhaul (Issue #82)** — Reduce 26 manual setup steps to 1 command
  - `nmem init --full`: auto-detect embeddings, enable dedup, generate maintenance script, print guide URL
  - `nmem doctor` enhanced: 11 checks (was 8), `--fix` flag for auto-remediation (hooks, dedup, embedding)
  - Interactive quickstart guide page (MkDocs + animated terminal demos, scroll reveals, feature cards)
  - Dashboard `GuideCard` for new users (<50 neurons) — dismissible, persisted via localStorage
  - Help button (?) in dashboard TopBar linking to quickstart guide
  - CLI banners link to guide URL after init and doctor
  - 35 new tests (test_full_setup + test_doctor_enhanced)

### Fixed

- **Windows npm install**: OpenClaw plugin postinstall uses cross-platform Node.js instead of Unix shell syntax

## [4.9.0] - 2026-03-16

### Added

- **Knowledge Surface (.nm format)** — Two-tier memory architecture: Tier 1 = `.nm` flat file (~1000 tokens, loaded every session), Tier 2 = `brain.db` SQLite graph (queried on-demand)
  - `.nm` format with 5 sections: GRAPH (causal edges), CLUSTERS (topic groups), SIGNALS (urgent/watching/uncertain), DEPTH MAP (self-routing hints), META (brain stats)
  - `SurfaceGenerator` — algorithmic extraction from brain.db using composite scoring (activation + recency + connections + priority)
  - Depth-aware recall routing: SUFFICIENT entities answered from surface (0 latency), NEEDS_DEEP triggers depth=2 recall
  - Auto-injected into MCP `instructions` on session init for immediate agent context
  - `nmem_surface` MCP tool — generate (rebuild from brain.db) and show (inspect current surface)
  - Auto-regeneration on `nmem_auto(action="process")` session-end
  - Atomic file writes (tmp + rename), project-level and global surface resolution
  - Surface reload on brain switch, cached by brain name
  - 73 new tests across 4 test files

### Fixed

- **CI fixes**: doc_trainer mock using real `BrainConfig` instead of `MagicMock` (lazy entity promotion attrs), auto_tags tests accept bigrams from keyword extractor
- **Docs freshness**: regenerated CLI reference (new PostgreSQL migrate options)

## [4.8.0] - 2026-03-16

### Added

- **B7: Lazy Entity Promotion** — Entities need 2+ mentions before becoming neurons; `entity_refs` table (schema v29), retroactive synapses on promotion, high-confidence/user-tagged exceptions
- **A4: Auto-Importance Scoring** — Heuristic priority when user doesn't set explicit priority; type bonus, causal/comparative language signals, entity richness
- **A4: Reflection Engine** — Accumulates importance from saved memories, detects patterns (recurring entities, temporal sequences, contradictions) at threshold
- **PostgreSQL Migration** — `nmem migrate postgres` CLI command with full connection params (#80)
- **B1-B6, B8: Brain Quality Track B** — Auto-consolidation, Hebbian retrieval, cross-memory linking, IDF keywords, fiber scoring, contextual compression, adaptive decay
- **A1: Smart Instructions** — Decision framework injected into MCP `instructions` to guide proactive memory saving
- **Schema v29** — `entity_refs` table for lazy entity promotion + `keyword_document_frequency` for IDF scoring
- **73 new tests**: lazy entity (11), importance (16), reflection (12), compression (12), adaptive decay (11), postgres migration (5), cross-memory link (9), IDF (7), fiber scoring (8)

### Improved

- All quality improvements are purely algorithmic — zero LLM calls added
- Pipeline steps use `getattr` for backward compat with SimpleNamespace contexts
- Entity ref operations gracefully degrade when table doesn't exist

## [4.7.0] - 2026-03-16

### Added

- **PostgreSQL + pgvector backend** — Full async storage backend via `asyncpg` with vector similarity search. Supports neurons, synapses, fibers, brains, typed queries. Docker Compose included. Contributed by @zsecducna (#56)
- **NeuralMemory vs Mem0 benchmark** — Head-to-head comparison: 121x faster writes, equal accuracy, 0 API calls vs 70. Script at `scripts/benchmark_mem0_vs_nm.py`
- **Chatbot v2** — Upgraded HF Spaces chatbot with conversation memory, cognitive reasoning for low-confidence answers, source citations, and retrieval stats panel

### Fixed

- `ReinforcementManager.reinforce()` test — updated assertion to match batch API (`update_neuron_states_batch`)
- `check_distribution.py` — Fixed ClawHub JSON parser, Windows shell compat, independent version channels

## [4.6.0] - 2026-03-14

### Added

- **`nmem setup rules`** — IDE rules file generator for multi-agent adoption. Generates `.cursorrules`, `.windsurfrules`, `.clinerules`, `GEMINI.md`, and `AGENTS.md` with NM usage instructions. Supports `--all`, `--ide <name>`, `--force`, and interactive selection
- **17 new tests** for IDE rules generator

## [4.5.0] - 2026-03-14

### Added

- **Context merger (Phase A)** — `nmem_remember` accepts optional `context` dict (e.g. `{reason, alternatives, cause, fix, steps}`) that gets merged into content server-side using type-specific templates. Works with any agent — no need to craft perfect prose
- **Quality scorer (Phase B)** — Every `nmem_remember` response now includes `quality` ("low"/"medium"/"high"), `score` (0-10), and `hints` (actionable improvement suggestions). Soft gate: always stores, never rejects
- **36 new tests** for quality scorer (20) and context merger (16)

### Fixed

- **Tool memory config default** — test assertion updated to match `enabled=True` default

## [4.4.1] - 2026-03-14

### Improved

- **Embedding config-status 3-state detection** — Quick Actions card now distinguishes "configured", "installed but disabled", and "not installed" for embedding provider, with actionable enable/disable commands

## [4.4.0] - 2026-03-14

### Added

- **Dashboard Quick Actions card** — Overview page now shows configuration status for 6 features (tool memory, cloud sync, embedding, consolidation, review queue, orphan rate) with actionable shortcut commands and copy buttons
- **`/api/dashboard/config-status` endpoint** — returns per-feature config status with status badges and commands
- **Source-Aware Brain plan** — 4-phase architecture plan for smart index with exact citations from source documents

### Fixed

- **Plugin skills path (#71)** — `skills` field in `plugin.json` changed from `"./SKILL.md"` (file) to `"./skills"` (directory) to match Claude Code's expected format. Fixes 2 load errors on plugin install
- **Tool stats empty** — `tool_memory.enabled` defaulted to `false`, causing dashboard Tool Stats page to show no data. Now defaults to `true` — tool usage tracking works out of the box
- **E2E health test** — fixed assertion mismatch (`"healthy"` vs `"ok"`)

### Added

- **Source-Aware Brain plan** — 4-phase architecture plan for smart index with exact citations from source documents (source locators, `nmem_cite` tool, source refresh, cloud resolvers)

## [4.3.1] - 2026-03-14

### Fixed

- **Plugin manifest validation (#70)** — removed invalid `features`, `instructions`, `agents` keys from `plugin.json` that broke Claude Code plugin install
- **Doc trainer orphan neurons** — heading-less chunks now get synthetic heading from filename; added per-file tags for cross-cluster ENRICH linking; increased heading dedup limit 20→100 for common headings like "Overview"
- **Chatbot brain loading** — use `find_brain_by_name("neuralmemory-docs")` instead of non-existent `list_brains()` method
- **HF deploy script username** — fixed `nhadaututtheky` typo (double t)

### Added

- **`/health` + `/ready` endpoints** — `nmem serve` now exposes health check (brain name, uptime, schema version) and readiness probe (503 when uninitialized) for production monitoring
- **Cloud sync privacy docs** — privacy model table, encryption details, CF free tier limits in `docs/guides/cloud-sync.md`

### Improved

- **Self-hosted cloud sync** — switched default from shared hub to self-hosted model. Users deploy their own CF Worker + D1 database. Data stays on user's own Cloudflare account
- **Sync setup instructions** — updated README, FAQ, dashboard SyncPage, and MCP setup flow to guide self-hosted deployment first

### Tests

- 14 new health endpoint tests
- Total: 3748 passing

## [4.3.0] - 2026-03-13

### Added

- **`nmem_tool_stats` MCP tool** — exposes tool usage analytics (summary + daily breakdown) via MCP (#63)
- **`/api/dashboard/tool-stats` REST endpoint** — tool usage analytics for dashboard integration
- **Dashboard: Tool Stats page** — top tools bar chart, usage-over-time line chart, detailed table with success rates and durations (#63)
- **Background consolidation daemon** — `nmem serve` now runs periodic consolidation using existing `maintenance.scheduled_consolidation_*` config (#65)
- **HuggingFace Spaces deployment** — chatbot ready for HF Spaces with proper metadata, async Gradio handlers, deploy script, and docs guide (#60)
- **Cascading retrieval with fiber summary tier** — FTS5 search on fiber summaries as step 2.8 before neuron pipeline, sufficiency gate for early termination, schema v27 (#61, #62)

### Improved

- **Docs messaging** — restructured README and mcp-server.md with "3 tools you need, 41 the agent handles" hierarchy (#59)

### Fixed

- **`nmem doctor` schema version check** — was using `PRAGMA user_version` (always 0) instead of `schema_version` table; now correctly reports v26
- **`nmem brain health` crash in shared mode** — hardcoded `limit=10000` exceeded server max (1000), causing 422 errors (#67)
- **`nmem info` crash in shared mode** — same limit issue for typed memories query
- **`nmem consolidate` FK crash** — summarize strategy referenced anchor neurons pruned by earlier tier; now validates neuron existence before creating summary fibers (#68)

## [4.1.1] - 2026-03-12

### Fixed

- **`nmem doctor` crash** — fixed `No module named 'neural_memory.storage.sqlite'` caused by stale import after storage restructuring (now imports from `sqlite_schema`)
- **`nmem_pin action=list`** — new `list` action to query pinned fibers (#57)

### Improved

- **Stale references audit** — updated tool counts (39→44), schema version (v22→v26), test counts across README, ROADMAP, plugin.json, mcp-server.md
- **FAQ** — added "Why is my consolidation 0%?" entry
- **Regenerated docs** — MCP tools + CLI reference refreshed for v4.1.x

## [4.1.0] - 2026-03-12

### Added

- **Auto-generated MCP Tool Reference** — `scripts/gen_mcp_docs.py` introspects all 44 MCP tool schemas and generates `docs/api/mcp-tools.md` with parameter tables, categories, and tier badges
- **Auto-generated CLI Reference** — `scripts/gen_cli_docs.py` introspects all 66 CLI commands (Typer/Click) and generates `docs/getting-started/cli-reference.md`
- **Documentation Chatbot** — Gradio UI (`chatbot/app.py`) powered by NeuralMemory's ReflexPipeline, answers docs questions without an LLM using spreading activation retrieval
- **Docs Brain Trainer** — `chatbot/train_docs_brain.py` trains a brain from project docs (40 files → 1045 chunks → 9175 neurons)
- **CI Docs Freshness Check** — new `docs` job in GitHub Actions runs `--check` mode on both generators, fails CI when auto-generated docs are stale

### Fixed

- **Brain lookup fallback** — `get_brain(name)` now falls back to `find_brain_by_name()` when id-based lookup fails, preventing duplicate "brain.v2" creation for users upgrading from older versions with UUID-based brain ids

### Improved

- **Docs navigation** — added orphan pages (Companion Setup, Lessons Learned) to mkdocs.yml nav
- **Cross-links** — CLI Guide, CLI Reference, and MCP Tools Reference now link to each other via admonition boxes
- **CLI Guide renamed** — title changed from "CLI Reference" to "CLI Guide" to avoid confusion with auto-generated reference

## [4.0.1] - 2026-03-12

### Security

- **Fix path traversal** in `index_handler.py` — adapter connection paths now validated with `is_relative_to()` against allowed directories (cwd, home, temp)
- **Fix path traversal** in `pre_compact.py` hook — stdin transcript path now validated against `~/.claude` directory
- **Update `cryptography>=46.0.5`** — fix CVE-2026-26007
- **Add `python-multipart>=0.0.22`** floor constraint — fix CVE-2026-24486
- **Remove internal info from error messages** — 9 locations no longer leak memory IDs, hypothesis IDs, or filesystem paths to clients
- **CORS hardening** — replace `localhost:*` wildcard with explicit port list (3000, 3001, 5173, 5174, 8000, 8080, 8888)

### Fixed

- Fix 8 silent `except Exception: pass` blocks — all now log at DEBUG level with `exc_info=True`
- Fix 14 redundant exception tuples (`except (AttributeError, Exception)` → `except Exception`)
- Remove unused `python-dateutil` from core dependencies

## [4.0.0] - 2026-03-12

### Added

- **Semantic Drift Detection** — Find tag synonyms/aliases via Jaccard similarity on co-occurrence data
- **Tag Co-Occurrence Matrix** — Automatically recorded on every memory encode, tracks which tags appear together
- **Union-Find Clustering** — Groups related tags with confidence thresholds: merge (>0.7), alias (>0.4), review (>0.3)
- **Temporal Drift Detection** — Compares early vs recent session topics to detect terminology shifts
- **`nmem_drift` MCP Tool** — detect/list/merge/alias/dismiss actions for managing drift clusters
- **`detect_drift` Consolidation Strategy** — Runs drift analysis during periodic consolidation
- **Schema v26** — New `tag_cooccurrence` and `drift_clusters` tables

### Improved

- **Brain Intelligence Complete** — v4.0 milestone: session intelligence, adaptive depth, predictive priming, and semantic drift detection work together as feedback loops
- Consolidation engine now includes drift detection in the final tier alongside semantic_link

### Tests

- 51 new drift detection tests (Jaccard, clustering, storage, MCP handler, Union-Find)
- Total: 3810 passing

## [3.5.0] - 2026-03-12

### Added

- **Predictive Priming** — Brain anticipates next query from session context with 4-source priming engine
- **Activation Cache** — Recent query results carry forward as soft activation with exponential decay (`0.7^n` per query)
- **Topic Pre-Warming** — Session topics with EMA > 0.5 pre-warm related neurons before query parsing (truly predictive)
- **Habit-Based Priming** — Query pattern co-occurrence (CONCEPT neurons + BEFORE synapses) predicts next topic, max 3 predicted topics
- **Co-Activation Priming** — Hebbian binding data (strength >= 0.5, count >= 3) boosts associated neurons
- **Priming Metrics** — Hit rate tracking with auto-adjusted aggressiveness (0.5x-1.5x) based on priming effectiveness
- **Session priming fields** — `priming_hit_rate`, `priming_total` exposed in session summaries and result metadata

### Tests

- 57 new tests covering all priming sources, metrics, orchestration, merging, backward compat
- Total: 3759 passing

## [3.4.0] - 2026-03-12

### Added

- **Session-aware depth selection** — Primed topics go shallower (already in context), new topics go deeper (need exploration). Uses session EMA topic weights
- **Calibration-driven gate tuning** — High-accuracy gates get confidence boost (+10%), low-accuracy gates get dampened (-30%), very low avg_confidence triggers downgrade to insufficient
- **Agent feedback signal** — `agent_used_result` parameter: remember-after-recall = strong positive, unused recall = raised bar for success
- **Dynamic RRF weights** — Per-brain retriever weights evolve from outcome history via `retriever_calibration` table and EMA
- **Auto activation strategy** — `activation_strategy="auto"` selects classic/PPR/hybrid based on graph density (synapses/neuron ratio)
- **Schema v25** — `retriever_calibration` table + `graph_density` column on brains

### Tests

- 30 new tests covering all 5 features + backward compatibility
- Total: 3702 passing

## [3.3.0] - 2026-03-12

### Added

- **Cloud Sync Hub** — Cloudflare Workers + D1 sync hub with API key auth, brain ownership, device management. Live at `neural-memory-sync-hub.vietnam11399.workers.dev`
- **API key auth** — `nmk_` prefixed keys, SHA-256 hashed storage, Bearer token transport, key masking in all outputs
- **`nmem_sync_config(action='setup')`** — Guided onboarding flow for cloud sync setup
- **URL versioning** — Cloud hub uses `/v1/` prefix, localhost preserves backward-compatible paths
- **HTTP error mapping** — User-friendly messages for 401/403/413/429 status codes
- **Cloud profile in `nmem_sync_status`** — Shows tier, email, usage when connected to cloud hub
- **HTTPS enforcement** — Refuses non-HTTPS for cloud hub URLs (localhost exempt)

### Tests

- 22 new tests: SyncConfig api_key, key masking, URL versioning, HTTP error handling
- Sync hub: 10 Vitest tests (health, auth, validation, type shapes)
- Total: 3672 passing

## [3.2.0] - 2026-03-11

### Added

- **Session Intelligence (v4.0 Phase 1)** — In-memory session state tracking across MCP calls with topic EMA scoring, LRU eviction (max 10 sessions), 2h auto-expiry, and SQLite persistence via `session_summaries` table (schema v24)
- **Dashboard assets in wheel** — Bundled `server/static/dist/` via hatch artifacts config, fixing blank dashboard on pip install (#54)

### Fixed

- **Config singleton mutation** — `wizard.py` and `embedding_setup.py` now use immutable `replace()` pattern instead of mutating the cached config singleton (H1/H2)
- **Structure detector false positives** — Added 4096-char size guard and CSV all-text column rejection heuristic (H4/H5)
- **Source registry validation** — `_row_to_source()` handles invalid SourceType/SourceStatus gracefully, `update_source()` validates before SQL write (H2/H3)
- **Source handler error handling** — `_require_brain_id()` and `Source.create()` wrapped in try/except ValueError (H1/M1)

### Tests

- 40 new tests for session intelligence (QueryRecord, SessionState EMA, SessionManager LRU, SQLite persistence)
- Total: 3650 passing

## [3.1.0] - 2026-03-11

### Added

- **Source-Aware Memory (v3.0 Pillar 4)** — Brain that knows its sources. 6-phase plan fully shipped.
- **`nmem_show` tool** — Retrieve exact verbatim content of a memory by fiber ID
- **Exact recall mode** — `mode="exact"` in `nmem_recall` returns verbatim content without summarization
- **Source Registry** — Schema v23 with `sources` table, `SOURCE_OF` synapse type, `nmem_source` tool for registering and querying memory provenance
- **Structured encoding** — Schema-aware encoder detects tabular data (CSV, markdown tables, JSON arrays) and preserves structure through the pipeline
- **Citation engine** — `citation.py` generates citation metadata with audit synapses linking memories to their sources
- **`nmem init --wizard`** — Interactive first-run wizard: brain name → embedding provider → MCP config → test memory
- **`nmem doctor`** — System health diagnostics with 8 checks (Python, config, brain, deps, embeddings, schema, MCP, CLI tools)
- **`nmem setup embeddings`** — Interactive embedding provider setup with installation status and API key detection
- **Change log tracking** — `sqlite_change_log.py` records schema and data mutations for audit trail

### Fixed

- **SharedStorage brain_id parity** — Abstract `brain_id` property on base class, all backends implement consistently (#53)
- **Hub auto-creates brain** — First sync or device registration no longer fails on missing brain
- **Error message leaks** — Batch remember no longer exposes `str(e)` exception details to clients

### Improved

- **DX Sprint** — Actionable error messages across CLI and MCP, embedding setup guides new users through provider selection
- **VS Code extension v0.5.0** — 6 lifecycle and config bug fixes

### Tests

- 200+ new tests across all v3.0 phases (show handler, source registry, structured encoding, citation, audit synapses, DX wizard/doctor/embedding)
- Total: 3515 passing

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
