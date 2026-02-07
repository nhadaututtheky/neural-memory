# Changelog

## [0.1.9] - 2026-02-07

### Changed

- Updated marketplace listing with new features and commands
- Eternal Context now stores everything in the neural graph (no more JSON sidecar files)
- Decisions, instructions, and project context are fully discoverable via recall/spreading activation
- MCP server refactored into modular handler mixins for better maintainability
- `max_tokens` parameter capped at 10,000 (schema + runtime validation)
- API key handling prefers environment variables (`MEM0_API_KEY`, `COGNEE_API_KEY`) over connection params
- CLI commands refactored: smaller functions, better error messages
- 584 tests passing, all bare `except` clauses eliminated
- Synced with NeuralMemory v0.9.3

## [0.1.7] - 2026-02-07

### Security

- Input validation: content length limits (100KB), enum validation, path boundary checks
- ReDoS protection: text truncation before regex processing
- Spreading activation queue cap to prevent memory exhaustion
- Synced with NeuralMemory v0.9.2 security hardening

## [0.1.6] - 2026-02-07

### Added

- Eternal Context commands: Recap Session (`Ctrl+Shift+M C`), Recap by Topic, Save Eternal Context, Eternal Context Status
- Import Memories from external sources (ChromaDB, Mem0, AWF, Cognee, Graphiti, LlamaIndex)
- Synced with NeuralMemory v0.9.2: 3-tier auto-save, trigger engine, lazy loading

## [0.1.5] - 2026-02-06

### Changed

- Parallel activation: asyncio.gather for reflex + classic activation (~3x speedup)
- Scored intent detection: fixes first-match-wins misclassification
- Weighted keywords with bi-gram extraction and position-based scoring
- Code entity detection: PascalCase, snake_case, file path patterns
- Content-aware synapse weights: mention frequency + keyword importance
- Session-aware recall: injects active session context into queries

## [0.1.4] - 2026-02-06

### Changed

- Intelligence upgrade: multi-factor confidence scoring (freshness + frequency)
- Frequency-boosted spreading activation (myelination metaphor)
- Recall reinforcement feedback loop — recalled memories become easier to find
- Complexity-aware depth detection for multi-entity queries
- Batch neuron fetch for 25x fewer DB queries in context formatting
- Composite fiber scoring (salience * freshness * conductivity)
- Pathway index cache for O(1) fiber position lookups

## [0.1.3] - 2026-02-06

### Changed

- Sync with NeuralMemory v0.9.1 system optimizations
- Backend: fiber-neuron junction table for 100-1000x faster lookups
- Backend: sigmoid time decay, weighted co-activation binding
- Backend: smarter auto-capture with length validation and hash dedup
- Backend: bridge synapse protection, time-aware fiber merging

## [0.1.2] - 2026-02-06

### Added

- Codebase indexing command (`Ctrl+Shift+M I`)
- New neuron types: Spatial (Files), Sensory, Intent in tree view

## [0.1.1] - 2026-02-05

### Fixed

- Remove Windows `nul` artifact from VSIX package
- Remove broken screenshot URLs from README

## [0.1.0] - 2026-02-05

### Added

- Memory tree view in dedicated activity bar sidebar
  - Neurons grouped by type (Concept, Entity, Action, Time, State)
  - Relative timestamps and neuron counts per group
  - Click to recall related memories
- Interactive graph explorer with Cytoscape.js
  - Force-directed layout with type-based color coding
  - Sub-graph navigation via double-click
  - Node details panel with recall actions
  - Dark/light theme support
- Encode commands
  - Encode selected text as memory with tag selection
  - Encode typed input as memory
- Recall workflow with depth selection (Instant, Context, Habit, Deep)
  - Paste to editor, copy to clipboard, or view full details
- CodeLens integration
  - Memory counts on functions and classes (Python, TS, JS, Go, Rust, Java, C#)
  - Comment trigger detection (`remember:`, `note:`, `decision:`, `todo:`)
- Brain management (switch, create) via status bar and command palette
- Real-time WebSocket sync for tree, graph, and status bar
- Configurable server URL, Python path, graph node limit, and CodeLens triggers
- Status bar with live brain stats (neurons, synapses, fibers)
