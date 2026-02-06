# Changelog

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
