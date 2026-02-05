# Changelog

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
