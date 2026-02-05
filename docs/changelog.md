# Changelog

All notable changes to NeuralMemory are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.7.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.7.0
[0.6.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.6.0
[0.5.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.5.0
[0.4.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.4.0
[0.3.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.3.0
[0.2.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.2.0
[0.1.0]: https://github.com/nhadaututtheky/neural-memory/releases/tag/v0.1.0
