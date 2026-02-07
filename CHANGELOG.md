# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
