# Phase 2: Source Registry — Schema v23 + SOURCE_OF Synapse

## Goal

Sources become first-class entities. Every neuron can answer "where did this come from?" via SOURCE_OF synapses. Sources have version, status, and integrity tracking.

## Tasks

- [ ] 2.1: Schema migration v23 — `sources` table
  - New table: `sources(id, brain_id, name, type, version, effective_date, expires_at, status, file_hash, metadata, created_at, updated_at)`
  - `type`: "law", "contract", "ledger", "document", "api", "manual"
  - `status`: "active", "superseded", "repealed", "draft"
  - Add migration in `src/neural_memory/storage/migrations/`
  - Update schema version constant

- [ ] 2.2: Source dataclass + storage mixin
  - New frozen dataclass `Source` in `src/neural_memory/core/source.py`
  - New mixin `SQLiteSourceMixin` in `src/neural_memory/storage/mixins/`
  - CRUD: `add_source`, `get_source`, `update_source`, `list_sources`, `delete_source`
  - `find_sources(name=, type=, status=)` with parameterized queries

- [ ] 2.3: SOURCE_OF synapse type
  - Add `SOURCE_OF` to `SynapseType` enum if not exists
  - When encoding with `source_id`, auto-create `SOURCE_OF` synapse from anchor neuron to source
  - Source ID stored as synapse target

- [ ] 2.4: `nmem_source` MCP tool
  - Actions: "register" | "list" | "get" | "update" | "delete"
  - `register`: create source entry, return source_id
  - `list`: filter by type, status, brain
  - `get`: full source detail + linked neuron count
  - `update`: change status (e.g. "active" → "superseded"), version
  - `delete`: soft-delete (mark superseded), warn if neurons linked

- [ ] 2.5: Enhance `nmem_remember` with source_id
  - Add optional `source_id` param to remember/remember_batch
  - When provided, auto-create SOURCE_OF synapse
  - Validate source_id exists before encoding

- [ ] 2.6: Enhance `nmem_recall` response with source info
  - When recalled neuron has SOURCE_OF synapse, include source metadata in response
  - `source: {name, type, version, status}` in each result item

- [ ] 2.7: Tests
  - Source CRUD operations
  - SOURCE_OF synapse auto-creation on remember
  - Recall includes source metadata
  - Schema migration (fresh + upgrade path)
  - Source deletion with linked neurons warning

## Acceptance Criteria

- [ ] `nmem_source(action="register", name="BLDS 2015", type="law")` creates source
- [ ] `nmem_remember(content="...", source_id="src-123")` creates neuron + SOURCE_OF synapse
- [ ] `nmem_recall(query="lãi suất")` response includes `source: {name: "BLDS 2015", ...}`
- [ ] Schema v23 migration works (fresh + upgrade)
- [ ] All tests pass, mypy clean

## Files Touched

### New
- `src/neural_memory/core/source.py`
- `src/neural_memory/storage/mixins/sqlite_source_mixin.py`
- `src/neural_memory/storage/migrations/v23_sources.py`
- `tests/unit/test_source_registry.py`

### Modified
- `src/neural_memory/storage/base.py` — add source method stubs
- `src/neural_memory/storage/sqlite_storage.py` — include source mixin
- `src/neural_memory/core/synapse.py` — add SOURCE_OF if missing
- `src/neural_memory/mcp/tool_schemas.py` — nmem_source schema + source_id on remember
- `src/neural_memory/mcp/tool_handlers.py` — _source handler + remember enhancement
- `src/neural_memory/mcp/server.py` — dispatch
- `src/neural_memory/engine/retrieval.py` — attach source info to results

## Dependencies

- Phase 1 complete (exact recall available for source content viewing)
