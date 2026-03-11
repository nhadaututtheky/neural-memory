# Phase 1: Exact Recall — nmem_show + mode="exact"

## Goal

Users can retrieve exact, unmodified content of any memory by ID (`nmem_show`), and recall can return raw neuron content without summarization (`mode="exact"`). Closes #35.

## Tasks

- [ ] 1.1: `nmem_show` MCP tool
  - New tool: `nmem_show(memory_id) -> full content + metadata + synapses + source info`
  - Returns: anchor neuron content (verbatim), all neuron types, tags, timestamps, priority
  - Also returns: connected synapses (type, target, weight), fiber metadata
  - Schema in `tool_schemas.py`, handler in `tool_handlers.py`, dispatch in `server.py`
  - Tool count: 40 → 41

- [ ] 1.2: Exact recall mode in retrieval pipeline
  - Add `mode` param to `nmem_recall`: "associative" (default) | "exact"
  - `mode="exact"`: skip summarization step, return raw neuron contents as-is
  - Modify `src/neural_memory/engine/retrieval.py` — conditional bypass of summary
  - Still uses spreading activation for finding — only output format changes

- [ ] 1.3: Update recall tool schema
  - Add `mode` to `nmem_recall` inputSchema in `tool_schemas.py`
  - Update MCP instructions to explain when to use exact mode
  - Backward compatible: default remains "associative"

- [ ] 1.4: REST API endpoint for show
  - `GET /api/memory/{memory_id}` — full memory detail
  - Returns same data as MCP `nmem_show`
  - Reuse same handler logic

- [ ] 1.5: Tests
  - Test nmem_show returns complete memory with all metadata
  - Test nmem_show with invalid ID returns error
  - Test recall mode="exact" returns raw content
  - Test recall mode="associative" (default) unchanged behavior
  - Test REST endpoint

## Acceptance Criteria

- [ ] `nmem_show(memory_id="abc")` returns full verbatim content + metadata + synapses
- [ ] `nmem_recall(query="X", mode="exact")` returns raw neuron contents, no summarization
- [ ] Default recall behavior unchanged (backward compatible)
- [ ] All tests pass, mypy clean

## Files Touched

### New
- `tests/unit/test_show_handler.py`

### Modified
- `src/neural_memory/mcp/tool_schemas.py` — add nmem_show schema + recall mode param
- `src/neural_memory/mcp/tool_handlers.py` — add _show handler + modify _recall for mode
- `src/neural_memory/mcp/server.py` — add dispatch entry
- `src/neural_memory/engine/retrieval.py` — exact mode bypass
- `src/neural_memory/server/routes/` — add show endpoint
- `tests/unit/test_mcp.py` — update tool count
- `tests/unit/test_tool_tiers.py` — update tool count

## Dependencies

- Phase 0 complete (storage.brain_id available)
