# Phase 4: Citation Engine + Audit Synapses — v3.0.0

## Goal

Every recalled neuron can produce a citable reference. Audit trail tracks who stored what, when, and why. This completes Pillar 4 (Source-Aware Memory) and earns the v3.0 tag.

## Tasks

- [ ] 4.1: Citation format engine
  - New module `src/neural_memory/engine/citation.py`
  - `format_citation(neuron, source, format)` → citation string
  - Formats: "inline" (`[cite:source/article]`), "footnote", "full"
  - Legal: `[BLDS 2015, Điều 468]` — source name + article reference
  - Accounting: `[Cafe Saigon, 2026-01-15, Salary]` — source + date + category
  - Generic: `[source_name, neuron_id[:8]]`
  - Configurable via BrainConfig `citation_format`

- [ ] 4.2: Citation in recall response
  - Extend recall result items with `citation` field
  - When neuron has SOURCE_OF synapse → auto-generate citation
  - Citation includes: source name, source type, version, effective_date
  - `include_citations=True` (default) in recall args

- [ ] 4.3: Audit synapse types
  - Add to `SynapseType` enum: `STORED_BY`, `VERIFIED_AT`, `APPROVED_BY`
  - `STORED_BY`: auto-created on encode, target = agent/user identifier
  - `VERIFIED_AT`: manual via `nmem_source(action="verify", source_id=...)`
  - `APPROVED_BY`: manual via `nmem_source(action="approve", source_id=...)`
  - Synapse metadata stores timestamp + actor

- [ ] 4.4: Audit trail in recall
  - When recalled, include audit info: who stored, when, verification status
  - `audit: {stored_by, stored_at, verified: bool, verified_at, approved_by}`
  - Only populated when audit synapses exist (backward compatible)

- [ ] 4.5: `nmem_provenance` MCP tool
  - Action: "trace" — given a neuron_id, return full provenance chain
  - Output: source → stored_by → verified_at → approved_by + timestamps
  - Action: "verify" — mark a neuron/source as verified by current agent
  - Action: "approve" — mark a neuron/source as approved

- [ ] 4.6: Tests
  - Citation formatting for all formats (inline, footnote, full)
  - Recall includes citations when source exists
  - Audit synapses created on encode
  - Provenance trace returns complete chain
  - Backward compat: neurons without source/audit still recall fine

## Acceptance Criteria

- [ ] `nmem_recall(query="lãi suất")` returns `citation: "[BLDS 2015, Điều 468]"` in result
- [ ] `nmem_provenance(action="trace", neuron_id="...")` returns full chain
- [ ] `nmem_provenance(action="verify", neuron_id="...")` creates VERIFIED_AT synapse
- [ ] Recall without sources still works (no regression)
- [ ] All citation formats produce valid, parseable output
- [ ] v3.0.0 version bump passes all checks

## Files Touched

### New
- `src/neural_memory/engine/citation.py`
- `tests/unit/test_citation_engine.py`
- `tests/unit/test_audit_synapses.py`

### Modified
- `src/neural_memory/core/synapse.py` — STORED_BY, VERIFIED_AT, APPROVED_BY
- `src/neural_memory/engine/retrieval.py` — citation + audit in results
- `src/neural_memory/engine/encoder.py` — auto-create STORED_BY on encode
- `src/neural_memory/mcp/tool_schemas.py` — nmem_provenance schema
- `src/neural_memory/mcp/tool_handlers.py` — provenance handler
- `src/neural_memory/mcp/server.py` — dispatch nmem_provenance
- `src/neural_memory/storage/base.py` — provenance query stubs

## Dependencies

- Phase 2 complete (Source Registry + SOURCE_OF synapse)
- Phase 3 complete (structured metadata for rich citations)
