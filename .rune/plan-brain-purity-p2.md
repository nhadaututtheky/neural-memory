# Phase 2: Dedup Improvement

## Goal

Reduce duplicate neurons entering the brain by tuning the existing 3-tier dedup pipeline. Not rebuilding — just fixing the gaps that let 3,589 duplicates through.

## Why This Matters

- `max_candidates=10` means only 10 existing neurons checked per write — brain with 7k+ neurons misses most
- `simhash_threshold=10` bits (~85% similarity) too loose — semantic near-dupes pass
- Dedup match creates alias + DEDUP synapse but still creates new fiber — bloat
- Session-end consolidation runs MATURE+INFER+ENRICH but NOT DEDUP
- No dedup check on batch operations (remember_batch skips pipeline per-item)

## Tasks

- [x] 2.1 — Tune dedup defaults in `DedupConfig`
  - `max_candidates`: 10 → 30, `simhash_threshold`: 10 → 7
  - Wired `max_candidates` from DedupSettings to DedupConfig in tool_handlers.py
  - Pipeline caps at 50 (safety limit)

- [ ] 2.2 — Dedup match → UPDATE instead of alias (DEFERRED — separate PR, more complex)

- [x] 2.3 — Add DEDUP to session-end consolidation
  - Strategies now: [DEDUP, MATURE, INFER, ENRICH] (DEDUP first)

- [ ] 2.4 — Content hash index (DEFERRED — schema migration, separate PR)

- [x] 2.5 — Tests (17 new tests in test_dedup_improvements.py + 3 fixes in existing tests)
  - Config defaults, DedupSettings, session-end DEDUP, pipeline cap, validation
  - Updated test_dedup_config.py, test_dedup_default.py, test_dedup_integration.py

## Acceptance Criteria

- [ ] Exact-duplicate content never creates new fiber (content hash catches it)
- [ ] Near-duplicate (>89% SimHash) merges into existing fiber instead of alias
- [ ] Session-end runs DEDUP pass automatically
- [ ] max_candidates scales with brain size (not fixed at 10)
- [ ] Existing dedup tests still pass
- [ ] 15+ new tests

## Files Touched

- `src/neural_memory/engine/dedup/pipeline.py` — tune defaults, merge behavior
- `src/neural_memory/engine/dedup/config.py` — update DedupConfig defaults
- `src/neural_memory/mcp/maintenance_handler.py` — add DEDUP to session-end
- `src/neural_memory/storage/sqlite_schema.py` — content_hash column + index
- `src/neural_memory/storage/sqlite_neuron.py` — hash lookup method
- `tests/unit/test_dedup_pipeline.py` — extend existing tests
- `tests/unit/test_dedup_merge.py` — new: merge behavior tests

## Design Notes

- Content hash = SHA-256 of `content.strip().lower()` — simple normalization
- Schema migration adds column with NULL default, background job fills existing
- Merge preserves older fiber's ID (stable references) but takes newer content if longer
- DEDUP in session-end is lightweight — only checks fibers created this session against existing
