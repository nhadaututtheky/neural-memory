# Phase 1: Schema + Essence Field

## Goal
Add `essence` column to Fiber, build extractive essence generator, hook into consolidation.
No LLM required. This is the data foundation for all fidelity features.

## Tasks
- [x] Add `essence: str | None = None` to Fiber dataclass (`core/fiber.py`)
- [x] Update `Fiber.create()` to accept optional `essence` param
- [x] Add `Fiber.with_essence()` immutable update method
- [x] Schema migration v34: `ALTER TABLE fibers ADD COLUMN essence TEXT`
- [x] Update `sqlite_fibers.py` INSERT/UPDATE to include essence
- [x] Update `row_to_fiber()` in row mappers (backward compat: `"essence" in row_keys`)
- [x] Create `engine/fidelity.py` with `extract_essence(content: str) -> str`
  - Sentence-level scoring: entity density + position bias (first/last)
  - Entity detection: acronyms, multi-word proper nouns, code refs, quoted terms
  - Common starter filtering (The, This, That, etc.)
  - Return highest-scoring sentence, max 150 chars
  - Fallback: truncated at word boundary with "..."
- [x] Hook into consolidation MATURE strategy (Phase 3 of _mature())
- [x] Add `ESSENCE_BACKFILL` consolidation strategy (standalone + in MATURE)
  - Finds fibers with anchor neuron content but no essence
  - Generates extractive essence, updates fiber
  - Added to STRATEGY_TIERS (tier 3 with SUMMARIZE/INFER)
- [x] Add `essences_generated` counter to ConsolidationReport
- [x] Add `decay_floor` to BrainConfig: `decay_floor: float = 0.05`
- [x] 25 new tests in `test_fidelity.py` (all passing)
- [x] Fix pre-existing test count assertions (doctor 11→12, schema 33→34)

## Acceptance Criteria
- [x] New fibers after consolidation have `summary` + `essence`
- [x] `essence_backfill` populates existing fibers
- [x] Essence is 1 sentence, max 150 chars
- [x] Schema migration is non-breaking (nullable column)
- [x] 4280 unit tests passing, 0 failures
- [x] Mypy: 0 new errors (only pre-existing llm_judge.py)
- [x] Ruff: all checks passed

## Files Touched
- `src/neural_memory/core/fiber.py` — modify (add field + with_essence)
- `src/neural_memory/core/brain.py` — modify (decay_floor)
- `src/neural_memory/storage/sqlite_schema.py` — modify (v33→v34, essence column)
- `src/neural_memory/storage/sqlite_fibers.py` — modify (INSERT/UPDATE)
- `src/neural_memory/storage/sqlite_row_mappers.py` — modify (row_to_fiber)
- `src/neural_memory/engine/fidelity.py` — NEW (~120 LOC)
- `src/neural_memory/engine/consolidation.py` — modify (ESSENCE_BACKFILL strategy + hook)
- `tests/unit/test_fidelity.py` — NEW (25 tests)
- `tests/unit/test_baby_mi_features.py` — fix schema version
- `tests/unit/test_cascading_retrieval.py` — fix schema version
- `tests/unit/test_ephemeral.py` — fix schema version
- `tests/unit/test_source_registry.py` — fix schema version
- `tests/unit/test_doctor_enhanced.py` — fix doctor check count
- `tests/unit/test_dx_wizard.py` — fix doctor check count + mock

## Dependencies
- None (first phase)
