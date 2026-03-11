# Phase 0: Stability — SharedStorage Parity + Bug Fixes

## Goal

Fix SharedStorage crash (#53) and establish storage-backend-agnostic patterns so all 20+ `_current_brain_id` refs work with both SQLite and SharedStorage. Close remaining bugs.

## Tasks

- [ ] 0.1: Add abstract `brain_id` property to `NeuralStorage` base class
  - `src/neural_memory/storage/base.py` — add `@property brain_id -> str | None`
  - `src/neural_memory/storage/sqlite_storage.py` — implement returning `_current_brain_id`
  - `src/neural_memory/storage/shared_storage.py` — implement returning `_brain_id`
  - This is the root fix — all code should use `storage.brain_id` not `storage._current_brain_id`

- [ ] 0.2: Fix `_helpers.py` logic precedence (Bug 1 from #53)
  - `src/neural_memory/cli/_helpers.py:82` — `force_sqlite` must override `is_shared_mode`
  - `use_shared = (config.is_shared_mode or force_shared) and not force_local and not force_sqlite`

- [ ] 0.3: Replace all `_current_brain_id` refs with `storage.brain_id`
  - cli/commands/train.py, version.py, habits.py, memory.py
  - cli/storage.py, graph_export.py, tui.py
  - mcp/tool_handlers.py, maintenance_handler.py
  - ~20 locations total — grep + replace, verify each

- [ ] 0.4: Tests for SharedStorage compatibility
  - Test `storage.brain_id` on both backends
  - Test `get_storage(config, force_sqlite=True)` when `is_shared_mode=True`
  - Test `nmem train` with mock SharedStorage

- [ ] 0.5: Close #52 (already fixed) and #53

## Acceptance Criteria

- [ ] `grep -r "_current_brain_id" src/` returns 0 hits outside storage implementations
- [ ] `nmem train` works with SharedStorage (no AttributeError)
- [ ] All existing tests pass
- [ ] mypy clean

## Files Touched

- `src/neural_memory/storage/base.py` — add abstract property
- `src/neural_memory/storage/sqlite_storage.py` — implement property
- `src/neural_memory/storage/shared_storage.py` — implement property
- `src/neural_memory/cli/_helpers.py` — fix logic precedence
- ~15 files with `_current_brain_id` refs — migrate to `storage.brain_id`
- `tests/unit/test_shared_storage_parity.py` — new

## Dependencies

- None — this is the foundation phase
