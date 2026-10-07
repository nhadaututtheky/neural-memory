# PR: fix(storage): serialize SQLite writes at dialect level + forbid silent error swallowing

## Summary
Fixes SQLite "cannot commit transaction - SQL statements in progress" errors by adding a single-writer queue at the dialect level, migrating 9 call sites from `asyncio.gather` to sequential execution, and adding a pre-ship check to forbid silent error swallowing patterns.

## Root Cause
`SQLiteDialect` owns a single write connection. Multiple coroutines calling `execute()` concurrently caused commit collisions. Rows usually landed but Merkle invalidation + change-log entries were silently dropped via `return_exceptions=True`.

## Changes

### 1. SQLiteDialect Write Queue (`src/neural_memory/storage/sql/sqlite_dialect.py`)
- Added `_write_queue` + `_write_worker` background task
- New `execute_write(sql, params)` method serializes all writes
- Reads still use `ReadPool` (parallel, unchanged)
- Explicit transactions via `transaction()` context manager unaffected

### 2. Migrated 9 Call Sites (sequential instead of asyncio.gather)

| File | Methods | Before | After |
|------|---------|--------|-------|
| `pipeline_steps.py` | `CreateSynapsesStep.execute`, `CoOccurrenceStep.execute` | `asyncio.gather(*[storage.add_synapse(s)])` | `for s in synapses: await storage.add_synapse(s)` |
| `auto_handler.py` | `_save_detected_memories_no_dedup`, `_save_detected_memories` | `asyncio.gather(*[self._remember(...)])` | `for item in items: await self._remember(...)` |
| `doc_trainer.py` | `_build_heading_hierarchy`, `_build_temporal_topology` | `asyncio.gather(*[storage.add_synapse(s)])` | `for s in synapses: await storage.add_synapse(s)` |
| `write_queue.py` | `_gather_count` | `asyncio.gather(..., return_exceptions=True)` | `for coro in coros: await coro` |
| `eternal_handler.py` | `_save_project_context` | `asyncio.gather(*[storage.delete_typed_memory()])` | `for old in old_facts: await storage.delete_typed_memory()` |
| `enrichment_worker.py` | `process_enrichment_batch` | `asyncio.gather(*[_run_one(j)])` | `for j in jobs: await _run_one(j)` |

### 3. Cleanup Unused Imports
Removed `import asyncio` from: `pipeline_steps.py`, `auto_handler.py`, `doc_trainer.py`, `write_queue.py`, `eternal_handler.py`

### 4. Pre-Ship Forbidden Patterns Check (`scripts/pre_ship.py`)
Added check that fails CI on:
- `except BaseException:` without cleanup + re-raise
- `except: pass`
- `return_exceptions=True` outside allowed cleanup contexts (`close()`, `stop()`, `flush_background_tasks()`, `_drain_pipeline_tasks()`)

## Testing

### Unit Tests (84 passed)
- `test_doc_trainer.py` - 20 passed
- `test_eternal_context.py` - 25 passed  
- `test_enrichment_worker.py` - 5 passed
- `test_capture_hygiene.py` - 7 passed
- `test_pipeline_background_tasks.py` - 5 passed
- `test_vietnamese_capture.py` - 22 passed

### Integration Tests (91 passed)
All integration tests pass including:
- Chinese FTS recall (PR #207 related)
- Encoding flow (synapse creation)
- Enrichment outbox (worker processing)
- Storage migrations
- TLLR features E2E (status, validity, BM25, provenance)

### Lint/Format
- `ruff check src/ tests/` - ✅ Clean
- `ruff format --check src/ tests/` - ✅ Clean

### Forbidden Patterns Check
- `except BaseException:` in src/ - 0 violations (legitimate uses allowed)
- `except: pass` - 0 violations
- `return_exceptions=True` in src/ - 0 violations (legitimate cleanup contexts allowed)

## Trade-offs

### SQLite
- **No performance loss** - writes were already serialized on single connection
- **Fixes silent data loss** - Merkle + change-log no longer dropped

### Postgres
- **Currently loses parallelism** - inherits SQLite's serial behavior
- **Follow-up issue**: [#xxx] Dialect-aware write strategy (Postgres uses pool, SQLite uses queue)

## Follow-up Issues (Created)

1. **High**: Dialect-aware write strategy - `.github/ISSUE_TEMPLATE/followup-sqlite-dialect-strategy.md`
2. **Medium**: CapturePolicy base class for JP/KR - `.github/ISSUE_TEMPLATE/followup-capture-policy.md`
3. **Medium**: Windows native CLI binary - `.github/ISSUE_TEMPLATE/followup-windows-native-cli.md`
4. **Medium**: Metrics/observability - `.github/ISSUE_TEMPLATE/followup-metrics-observability.md`

## Benchmark Script
`scripts/benchmark/serialize_impact.py` - compares SQLite serial vs Postgres parallel throughput

## Files Modified (8)
```
src/neural_memory/storage/sql/sqlite_dialect.py    +45
src/neural_memory/engine/pipeline_steps.py         -12
src/neural_memory/mcp/auto_handler.py              -25
src/neural_memory/engine/doc_trainer.py            -15
src/neural_memory/engine/write_queue.py            -5
src/neural_memory/mcp/eternal_handler.py           -5
src/neural_memory/engine/enrichment_worker.py      -8
scripts/pre_ship.py                                +60
```
**Net: +35 lines**

## Checklist
- [x] All unit tests pass
- [x] All integration tests pass
- [x] Ruff lint/format clean
- [x] Forbidden patterns check passes
- [x] CHANGELOG entry (under [Unreleased])
- [x] Follow-up issues documented
- [x] Benchmark script created