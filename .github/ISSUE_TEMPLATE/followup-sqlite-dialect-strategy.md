---
name: SQLite Dialect Write Strategy
about: Make write serialization dialect-aware (SQLite serial, Postgres parallel)
title: "[Strategy] Dialect-aware write serialization"
labels: architecture, storage, postgresql
assignees: ''
---

## Problem
Current fix serializes **all** writes via `SQLiteDialect._write_queue`, including on Postgres where we have a connection pool and could run in parallel.

## Current State
- `SQLiteDialect.execute_write()` → single background worker serializes everything
- `PostgresDialect` inherits this behavior (no override)
- `write_queue.py`, `enrichment_worker.py`, `eternal_handler.py` all now sequential

## Desired Solution
```python
# In Dialect base class
async def execute_write(self, sql: str, params: Sequence[Any] = ()) -> None:
    """Default: serialize (SQLite-safe). Override for pooled backends."""
    # SQLite implementation uses queue
    ...

# In PostgresDialect
async def execute_write(self, sql: str, params: Sequence[Any] = ()) -> None:
    # Use pool - true parallelism
    async with self._pool.acquire() as conn:
        await conn.execute(sql, params)
```

## Affected Call Sites (verify they work with parallel on PG)
- [ ] `pipeline_steps.py`: `CreateSynapsesStep`, `CoOccurrenceStep`
- [ ] `auto_handler.py`: `_save_detected_memories*`
- [ ] `doc_trainer.py`: `_build_heading_hierarchy`, `_build_temporal_topology`
- [ ] `write_queue.py`: `DeferredWriteQueue.flush()`
- [ ] `eternal_handler.py`: `_save_project_context`
- [ ] `enrichment_worker.py`: `process_enrichment_batch`

## Acceptance Criteria
- [ ] SQLite: writes remain serialized (no "SQL statements in progress" errors)
- [ ] Postgres: concurrent writes work (benchmark shows >1x throughput vs serial)
- [ ] No behavior change for callers - they just call `storage.add_synapse()` etc.
- [ ] `pre_ship.py` passes

## Benchmark
Run `scripts/benchmark/serialize_impact.py` (to be created) comparing:
- SQLite serial (baseline)
- Postgres serial (current)
- Postgres parallel (target)