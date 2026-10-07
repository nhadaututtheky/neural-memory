---
name: Metrics & Observability
about: Add Prometheus metrics for failure modes fixed in PR #209/#210
title: "[Observability] Metrics for silent failure detection"
labels: observability, metrics, prometheus
assignees: ''
---

## Problem
PR #209 and #210 fixed **silent failures** that had no visibility:
1. SQLite write collisions → Merkle/change-log dropped silently
2. Windows encoding damage → CJK stored as `???` or `\uXXXX` 
3. Auto-capture empty results → indistinguishable from "nothing to capture"

Without metrics, we can't detect regressions or measure production health.

## Solution: Prometheus Metrics

### 1. SQLite Write Collisions
```python
# src/neural_memory/utils/metrics.py
from prometheus_client import Counter, Histogram

sqlite_write_collision_total = Counter(
    "neural_memory_sqlite_write_collision_total",
    "SQLite commit collisions detected (serialized writes)",
    ["dialect"],  # "sqlite" or "postgres"
)

sqlite_write_latency_seconds = Histogram(
    "neural_memory_sqlite_write_latency_seconds",
    "Time spent waiting for write queue + executing",
    ["dialect"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
```

**Instrument in `SQLiteDialect._write_worker`:**
```python
async def _write_worker(self) -> None:
    while True:
        sql, params, fut = await self._write_queue.get()
        if sql is None:
            break
        start = time.monotonic()
        try:
            await self._conn.execute(sql, params)
            await self._conn.commit()
            sqlite_write_latency_seconds.labels("sqlite").observe(time.monotonic() - start)
            fut.set_result(None)
        except Exception as e:
            if "SQL statements in progress" in str(e):
                sqlite_write_collision_total.labels("sqlite").inc()
            fut.set_exception(e)
```

### 2. CLI Encoding Damage
```python
cli_encoding_damage_rejected_total = Counter(
    "neural_memory_cli_encoding_damage_rejected_total",
    "CLI input rejected due to encoding damage",
    ["command", "damage_type"],  # remember/forget/list, question_mark_flood/unicode_escape/replacement_char
)
```

**Instrument in `detect_encoding_damage()`:**
```python
def detect_encoding_damage(content: str) -> EncodingDamageResult:
    # ... existing logic ...
    if damaged:
        cli_encoding_damage_rejected_total.labels(
            command=current_command,  # from context
            damage_type=damage_type
        ).inc()
```

### 3. Auto-Capture Empty Results
```python
auto_capture_empty_total = Counter(
    "neural_memory_auto_capture_empty_total",
    "Auto-capture returned no memories",
    ["reason", "language"],  # cjk_unsupported/too_short/no_trigger, zh/vi/en/ja/ko
)
```

**Instrument in `empty_capture_hint()`:**
```python
def empty_capture_hint(text: str, language: str) -> str:
    reason = classify_reason(text, language)
    auto_capture_empty_total.labels(reason=reason, language=language).inc()
    return hint_text
```

### 4. Forbidden Pattern Violations (Pre-Ship)
```python
forbidden_pattern_violations_total = Counter(
    "neural_memory_forbidden_pattern_violations_total",
    "Forbidden patterns caught in CI (should be 0)",
    ["pattern", "file"],
)
```

## Implementation
1. Add `prometheus-client` to `pyproject.toml` (already in deps? check)
2. Create `src/neural_memory/utils/metrics.py`
3. Instrument 3 locations above
4. Add `/metrics` endpoint to MCP server (optional) and FastAPI server
5. Document in `docs/guides/observability.md`

## Acceptance Criteria
- [ ] `sqlite_write_collision_total` increments on collision (test with concurrent writes)
- [ ] `cli_encoding_damage_rejected_total` increments on damaged input (test with `???`, `\uXXXX`)
- [ ] `auto_capture_empty_total` increments with correct labels (test with short/unsupported input)
- [ ] Metrics exposed at `GET /metrics` on FastAPI server
- [ ] Grafana dashboard JSON committed to `docs/grafana/`
- [ ] Alert rules: `sqlite_write_collision_total > 0` = warning

## Related
- PR #210: SQLite write serialization
- PR #209: Chinese auto-capture + encoding guard
- `scripts/pre_ship.py` forbidden patterns check