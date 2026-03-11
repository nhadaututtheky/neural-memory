# Performance Audit Report — NeuralMemory Project

**Date:** 2025-03-12  
**Scope:** Toàn bộ project (storage, engine, MCP, server, CLI)  
**Auditor:** Performance Engineer Team  

---

## Executive Summary

| Category | Status | Top Issues |
|----------|--------|------------|
| Database I/O | ⚠️ | Large batch fetches (100k neurons), N+1 trong lifecycle/cognitive |
| Async/Parallel | ✅ | ReadPool, batch APIs, asyncio.gather |
| Caching | ✅ | NeuronLookupCache, TTL 60s |
| Hot Paths | ⚠️ | Consolidation, enrichment, semantic_discovery full scans |
| Memory | ⚠️ | find_neurons(limit=100000) có thể OOM |

---

## 1. Database Layer

### 1.1 SQLite

| Aspect | Implementation | Assessment |
|--------|----------------|------------|
| WAL mode | `PRAGMA journal_mode=WAL` | ✅ Cho phép đọc song song |
| ReadPool | 3 connections round-robin | ✅ Giảm contention |
| busy_timeout | 5000ms | ✅ Tránh lock timeout |
| cache_size | -8000 (8MB) | ✅ Hợp lý cho personal use |

**Vấn đề tiềm ẩn:**
- `executemany` cho fiber_neurons: SQLite dùng batch tốt
- Một writer duy nhất — bottleneck khi encode nhanh liên tục

### 1.2 PostgreSQL (mới thêm)

| Aspect | Implementation | Assessment |
|--------|----------------|------------|
| Connection pool | asyncpg min=1, max=10 | ✅ |
| command_timeout | 60s | ✅ |
| fiber_neurons insert | Loop `await _query` từng row | ⚠️ N queries thay vì 1 batch |

**Khuyến nghị:** Postgres fiber_neurons nên dùng `executemany` hoặc `COPY` thay vì loop INSERT.

### 1.3 FalkorDB

| Aspect | Implementation | Assessment |
|--------|----------------|------------|
| Connection pool | BlockingConnectionPool max 16 | ✅ |
| socket_timeout | 10s | ✅ |
| Graph traversal | Cypher native (shortestPath) | ✅ Tối ưu hơn BFS SQL |

---

## 2. N+1 Query Patterns

### 2.1 Đã xử lý đúng

| Location | Pattern |
|----------|---------|
| `activation.py` | `get_neurons_batch`, `get_neuron_states_batch`, `get_synapses_for_neurons` |
| `retrieval_context.py` | `get_neurons_batch(anchor_ids)` |
| `compression.py` | `get_neurons_batch(list(fiber.neuron_ids))` |
| `reconstruction.py` | `get_neurons_batch(top_ids)` |
| `lifecycle.py` | `get_neuron_states_batch`, `get_synapses_for_neurons` |
| `cognitive_handler` | Comment "Batch-fetch neurons to avoid N+1" |

### 2.2 Vấn đề — cognitive_handler L977-982

```python
neurons = await asyncio.gather(*(storage.get_neuron(nid) for nid in all_neuron_ids))
```

**Phân tích:** Dùng `asyncio.gather` với N coroutines `get_neuron` → N round-trips riêng. Storage đã có `get_neurons_batch` — nên dùng:

```python
neurons_map = await storage.get_neurons_batch(all_neuron_ids)
neurons = [neurons_map.get(nid) for nid in all_neuron_ids]
```

**Tác động:** Với 50 hypotheses + 50 predictions = 100 IDs → 100 query thay vì 1 batch.

### 2.3 Vấn đề — lifecycle.py L261, L327, L341

```python
for synapse_id in fiber.synapse_ids:
    ...
    await storage.update_synapse(reinforced)  # Sequential
```

```python
for neuron_id in neuron_ids:
    ...
    await storage.update_neuron_state(reinforced_state)  # Sequential
```

**Phân tích:** Không có `update_synapses_batch` / `update_neuron_states_batch`. Mỗi vòng lặp 1 round-trip. Với fiber 50 synapses → 50 UPDATE. Với 100 neurons → 100 UPDATE.

**Khuyến nghị:** Thêm batch update APIs hoặc gộp trong transaction + executemany.

---

## 3. Large Data Fetches (Memory Risk)

### 3.1 find_neurons pagination (đã fix)

**Trước đây:** `find_neurons(limit=100000)` — OOM risk, storage cap 1000 nên chỉ xử lý subset.

**Đã sửa (2025-03):**
- Thêm `offset` vào `find_neurons` (base, sqlite, postgres, falkordb, memory_store)
- Full-scan path: limit tối đa 10000, ORDER BY id cho pagination
- **consolidation _prune**: paginate batch 5k, thu thập hết orphans
- **consolidation _dedup**: paginate batch 5k, thu thập hết anchors
- **semantic_discovery**: paginate batch 5k, thu thập hết CONCEPT+ENTITY

→ Logic chính xác, không còn bỏ sót do cap.

### 3.2 get_fibers(limit=10000)

| File | Limit |
|------|-------|
| consolidation.py | 10000 |
| enrichment.py | 10000 |
| compression.py | 10000 |
| diagnostics.py | 10000 |
| brain_evolution.py | 10000 |

**Phân tích:** 10k fibers × ~500B–2KB ≈ 5–20MB. Chấp nhận được. Nhưng consolidation gọi nhiều lần (`_prune`, `_merge`, `_summarize`, `_mature`, `_dream`, `_dedup`, `_compress`) — mỗi strategy fetch 10k fibers độc lập → có thể cache hoặc truyền fibers qua pipeline.

---

## 4. Hot Paths

### 4.1 Retrieval (ReflexPipeline)

| Bước | Hoạt động | Độ phức tạp |
|------|-----------|-------------|
| Anchor fetch | FTS5/embedding + get_neurons_batch | O(K) |
| Spreading | get_neighbors, get_synapses_for_neurons | O(hops × avg_degree) |
| RRF fusion | In-memory | O(anchors × candidates) |

**Benchmark hiện tại:** Hybrid 0.67–7.81ms cho 100–5000 neurons. Tốt.

### 4.2 Encoding (MemoryEncoder)

- `find_neurons_exact_batch` — đã batch
- `NeuronLookupCache` — cache exact-match (TTL 60s, max 2000 entries)
- Pipeline steps: time/entity/action/intent — mỗi step vài `find_neurons` với limit 1–10

**Đánh giá:** Khá tối ưu. Cache giảm đáng kể DB hits cho content trùng lặp.

### 4.3 Consolidation

- Chạy nhiều strategies: prune, merge, summarize, mature, dream, habits, dedup, semantic_link, compress, tool_events
- Mỗi strategy: `get_fibers(limit=10000)` hoặc `find_neurons(limit=100000)`
- Sequential: strategy 1 → strategy 2 → … (có lý do — phụ thuộc dữ liệu)

**Bottleneck:** Full scan neurons/fibers mỗi strategy. Với brain 50k neurons → mỗi run consolidation có thể fetch 50k + 10k × 9 = 140k rows tổng cộng.

---

## 5. Algorithmic Complexity

### 5.1 get_path (BFS)

| Backend | Implementation | Complexity |
|---------|----------------|------------|
| SQLite | BFS với queue, mỗi bước 1–2 SQL | O(E) trong worst case |
| FalkorDB | `shortestPath` Cypher | O(V+E) native |
| PostgreSQL | BFS Python tương tự SQLite | O(E) |

**Lưu ý:** max_hops đã cap 10 (postgres), SQLite không cap trong code — nên thêm `min(max_hops, 10)` cho consistency.

### 5.2 ReinforcementManager.reinforce

- Batch `get_neuron_states_batch` — OK
- Loop `update_neuron_state` — sequential
- `find_fibers_batch(neuron_ids[:10], limit_per_neuron=3)` — OK
- Loop `get_maturation` + `save_maturation` — N round-trips (N ≤ 10)

### 5.3 _evict_oldest (NeuronLookupCache)

```python
oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
```

**O(n)** mỗi lần evict. Với max 2000 entries — chấp nhận được. Có thể cải thiện bằng heap/LRU nếu cần.

---

## 6. Connection Pool & Timeouts

| Component | Timeout | Pool |
|-----------|---------|------|
| SQLite ReadPool | N/A | 3 connections |
| PostgreSQL | command_timeout 60s | min 1, max 10 |
| FalkorDB | socket 10s, connect 5s | max 16 |
| SharedStorage (aiohttp) | configurable (default 30s) | Session per request |
| MCP tool call | 30s | N/A |
| Ollama embedding | 300s | httpx AsyncClient |

---

## 7. External API Calls

| Service | Batch | Timeout |
|---------|-------|---------|
| Gemini embedding | 100 texts/request | Default |
| Ollama | _BATCH_SIZE (capped) | 300s |
| Mem0 sync | N/A | 30s |
| Sync hub | N/A | 10s |

---

## 8. Recommendations Summary

| Priority | Issue | Location | Action |
|----------|-------|----------|--------|
| **High** | N+1: asyncio.gather(get_neuron × N) | cognitive_handler.py L981 | Dùng `get_neurons_batch` |
| ~~High~~ | ~~find_neurons(limit=100000) OOM risk~~ | consolidation.py, semantic_discovery.py | ✅ Đã fix: pagination batch 5k |
| **Medium** | Sequential update_synapse loop | lifecycle.py L261 | Batch update API |
| **Medium** | Sequential update_neuron_state loop | lifecycle.py L327 | Batch update API |
| **Medium** | Postgres fiber_neurons: loop INSERT | postgres_fibers.py | executemany / COPY |
| **Low** | get_path max_hops không cap (SQLite) | sqlite_synapses.py | `max_hops = min(max_hops, 10)` |
| **Low** | Consolidation: repeated get_fibers | consolidation.py | Cache fibers trong 1 run |

---

## 9. Benchmark Summary (từ docs/benchmarks.md)

| Metric | Value |
|--------|-------|
| Classic vs Hybrid speedup | 3–9× |
| Classic vs Reflex speedup | 34–152× |
| Hybrid recall | ~58% |
| Full pipeline (15 memories, 5 queries) | ~16ms total |
| Activation (5000 neurons) | Hybrid ~0.67ms |

---

## 10. Conclusion

Project đã có nền tảng performance tốt: ReadPool, batch APIs, cache, WAL. Các điểm cần cải thiện:

1. **N+1 trong cognitive_handler** — chuyển sang `get_neurons_batch`.
2. **Full scan 100k neurons** — thêm pagination hoặc streaming.
3. **Sequential updates trong lifecycle** — xem xét batch update APIs.
4. **Postgres fiber_neurons** — dùng batch insert thay vì loop.
