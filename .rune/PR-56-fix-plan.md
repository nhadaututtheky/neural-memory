# PR #56 — Fix Plan: PostgreSQL + pgvector Backend

> **Nguồn:** [PR #56 Review Comments](https://github.com/nhadaututtheky/neural-memory/pull/56)  
> **Mục tiêu:** Senior Python dev fix các vấn đề được reviewer chỉ ra.

---

## Tổng quan mức độ ưu tiên

| Ưu tiên | Mã | Mô tả | File(s) |
|---------|-----|-------|---------|
| **CRITICAL** | C1 | asyncpg `IN ($2, $3)` sai cú pháp → crash | postgres_synapses.py |
| **CRITICAL** | C2 | 8+ abstract methods chưa implement → TypeError | postgres_store.py, postgres_*.py |
| **CRITICAL** | C3 | Không có test | tests/storage/postgres/ |
| **HIGH** | H1 | `refractory_until` ghi sai giá trị (data corruption) | postgres_neurons.py |
| **HIGH** | H2 | Brain auto-create bỏ qua khi switch brain | unified_config.py |
| **HIGH** | H3 | Exception bị nuốt ở `logger.debug` | postgres_neurons.py |
| **HIGH** | H4 | Tags filter sau LIMIT → kết quả sai | postgres_fibers.py |
| **HIGH** | H5 | String `%s` trong SQL (vi phạm convention) | postgres_synapses.py |
| **MEDIUM** | M1 | `embedding vector(384)` hardcode | postgres_schema.py |
| **MEDIUM** | M2 | `row_to_fiber` tags fallback khác SQLite | postgres_row_mappers.py |
| **MEDIUM** | M3 | `unset_shared_storage()` có clear `_postgres_storage`? | unified_config.py |
| **MEDIUM** | M4 | `is_public INTEGER` → dùng `BOOLEAN` | postgres_schema.py, postgres_brains.py |
| **DOCS** | — | 3 file audit AI-generated → tách khỏi PR | — |

---

## C1 — asyncpg `IN ($2, $3)` invalid syntax

**Vấn đề:** asyncpg không hỗ trợ `IN ($2, $3)` với 2 tham số riêng. Phải dùng `= ANY($2::text[])` với list.

**File:** `src/neural_memory/storage/postgres/postgres_synapses.py`

**Vị trí 1 — `add_synapse` (dòng 48-54):**

```python
# CŨ (sai)
rows = await self._query_ro(
    "SELECT id FROM neurons WHERE brain_id = $1 AND id IN ($2, $3)",
    brain_id,
    synapse.source_id,
    synapse.target_id,
)

# MỚI (đúng)
rows = await self._query_ro(
    "SELECT id FROM neurons WHERE brain_id = $1 AND id = ANY($2::text[])",
    brain_id,
    [synapse.source_id, synapse.target_id],
)
```

**Vị trí 2 — `get_path` (dòng 237-244):**

```python
# CŨ
check = await self._query_ro(
    "SELECT id FROM neurons WHERE brain_id = $1 AND id IN ($2, $3)",
    brain_id,
    source_id,
    target_id,
)

# MỚI
check = await self._query_ro(
    "SELECT id FROM neurons WHERE brain_id = $1 AND id = ANY($2::text[])",
    brain_id,
    [source_id, target_id],
)
```

---

## C2 — 8+ abstract methods chưa implement

**Vấn đề:** `PostgreSQLStorage` kế thừa `NeuralStorage` nhưng thiếu các abstract method. Python sẽ raise `TypeError` khi khởi tạo.

**Các method cần implement:**

| Method | Reference implementation | File |
|--------|--------------------------|------|
| `get_fibers` | sqlite_fibers.py:324, falkordb_fibers.py:279 | postgres_fibers.py (thêm) |
| `export_brain` | sqlite_brain_ops.py:106, shared_store_collections.py:172 | postgres_brains.py (thêm) |
| `import_brain` | sqlite_brain_ops.py:149 | postgres_brains.py (thêm) |
| `get_stats` | sqlite_store.py:192 | postgres_brains.py hoặc postgres_store.py |
| `get_enhanced_stats` | sqlite_store.py:212 | postgres_brains.py hoặc postgres_store.py |
| `clear` | sqlite_store.py:310, falkordb_brains.py:474 | postgres_brains.py (thêm) |
| `get_all_neuron_states` | sqlite_neurons.py:424 | postgres_neurons.py (thêm) |
| `get_neuron_states_batch` | sqlite_neurons.py:333 | postgres_neurons.py (thêm) |

**Gợi ý:**
- Xem `shared_store_collections.py` và `sqlite_store.py` để nắm chữ ký và hành vi.
- `get_fibers`: trả về `list[Fiber]`, có `limit`, `order_by`, `descending`.
- `export_brain`/`import_brain`: dùng `BrainSnapshot`; export = SELECT brains + neurons + synapses + fibers + states, import = INSERT tương ứng.
- `get_stats`: `{neurons, synapses, fibers, ...}`.
- `get_enhanced_stats`: stats mở rộng (hot neurons, v.v.).
- `clear`: DELETE theo brain_id, thứ tự đúng để tránh FK.
- `get_all_neuron_states` / `get_neuron_states_batch`: đọc từ `neuron_states`.

---

## C3 — Thiếu tests

**Vấn đề:** Không có test cho PostgreSQL backend.

**Yêu cầu:**
- Tạo `tests/storage/postgres/` với CRUD tests cho neurons, synapses, fibers, brains.
- Dùng `pytest-asyncio` + `asyncpg`, test DB (VD: test instance hoặc temp DB).
- Có thể dùng fixture như SQLite tests: spin up Postgres (Docker/testcontainers) hoặc skip nếu không có Postgres.

**Gợi ý cấu trúc:**
```
tests/storage/postgres/
  conftest.py          # Fixture: PostgreSQLStorage, temp DB
  test_postgres_neurons.py
  test_postgres_synapses.py
  test_postgres_fibers.py
  test_postgres_brains.py
```

---

## H1 — `refractory_until` sai giá trị (data corruption)

**File:** `src/neural_memory/storage/postgres/postgres_neurons.py`  
**Hàm:** `update_neuron_state` (dòng 213-240)

**Vấn đề:** Tham số `$8` (refractory_until) đang nhận `state.last_activated` thay vì `state.refractory_until`.

**Fix:**

```python
# Dòng 230 — ĐỔI
state.last_activated,   # SAI
# THÀNH
state.refractory_until,  # ĐÚNG
```

Thứ tự đúng theo schema:
```
$1=neuron_id, $2=brain_id, $3=activation_level, $4=access_frequency,
$5=last_activated, $6=decay_rate, $7=firing_threshold,
$8=refractory_until,   # <-- phải là state.refractory_until
$9=refractory_period_ms, $10=homeostatic_target, $11=created_at
```

---

## H2 — Brain auto-create khi switch brain

**File:** `src/neural_memory/unified_config.py`  
**Hàm:** `_get_postgres_storage` (dòng 1514-1560)

**Vấn đề:** Khi `_postgres_storage` đã có, code chỉ gọi `set_brain(name)` rồi return. Không kiểm tra/tạo brain mới → FK violations khi switch sang brain chưa tồn tại.

**Fix:** Trong nhánh early-return, vẫn phải ensure brain tồn tại:

```python
if _postgres_storage is not None:
    _postgres_storage.set_brain(name)
    brain = await _postgres_storage.get_brain(name)
    if brain is None:
        brain = await _postgres_storage.find_brain_by_name(name)
    if brain is None:
        # Tạo brain mới (logic tương tự block dưới)
        ...
    return _postgres_storage
```

Copy logic create brain từ block phía dưới vào nhánh early-return.

---

## H3 — Exception bị nuốt ở `logger.debug`

**File:** `src/neural_memory/storage/postgres/postgres_neurons.py`  
**Hàm:** `find_neurons_by_embedding` (dòng 317-318)

**Vấn đề:** `except Exception` với `logger.debug` che mất lỗi (connection pool, pgvector extension, schema...).

**Fix:**

```python
# CŨ
except Exception:
    logger.debug("Embedding similarity search failed", exc_info=True)
    return []

# MỚI
except Exception:
    logger.error("Embedding similarity search failed", exc_info=True)
    return []
```

Theo project: luôn log lỗi thật sự ở mức `error`.

---

## H4 — Tags filter sau LIMIT

**File:** `src/neural_memory/storage/postgres/postgres_fibers.py`  
**Hàm:** `find_fibers` (dòng 77-118)

**Vấn đề:** `tags` đang filter trong Python sau khi đã `LIMIT` ở SQL → kết quả có thể thiếu so với mong đợi.

**Fix:** Đưa filter tags vào SQL bằng JSONB operator. VD:

```sql
-- tags ?& array['tag1','tag2'] = true khi fiber chứa tất cả tags
AND tags ?& $n::text[]
```

Thêm điều kiện vào `query` tương ứng, truyền `list(tags)` vào params, không filter trong Python sau `rows`.

---

## H5 — String `%s` trong SQL

**File:** `src/neural_memory/storage/postgres/postgres_synapses.py`  
**Hàm:** `get_neighbors` (dòng 187-193)

**Vấn đề:** Dùng `"$%s" % str(len(params) + 1)` để tạo placeholder. Dự án cấm string formatting trong SQL.

**Fix:** Build query mà không format placeholder trực tiếp. Ví dụ:

- Dùng số cố định hoặc build param list rõ ràng.
- Hoặc dùng helper build query với placeholder tuần tự ($1, $2, …) dựa trên `len(params)` mà không dùng `%`.

---

## MEDIUM

**M1 — `embedding vector(384)` hardcode**  
- File: `postgres_schema.py` dòng 35.  
- Fix: Lấy dimension từ config (embedding model), hoặc tham số schema; tránh hardcode 384.

**M2 — `row_to_fiber` tags fallback**  
- File: `postgres_row_mappers.py`.  
- Fix: So sánh logic tags với SQLite (`row_to_fiber` tương ứng) và đồng bộ.

**M3 — `unset_shared_storage`**  
- File: `unified_config.py`.  
- Fix: Gọi clear `_postgres_storage` trong `unset_shared_storage()` nếu có.

**M4 — `is_public INTEGER`**  
- File: `postgres_schema.py`, `postgres_brains.py`.  
- Fix: Dùng `BOOLEAN` thay cho `INTEGER` cho `is_public`.

---

## DOCS

- Các file `PERFORMANCE_AUDIT_REPORT.md`, `SECURITY_AUDIT_REPORT.md`, `SECURITY_AUDIT_POSTGRES_BACKEND.md` được xem là AI-generated và bao phủ toàn bộ codebase.
- Gợi ý: Tách khỏi PR #56; nếu cần, tạo issue riêng cho từng finding.

---

## Thứ tự thực hiện đề xuất

1. **C1** — Fix asyncpg syntax (nhanh, tránh crash).
2. **H1** — Fix refractory_until (tránh data corruption).
3. **C2** — Implement các abstract method (unblock khởi tạo storage).
4. **H2** — Fix brain auto-create khi switch brain.
5. **H3, H4, H5** — Fix logging, tags filter, placeholder.
6. **C3** — Thêm tests.
7. **M1–M4** — Cải thiện schema và consistency.

---

## Checklist trước khi re-request review

- [ ] C1 fixed (asyncpg `ANY`)
- [ ] C2 fixed (đủ abstract methods)
- [ ] C3 done (CRUD tests cho postgres)
- [ ] H1 fixed (refractory_until)
- [ ] H2 fixed (brain auto-create)
- [ ] H3 fixed (logger.error)
- [ ] H4 fixed (tags trong SQL)
- [ ] H5 fixed (không dùng % trong SQL)
- [ ] `pytest tests/storage/postgres/` pass
- [ ] `mypy src/` pass
- [ ] `ruff check src/` pass
