# PR #56 — Postgres Container Run: Issues Found & Fix Plan

> **Context:** Tạo container PostgreSQL + pgvector, chạy project, ghi nhận lỗi và plan fix.

---

## Đã làm

1. **docker-compose.postgres.yml** — PostgreSQL 16 + pgvector (port 5433)
2. **scripts/postgres-init.sh** — Tạo DB `neuralmemory`, `neuralmemory_test`
3. **tests/storage/conftest.py** — Bỏ FalkorDB skip cho tests trong `postgres/`
4. **Datetime fixes** — asyncpg cần `datetime`, không dùng ISO string; chuẩn hóa naive UTC khi đọc

---

## Lỗi đã xử lý

### 1. asyncpg: `expected datetime, got 'str'`

**Nguyên nhân:** Truyền `.isoformat()` thay vì object `datetime`.

**Đã sửa:** Dùng trực tiếp `datetime` thay cho `.isoformat()` ở tất cả INSERT/UPDATE trong:
- `postgres_brains.py` — save_brain, import_brain, get_enhanced_stats
- `postgres_neurons.py` — add_neuron, update_neuron_state, update_neuron_states_batch, find_neurons (time_range)
- `postgres_synapses.py` — add_synapse
- `postgres_fibers.py` — add_fiber, find_fibers (time_overlaps)

### 2. `TypeError: can't subtract offset-naive and offset-aware datetimes`

**Nguyên nhân:** TIMESTAMPTZ trả về timezone-aware, còn codebase dùng naive UTC.

**Đã sửa:** Thêm `_to_naive_utc()` trong `postgres_row_mappers.py`, áp dụng cho tất cả datetime khi map record → model.

### 3. FalkorDB skip cho Postgres tests

**Nguyên nhân:** `_require_falkordb` (autouse) trong `tests/storage/conftest.py` skip toàn bộ tests trong `storage/`, kể cả `postgres/`.

**Đã sửa:** Bỏ skip khi `"postgres" in str(request.node.fspath)`.

### 4. Typo: `SynapseType.RELATES_TO` → `RELATED_TO`

**Đã sửa:** Cập nhật `tests/storage/postgres/conftest.py`.

---

## Lỗi còn lại (chưa fix)

### 1. `add_typed_memory` / `find_typed_memories` / `get_expired_memories` chưa implement

**Ảnh hưởng:** `nmem remember`, `nmem stats` không chạy được với backend Postgres.

**Plan fix:**
- Tham khảo `sqlite_store.py` hoặc `memory_store.py` để implement:
  - `add_typed_memory`
  - `find_typed_memories`
  - `get_expired_memories`
  - `get_typed_memory`
  - `update_typed_memory`
- Typed memory lưu qua Fiber + metadata; cần mapping sang schema Postgres hiện có.

### 2. Init script không chạy khi volume đã có data

**Ảnh hưởng:** Lần chạy đầu tiên (volume trống) sẽ tạo DB, lần sau volume có data thì init không chạy.

**Giải pháp hiện tại:** Tạo thủ công:
```bash
docker exec neural-memory-postgres-1 createdb -U postgres neuralmemory neuralmemory_test
```

**Gợi ý:** Ghi rõ trong README hoặc docker-compose comment; hoặc dùng entrypoint script chạy `createdb` mỗi lần start (chỉ tạo nếu chưa tồn tại).

---

## Kết quả tests

- **22/22** Postgres storage tests pass.
- **nmem stats** fail do `find_typed_memories` chưa implement.

---

## Checklist cho senior dev

- [x] Docker Postgres + pgvector chạy được
- [x] Tests Postgres pass
- [x] Datetime: write dùng `datetime`, read chuẩn hóa naive UTC
- [ ] Implement typed memory APIs (`add_typed_memory`, `find_typed_memories`, …)
- [ ] Cập nhật init script/README cho môi trường dev/test
