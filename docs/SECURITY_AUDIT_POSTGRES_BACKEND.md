# Security Audit Report — PostgreSQL + pgvector Backend (New Code)

**Date:** 2025-03-12  
**Scope:** Source code mới thêm: `storage/postgres/`, `unified_config.py` (PostgresConfig, _get_postgres_storage)  
**Auditor:** Software Security Engineer Agent  

---

## Executive Summary

Backend PostgreSQL + pgvector tuân thủ khá tốt các quy tắc bảo mật. Mọi truy vấn dùng tham số hóa; không có SQL injection rõ ràng. Có vài điểm cần cải thiện về xử lý ngoại lệ và validation.

| Category | Status | Notes |
|----------|--------|-------|
| SQL Injection | ✅ Pass | Parameterized queries ($1, $2, $3) |
| Credential Handling | ✅ Pass | Password ưu tiên env, không lưu plaintext trong TOML |
| Exception Swallowing | ⚠️ Medium | find_neurons_by_embedding catch-all |
| Error Disclosure | ⚠️ Low | Một số ValueError trả ID ra client |
| Input Validation | ✅ Pass | Limit đã được cap |
| JSON Parsing | ✅ Pass | json.loads trên dữ liệu DB (trusted) |

---

## 1. SQL Injection

### Findings

- **Parameterized queries:** Tất cả truy vấn dùng asyncpg placeholders `$1`, `$2`, … với arguments.
- **Không có f-string vào SQL:** Các f-string chỉ dùng cho `len(params)` để sinh placeholder index, không ghép dữ liệu user vào câu SQL.
- **plainto_tsquery:** `content_contains` được truyền qua tham số, không ghép trực tiếp vào query.

### Ví dụ kiểm tra

```python
# postgres_neurons.py L117
query += f" AND content_tsv @@ plainto_tsquery('english', ${idx})"
params.append(safe_term)  # safe_term chỉ replace & | bằng space
```

`content_contains` được làm sạch `replace("&", " ").replace("|", " ")` trước khi truyền vào `plainto_tsquery` — an toàn.

### Recommendation

Tiếp tục dùng parameterized queries, không nên ghép chuỗi user trực tiếp vào SQL.

---

## 2. Credential & Connection

### Findings

- **Password:** `PostgresConfig` ưu tiên env var `NEURAL_MEMORY_POSTGRES_PASSWORD` hơn config file.
- **TOML save:** Không ghi password thật vào TOML (`password = ""` + comment dùng env).
- **Connection:** Dùng asyncpg pool, `command_timeout=60`; không thấy SSL mặc định.

### Recommendations

| Priority | Item | Action |
|----------|------|--------|
| Low | SSL/TLS | Cân nhắc thêm `ssl=require` cho production |
| — | — | Đảm bảo `NEURAL_MEMORY_POSTGRES_PASSWORD` được set khi chạy production |

---

## 3. Exception Handling

### Findings

**1. `find_neurons_by_embedding` — catch-all exception (Medium)**

```python
# postgres_neurons.py L270-271
except Exception:
    return []
```

- Bắt mọi exception và trả `[]`.
- Không log → khó debug.
- Có thể che lỗi nghiêm trọng (connection, schema, pgvector chưa cài).

**2. `add_synapse` / `add_fiber` — broad exception check**

```python
except Exception as e:
    if "unique" in str(e).lower() or "duplicate" in str(e).lower():
        raise ValueError(...) from e
    raise
```

- Chỉ kiểm tra chuỗi; có thể nhầm với exception khác chứa "unique"/"duplicate".
- Nên bắt cụ thể `asyncpg.UniqueViolationError` nếu có thể.

### Recommendations

| Priority | Item | Action |
|----------|------|--------|
| Medium | find_neurons_by_embedding | Thêm `logger.debug("Embedding search failed", exc_info=True)` trước `return []` |
| Low | add_synapse/add_fiber | Dùng `asyncpg.UniqueViolationError` thay vì kiểm tra `str(e)` |

---

## 4. Error Disclosure

### Findings

- Một số `ValueError` chứa `neuron.id`, `synapse.id`:
  - `"Neuron {neuron.id} does not exist"`
  - `"Synapse {synapse.id} already exists"`

Đây là ID nội bộ, có thể chấp nhận cho debugging, nhưng nếu dùng qua API cần xem xét không để lộ thông tin nhạy cảm.

- Không thấy stack trace hay exception type trả ra client.

### Recommendation

Không cần sửa gấp; nếu API expose các lỗi này, có thể chuyển sang message chung như "Resource not found".

---

## 5. Input Validation & Bounds

### Findings

- **Limit:** Đều cap: `min(limit, 1000)`, `min(limit, 20)`, `min(limit, 100)`.
- **Batch:** `get_neurons_batch`, `_get_synapses_batch` dùng `ANY($2::text[])` — kiểm soát qua tham số, không ghép chuỗi.
- **get_path:** `max_hops` có thể tùy API; nên có cap phía backend (ví dụ ≤ 10).

### Recommendation

Nếu `get_path` nhận `max_hops` từ user, nên thêm: `max_hops = min(max_hops, 10)`.

---

## 6. JSON & Deserialization

### Findings

- **json.loads:** Dùng trong `postgres_row_mappers.py` cho dữ liệu từ DB (metadata, config, shared_with, …).
- **Nguồn:** Dữ liệu do ứng dụng ghi, không phải user raw.
- **Rủi ro:** Thấp; nếu DB bị compromise thì đây không phải vector chính.

### Recommendation

Giữ nguyên; không cần thay đổi logic hiện tại.

---

## 7. Schema & Migration

### Findings

- **ensure_schema:** Chỉ bắt `DuplicateObjectError`; exception khác sẽ propagate.
- **CREATE EXTENSION vector:** Nếu pgvector chưa cài, lỗi sẽ rõ ràng.
- **Không lưu mật khẩu trong schema SQL:** Schema không chứa credential.

### Recommendation

Không cần sửa; đảm bảo pgvector được cài đúng trước khi dùng backend postgres.

---

## 8. PostgresConfig (unified_config)

### Findings

- **Validation:** `port` được giới hạn 1–65535; string được truncate 128/256 ký tự.
- **Password:** Ưu tiên env, không ghi ra TOML.
- **from_dict:** Không có logic rõ ràng cảnh báo khi dùng password từ config (khác với FalkorDB có warning).

### Recommendation

Có thể thêm warning tương tự FalkorDB khi đọc password từ config thay vì env:

```python
if not password_env and data.get("password"):
    logger.warning("PostgreSQL password from config — prefer NEURAL_MEMORY_POSTGRES_PASSWORD env")
```

---

## 9. Checklist vs CLAUDE.md Security Rules

| Rule | Status |
|------|--------|
| Parameterized SQL only | ✅ |
| Never expose internal errors/stack traces | ✅ |
| Cap server-side limits | ✅ |
| Validate paths (N/A for postgres backend) | — |
| No bare `except: pass` | ⚠️ `except Exception: return []` (nên log) |

---

## 10. Recommendations Summary

| Priority | Item | Location |
|----------|------|----------|
| Medium | Log exception trong find_neurons_by_embedding | postgres_neurons.py L270 |
| Low | Cảnh báo khi dùng password từ config | unified_config.py PostgresConfig.from_dict |
| Low | Bắt UniqueViolationError trong add_synapse/add_fiber | postgres_synapses.py, postgres_fibers.py |
| Low | Cap max_hops trong get_path nếu nhận từ user | postgres_synapses.py |

---

## 11. Conclusion

Backend PostgreSQL + pgvector không có lỗ hổng nghiêm trọng; SQL injection được kiểm soát, credential được xử lý cẩn thận. Ưu tiên cải thiện:

1. Log exception trong `find_neurons_by_embedding`.
2. Cảnh báo khi dùng password từ config.
3. Dùng exception cụ thể thay vì kiểm tra `str(e)` trong các thao tác add.
