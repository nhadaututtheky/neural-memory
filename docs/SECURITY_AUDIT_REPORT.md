# Security Audit Report — NeuralMemory

**Date:** 2025-03-12  
**Scope:** Full source code security review  
**Auditor:** Software Security Engineer Agent  

---

## Executive Summary

NeuralMemory follows solid security practices overall. The codebase adheres to documented security rules (CLAUDE.md) and implements defense-in-depth for SQL injection, path traversal, and input bounds. A few minor recommendations are provided for hardening.

| Category | Status | Notes |
|----------|--------|-------|
| SQL Injection | ✅ Pass | Parameterized queries, safe f-string usage |
| Path Traversal | ✅ Pass | `resolve()` + `is_relative_to()` validation |
| Error Exposure | ✅ Pass | Generic error messages, no stack traces to clients |
| Input Validation | ✅ Pass | Limits capped, Field validations |
| Network Security | ✅ Pass | Default 127.0.0.1, CORS safe defaults |
| Deserialization | ✅ Pass | No pickle/unsafe yaml |
| Subprocess | ⚠️ Low | Only safe patterns, Windows icacls documented |

---

## 1. SQL Injection

### Findings

- **Parameterized SQL:** All SQLite queries use `?` placeholders with `params` tuples. No user input is interpolated into query strings.
- **Dynamic placeholders:** Patterns like `placeholders = ",".join("?" for _ in ids)` build only placeholder counts from list length; values are passed in params — **safe**.
- **LIKE escaping:** `content_contains` is escaped for SQL LIKE wildcards before use (`replace("%","\\%"), replace("_","\\_")`) — **safe**.
- **FalkorDB/Cypher:** `max_hops` is capped with `min(max_hops, 10)` before injection into query — **safe**.

### Locations Checked

- `sqlite_store`, `sqlite_neurons`, `sqlite_synapses`, `sqlite_alerts`, `sqlite_maturation`, `sqlite_typed`, `sqlite_change_log`, etc.
- FalkorDB: `falkordb_graph.py`, `falkordb_neurons.py`, `falkordb_synapses.py`, `falkordb_fibers.py`

### Recommendation

No changes required. Maintain current practices.

---

## 2. Path Traversal / File Validation

### Findings

- **Validation pattern:** Paths are validated with `Path(...).resolve()` and `is_relative_to(allowed_root)` before file access.
- **Training handlers:** `mcp/db_train_handler.py` restricts DB paths to `cwd`, `home`, `temp`; rejects paths outside allowed roots.
- **Codebase / index:** `mcp/train_handler.py`, `mcp/index_handler.py` enforce `path.is_relative_to(cwd)`.
- **UnifiedConfig:** `get_brain_db_path()` enforces `db_path.is_relative_to(brains_dir)`.

### Locations

- `engine/db_introspector.py`, `engine/codebase_encoder.py`, `engine/doc_chunker.py`
- `cli/commands/train.py`, `cli/commands/codebase.py`, `cli/commands/brain.py`, `cli/commands/shortcuts.py`
- `mcp/db_train_handler.py`, `mcp/index_handler.py`, `mcp/train_handler.py`
- `server/routes/memory.py`, `hooks/stop.py`

### Recommendation

No changes required. Pattern is consistent across the codebase.

---

## 3. Error Handling & Information Disclosure

### Findings

- **MCP handlers:** Return generic messages like `"Tool X failed unexpectedly"`; full exceptions are logged via `logger.error(..., exc_info=True)`.
- **REST API:** Uses `HTTPException` with short, non-sensitive details (e.g., "Brain not found", "Path must be within working directory").
- **No stack traces or exception types** exposed to clients.
- **Brain names:** Not exposed in error responses per CLAUDE.md.

### Minor Note

- MCP JSON-RPC returns `"Resource not found: {uri}"` for resource requests — URI may be user-provided; low risk but could be generalized to `"Resource not found"` if desired.

### Recommendation

Consider normalizing resource-not-found messages to avoid any potential path leakage. Not critical.

---

## 4. Input Validation & Bounds

### Findings

- **Limit capping:** User limits are capped server-side: `min(user_limit, MAX)` in multiple modules.
- **Constants:** `MAX_CONTENT_LENGTH` (100,000), `MAX_BATCH_SIZE`, `MAX_BATCH_TOTAL_CHARS`, `MAX_RECALL_TAGS`, `_MAX_TAG_LENGTH`, etc.
- **Hub sync:** Pydantic `Field(max_length=...)` on `device_id`, `brain_id`, `changes` (capped at 1000).
- **Message size:** MCP `_MAX_MESSAGE_SIZE` 10 MB; WebSocket `_MAX_WS_MESSAGE_SIZE` 1 MB.

### Examples

- `sqlite_tool_events.py`: `safe_limit = min(limit, 10000)`
- `sqlite_change_log.py`: `safe_limit = min(limit, 10000)`
- `sqlite_cognitive.py`: `capped_limit = min(limit, 200)`
- `mcp/tool_handlers.py`: `limit = min(args.get("limit", 10), 200)`, `limit = min(args.get("limit", 20), 100)`

### Recommendation

No changes required. Bounds enforcement is consistent.

---

## 5. Network & Server Security

### Findings

- **Default host:** Config defaults to `127.0.0.1`, not `0.0.0.0`.
- **CLI:** `nmem serve --host 0.0.0.0` triggers a security warning (S104).
- **CORS:** Defaults to localhost origins; wildcard `["*"]` disables `allow_credentials`.
- **Trusted hosts:** `require_local_request` restricts API access to localhost or `NEURAL_MEMORY_TRUSTED_NETWORKS` CIDRs.
- **OAuth:** Uses `httpx` for token exchange; errors handled without exposing sensitive data.

### Recommendation

No changes required. Safe defaults and explicit opt-in for network exposure.

---

## 6. Subprocess & Deserialization

### Findings

- **Subprocess:** Only specific uses found:
  - `safety/encryption.py`: `icacls` on Windows (permission hardening), uses list args, no shell, `noqa: S603, S607` for linter.
  - `git_context.py`: `git` commands via list args, no shell.
  - `cli/setup.py`, `cli/commands/update.py`: `subprocess.run` with list args.
- **No `eval`/`exec`:** None found.
- **No unsafe deserialization:** No `pickle.load` or `yaml.load` (unsafe).

### Recommendation

Continue avoiding shell=True and list-arg subprocess calls. Current usage is acceptable.

---

## 7. Exception Handling

### Findings

- **No bare `except: pass`** in `src/`.
- Broad `except Exception` blocks log and return generic errors — acceptable for top-level handlers.
- Migration logic in `sqlite_schema.py` halts on non-benign errors; benign cases handled explicitly.

### Recommendation

No changes required.

---

## 8. Checklist vs CLAUDE.md Security Rules

| Rule | Status |
|------|--------|
| Parameterized SQL only | ✅ |
| Path validation: `resolve()` + `is_relative_to()` | ✅ |
| No internal errors to clients | ✅ |
| No brain names / stack traces in errors | ✅ |
| Bind to 127.0.0.1 by default | ✅ |
| CORS defaults to localhost, not `["*"]` | ✅ |
| Cap server-side limits | ✅ |
| Never bare `except: pass` | ✅ |
| MCP handlers: log before returning error dicts | ✅ |

---

## 9. Recommendations Summary

| Priority | Item | Action |
|----------|------|--------|
| Low | Resource-not-found URI in MCP | Optional: return generic `"Resource not found"` instead of including URI |
| — | Rest | No immediate changes needed |

---

## 10. Conclusion

NeuralMemory’s security posture is strong. The project consistently applies defensive coding for SQL, paths, limits, and error handling. Documented rules in CLAUDE.md are reflected in the implementation. No critical or high-severity issues were identified.
