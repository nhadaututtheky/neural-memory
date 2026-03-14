# Phase 2: Citation Tool (`nmem_cite`)

## Goal
New MCP tool that takes neuron IDs and returns exact quoted text from source documents, with staleness detection. Separate from recall to keep recall fast.

## Tasks
- [ ] 2.1 Define `SourceResolver` protocol in `engine/source_resolver.py`
- [ ] 2.2 Implement `LocalResolver` (read file span by char offset or page+line fallback)
- [ ] 2.3 Add path validation security (resolve + is_relative_to base_path)
- [ ] 2.4 Implement staleness detection (compare current file hash vs stored source_hash)
- [ ] 2.5 Create `cite_handler.py` MCP handler with `nmem_cite` tool
- [ ] 2.6 Register tool schema in `tool_schemas.py` and `server.py`
- [ ] 2.7 Add REST endpoint `/api/cite` in FastAPI app
- [ ] 2.8 Write tests: valid citation, stale source, missing file, invalid path, no locator
- [ ] 2.9 Update tool count in test_mcp.py + test_tool_tiers.py

## nmem_cite Interface
```
Input:  nmem_cite(neuron_ids=["abc", "def"], format="inline"|"footnote"|"full")
Output:
{
  "citations": [
    {
      "neuron_id": "abc",
      "neuron_summary": "Sa thải cần báo trước 30 ngày...",
      "source_name": "luat-lao-dong-2019.pdf",
      "source_type": "law",
      "page": 45,
      "line_range": [145, 162],
      "exact_text": "Điều 36. Quyền đơn phương chấm dứt...",
      "citation_formatted": "[BLDS 2019, Điều 36]",
      "source_stale": false,
      "source_id": "src_xxx"
    }
  ],
  "stale_count": 0,
  "missing_count": 0
}
```

## SourceResolver Protocol
```python
class SourceResolver(Protocol):
    async def read_span(self, source: Source, locator: dict[str, Any]) -> str: ...
    async def file_hash(self, source: Source) -> str: ...
    async def file_exists(self, source: Source) -> bool: ...
```

## Acceptance Criteria
- [ ] `nmem_cite` returns exact text from source file at correct location
- [ ] Stale sources detected and flagged (source_stale=true)
- [ ] Missing files handled gracefully (error message, not crash)
- [ ] Path traversal attacks blocked (security validation)
- [ ] Works with PDF (page-based), MD/TXT (line-based), DOCX (heading-based)
- [ ] Existing citation system (in recall) still works unchanged

## Files Touched
- `src/neural_memory/engine/source_resolver.py` — NEW: protocol + LocalResolver
- `src/neural_memory/mcp/cite_handler.py` — NEW: nmem_cite handler
- `src/neural_memory/mcp/tool_schemas.py` — add nmem_cite schema
- `src/neural_memory/mcp/server.py` — register cite handler
- `src/neural_memory/server/app.py` — add /api/cite endpoint
- `tests/unit/test_cite_handler.py` — NEW: citation tests
- `tests/unit/test_source_resolver.py` — NEW: resolver tests

## Dependencies
- Requires Phase 1 (source locators in neurons)
