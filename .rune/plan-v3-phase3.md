# Phase 3: Structured Encoding — Schema-Aware Encoder

## Goal

Neurons can store tabular and key-value data with structure preserved. The encoder detects structured input (CSV rows, JSON objects, key-value pairs) and stores both raw content and a structured representation for precise recall.

## Tasks

- [ ] 3.1: Structured content detection
  - New module `src/neural_memory/extraction/structure_detector.py`
  - Detect: CSV row, JSON object, key-value pairs, table row
  - Return `StructuredContent(format, fields, raw)` dataclass
  - Heuristics only (no LLM): delimiter detection, brace matching, `key: value` patterns
  - Called by encoder before embedding

- [ ] 3.2: Schema-aware neuron metadata
  - Add `structure` field to neuron metadata JSON: `{format, fields: [{name, type, value}]}`
  - `format`: "csv_row", "json_object", "key_value", "table_row", "plain"
  - Field types auto-detected: "number", "date", "currency", "text"
  - Backward compatible: existing neurons have `structure: null`

- [ ] 3.3: Encoder integration
  - In `MemoryEncoder.encode()`, run structure detection before embedding
  - If structured → store `structure` in metadata, store raw content as-is (no summarize)
  - If `source_id` provided → attach SOURCE_OF synapse (from Phase 2)
  - Structured neurons get tag `_structured:{format}` for filtering

- [ ] 3.4: Structured recall formatting
  - In retrieval pipeline, detect `structure` in metadata
  - Format output: key-value → `Key: Value` lines, CSV → aligned columns
  - `mode="exact"` (from Phase 1) returns raw content unchanged
  - Normal mode returns formatted structure + summary

- [ ] 3.5: Batch structured import
  - Enhance `nmem_train` to detect CSV/XLSX with structured rows
  - Each row → one neuron with structure metadata
  - Headers → field names, auto-detect field types
  - Link all rows to same source (if source_id provided)

- [ ] 3.6: Tests
  - Structure detection for all formats (CSV, JSON, KV, table, plain)
  - Encoder stores structure metadata correctly
  - Recall formats structured data properly
  - Batch import with structure preservation
  - Round-trip: import CSV → recall exact row → matches original

## Acceptance Criteria

- [ ] `nmem_remember(content="Date: 2026-01-15 | Amount: 25,000,000 VND | Payee: Nguyen Van A")` stores with `structure.format = "key_value"`
- [ ] `nmem_recall(query="payment Nguyen Van A")` returns formatted key-value output
- [ ] `nmem_recall(query="payment", mode="exact")` returns raw content unchanged
- [ ] CSV training: 100 rows → 100 structured neurons, each with field metadata
- [ ] Existing plain-text neurons unaffected (backward compatible)

## Files Touched

### New
- `src/neural_memory/extraction/structure_detector.py`
- `tests/unit/test_structured_encoding.py`

### Modified
- `src/neural_memory/engine/encoder.py` — structure detection call
- `src/neural_memory/engine/retrieval.py` — structured output formatting
- `src/neural_memory/engine/training.py` — batch structured import
- `src/neural_memory/core/memory_types.py` — StructuredContent dataclass (if needed)

## Dependencies

- Phase 1 complete (exact recall mode available)
- Phase 2 complete (source_id linkage available)
