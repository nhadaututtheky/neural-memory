# Phase 1: Source Locators in Training

## Goal
Enrich the training pipeline so every trained neuron knows exactly WHERE its content came from — file, page, line, character offset. Auto-create Source records and SOURCE_OF synapses during training.

## Tasks
- [ ] 1.1 Add `page_number`, `char_offset_start`, `char_offset_end` fields to `DocChunk`
- [ ] 1.2 Modify PDF extraction to emit page markers (`<!-- PAGE:N -->`) in markdown output
- [ ] 1.3 Modify `chunk_markdown()` to parse page markers and populate DocChunk page fields
- [ ] 1.4 Compute char offsets during chunking (byte position in original extracted text)
- [ ] 1.5 Add `source_locator` dict to neuron metadata during `_encode_chunks()`
- [ ] 1.6 Auto-create `Source` record per file in `train_file()` (using existing `Source.create()`)
- [ ] 1.7 Create `SOURCE_OF` synapses from Source → each chunk's anchor neuron
- [ ] 1.8 Store `source_hash` (SHA-256) in neuron metadata for staleness detection
- [ ] 1.9 Write tests: locators survive round-trip, SOURCE_OF count matches chunks, page numbers correct for PDF
- [ ] 1.10 Update `nmem_train` return value to include `sources_created` count

## source_locator Schema (neuron.metadata)
```json
{
  "source_locator": {
    "page": 12,
    "line_start": 145,
    "line_end": 162,
    "char_offset_start": 4520,
    "char_offset_end": 5100,
    "section_id": "3.2.1"
  },
  "source_hash": "sha256:abc123..."
}
```

## Acceptance Criteria
- [ ] Training a PDF produces neurons with correct `page_number` in source_locator
- [ ] Training any file auto-creates a Source record (no manual `nmem_source register`)
- [ ] Each chunk neuron has a SOURCE_OF synapse back to its Source
- [ ] `source_hash` matches SHA-256 of file at training time
- [ ] Existing training behavior unchanged (backward compatible)
- [ ] Line-based locators still work for non-PDF files (md, txt, docx)

## Files Touched
- `src/neural_memory/engine/doc_chunker.py` — add page/offset fields to DocChunk
- `src/neural_memory/engine/doc_extractor.py` — emit page markers for PDF
- `src/neural_memory/engine/doc_trainer.py` — auto-create Source, SOURCE_OF, store locators
- `tests/unit/test_doc_trainer.py` — new tests for locators + Source creation
- `tests/unit/test_doc_chunker.py` — new tests for page marker parsing

## Dependencies
- None — this is the foundation phase

## Technical Notes
- PDF page markers: pymupdf4llm `to_markdown()` doesn't emit page breaks by default. Use `page_chunks=True` parameter or iterate pages manually with `page.get_text("text")` to preserve boundaries
- Char offsets: compute from cumulative position in the full extracted text before chunking
- Source dedup: check `find_source_by_name(filename)` before creating — reuse existing Source if same file_hash
- Don't break existing tests: all new fields are Optional with None defaults
