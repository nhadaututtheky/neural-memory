# Architecture Analysis: NeuralMemory KB Retrieval Problems

**Date:** 2026-03-02 | **Type:** Senior Architecture Consultation
**Detailed reports:** `researcher-260302-0506-cross-language-retrieval.md`, `researcher-260302-0506-agent-recall-relevance.md`, `researcher-260302-0506-pdf-extraction-solutions.md`

---

## TL;DR

| # | Problem | Root Cause | Best Solution | Effort | Long-term fit |
|---|---------|-----------|---------------|--------|---------------|
| 1 | Cross-language recall fail | FTS5 = keyword-only, no embeddings | **BGE-M3 + RRF hybrid retrieval** | 2 weeks | YES — core upgrade |
| 2 | Agent ignores recall results | LLM lexical bias (language ≠ relevance) | **Pre-translate answer to query language** + metadata hints | 1 week | YES — MCP-level fix |
| 3 | PDF diagram/schematic garbage | MarkItDown can't handle visual content | **Docling (IBM)** for text + **Vision LLM** for diagrams | 1-2 weeks | PARTIAL — diagrams need VLM |

**Total estimated effort:** 4-5 weeks sequential, 2-3 weeks parallel.
**Total recurring cost:** $0-15/mo (all local-first solutions).

---

## Problem 1: Cross-Language Recall Failure

### Root Cause (Brutal Truth)

FTS5 with Porter stemmer is **English-only by design**. Vietnamese has no word boundaries (agglutinative), so "mức nhớt" is tokenized as garbage. NeuralMemory has embedding infrastructure but uses `all-MiniLM-L6-v2` — a **monolingual English model**. Embeddings exist but don't help cross-lingually.

### Architecture Decision

**Replace keyword-only retrieval with hybrid dense-sparse.**

```
                    CURRENT                              PROPOSED
                    ───────                              ────────
Query ──→ FTS5 ──→ results              Query ──→ ┌─ FTS5 (keyword) ──────┐
                                                   ├─ BGE-M3 dense (embed) ─┤──→ RRF merge ──→ results
                                                   └─ BGE-M3 sparse ────────┘
```

### Recommended Stack

| Component | Choice | Why | Cost |
|-----------|--------|-----|------|
| Embedding model | **BGE-M3** (BAAI) | 111 languages, Vi-En optimized, dense+sparse in one model, local | $0 (1.4GB download) |
| Vector storage | **sqlite-vec** | SQLite extension, SIMD-accelerated, no new deps | $0 |
| Merge strategy | **RRF** (Reciprocal Rank Fusion) | Simple, proven, 50 lines of code | $0 |
| Fallback | Query-time translation (MyMemory API) | Free tier, catches FTS5-only queries | $0 |

### Why BGE-M3 over alternatives

- **multilingual-e5**: Good but single-mode (dense only). BGE-M3 does dense + sparse natively.
- **OpenAI embeddings**: Good quality but API dependency + cost ($0.02/M tokens). Violates local-first principle.
- **Cohere multilingual**: Best API quality but $0.10/M tokens recurring cost.
- **ColBERT/late interaction**: Overkill — neurons are short text, not long docs.

### Performance Impact

| Method | Vi→En Quality | Latency | Cost/query |
|--------|--------------|---------|------------|
| FTS5 current | **FAIL** | 5ms | $0 |
| FTS5 + translation | 85% | 35ms | $0 |
| BGE-M3 + RRF | **95%+** | 150ms | $0 |
| OpenAI + RRF | 90% | 70ms | $0.00002 |

### Implementation Path

**Phase 1 (Week 1):** Quick wins
1. Add query-time translation fallback in `retrieval.py` — 2hrs
2. Implement RRF combiner — 1hr
3. Config: `embedding_model = "BAAI/bge-m3"` — 5min

**Phase 2 (Week 2):** Production-grade
1. sqlite-vec integration for vector storage — 5hrs
2. Batch re-embed existing KB neurons with BGE-M3 — 2hrs
3. Cross-encoder re-ranking (optional, +5% precision) — 3hrs

### Long-term Applicability: **YES**

This is the correct architectural direction. The hybrid retrieval pattern (FTS5 anchor + embedding similarity + spreading activation → RRF merge) proposed in the roadmap is exactly what the industry uses. NeuralMemory's unique spreading activation layer becomes even more valuable as a **third signal** in the fusion.

```
Proposed 3-signal fusion:
  Signal 1: FTS5 keyword (precision on exact terms)
  Signal 2: BGE-M3 embedding (semantic cross-lingual)
  Signal 3: Spreading activation (graph-based associative)
  ──→ RRF merge ──→ final ranked results
```

---

## Problem 2: Agent Ignores Recall Results

### Root Cause (Brutal Truth)

This is NOT a retrieval problem. Retrieval succeeds — semantically correct English content IS returned. The problem is **LLM reasoning bias**: Gemini Flash sees English content + Vietnamese query → concludes "not relevant" because **language mismatch triggers lexical bias** in relevance judgment.

Research (arxiv 2511.09984) calls this "decoder-level collapse" — smaller models conflate language matching with semantic relevance.

### Key Constraint

NeuralMemory is an MCP server. **Cannot control what the agent does with results.** Can only control output format/metadata.

### Architecture Decision

**Two-pronged approach:**

1. **Pre-translate answer to query language** (eliminates language mismatch entirely)
2. **Metadata hints** (helps smarter agents interpret cross-language results)

### Solution Design

```python
# In MCP recall handler, AFTER retrieval, BEFORE returning:

async def _format_recall_output(self, result, query):
    query_lang = detect(query)  # "vi", "en", etc.
    content_lang = detect(result.context)  # "en"

    if query_lang != content_lang:
        # Strategy 1: Translate answer to query language
        translated = await self._translate(result.answer, content_lang, query_lang)

        # Strategy 2: Add explicit metadata hints
        result = result.with_metadata({
            "query_language": query_lang,
            "content_language": content_lang,
            "semantic_similarity": 0.92,
            "note": f"Content is in {content_lang} but semantically matches "
                    f"query with {result.confidence:.0%} confidence. "
                    f"Translated summary provided.",
            "translated_answer": translated,
        })
    return result
```

### Why Pre-Translation Wins

| Approach | Effectiveness | Agent-side changes needed | Cost |
|----------|-------------|--------------------------|------|
| Metadata hints only | 20-40% | Agent prompt must read metadata | $0 |
| Score transparency | 30-50% | Agent must interpret score breakdown | $0 |
| **Pre-translate answer** | **80%+** | **None** — agent sees Vietnamese answer | $0-15/mo |
| Separate judgment tool | 40-60% | Agent must choose correct tool | $0 |

Pre-translation is the only solution that works **without agent cooperation**. The agent sees Vietnamese answer + Vietnamese query → no language mismatch → no bias.

### Translation Options

For NeuralMemory's scale (MCP server, individual user):
- **Free tier: MyMemory API** — 10K chars/day free, good enough for recall answers
- **Production: Google Translate** — $15/1M chars, ~$2-5/mo typical
- **Local: Helsinki-NLP/opus-mt** — 200MB model, 0 cost, offline, ~85% quality

**Recommendation:** Start with MyMemory free. Switch to local opus-mt if latency matters.

### Long-term Applicability: **YES**

The auto-translate layer in the retrieval pipeline (detect query language → translate to KB language → recall → translate result back) is the correct architecture. This is exactly how Google Search, Wikipedia, and enterprise multilingual search work.

Implementation should be:
1. **Query translation** (for FTS5 keyword matching) — Phase 1
2. **Answer translation** (for LLM consumption) — Phase 1
3. **Embedding-level** cross-language (BGE-M3) — Phase 2 (eliminates need for query translation)

After BGE-M3, only answer translation remains necessary.

---

## Problem 3: PDF Diagram/Schematic Extraction

### Root Cause (Brutal Truth)

MarkItDown + PyMuPDF extract **text streams from PDF objects**. Wiring diagrams are **vector graphics or raster images** — no text objects to extract. Custom fonts use **private Unicode mappings** (Type3/CIDFont with custom CMap) → text extraction returns garbage because font→Unicode mapping is wrong.

Two distinct sub-problems:
1. **Custom fonts → garbled text** (fixable with OCR fallback)
2. **Diagrams/schematics → no text at all** (requires visual understanding)

### Architecture Decision

**Replace MarkItDown with Docling for text + tables. Add VLM pipeline for diagrams.**

### Tool Comparison

| Tool | Text Quality | Tables | Diagrams | Custom Fonts | Speed | License | Dep Weight |
|------|-------------|--------|----------|--------------|-------|---------|-----------|
| MarkItDown (current) | Good | Poor | FAIL | FAIL | Fast | MIT | Light |
| **Docling (IBM)** | **Excellent** | **Excellent** | Partial | **OCR fallback** | Medium | MIT | Medium |
| Marker | Excellent | Good | FAIL | Good (OCR) | Fast | Rail-M* | Heavy (torch) |
| Nougat (Meta) | Good (academic) | Good | FAIL | N/A | Slow | CC-BY-NC | Heavy (GPU) |
| Surya OCR | Good | Coming | FAIL | OCR native | Medium | GPL-3 | Medium |

*Marker license restricts commercial use >$2M funding.

### Recommended Stack

**Tier 1: Docling (replace MarkItDown)**
- MIT license, 54K+ GitHub stars, IBM-backed
- AI-powered layout analysis (DocLayNet) + table structure (TableFormer)
- Built-in OCR fallback for custom fonts
- Runs locally on commodity hardware
- Python-native: `pip install docling`
- Granite-Docling-258M VLM for enhanced understanding (Apache 2.0)

```python
# Current:
from markitdown import MarkItDown
md = MarkItDown()
result = md.convert(pdf_path)

# Proposed:
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
doc = converter.convert(pdf_path)
markdown = doc.document.export_to_markdown()
```

**Tier 2: Vision LLM for diagrams (supplement)**

Diagrams cannot be extracted as text. Period. The only viable approach:
1. Detect diagram pages (Docling can classify page elements)
2. Render page as image
3. Send to Vision LLM for text description
4. Store description as neuron content

```python
# Diagram extraction pipeline:
async def extract_diagram(page_image: bytes) -> str:
    """Use Vision LLM to describe a diagram."""
    # Option A: Local - Granite-Docling-258M (free, 258MB)
    # Option B: API - Claude Vision / Gemini Pro Vision ($0.001/image)
    # Option C: Local - LLaVA 7B (free, 4GB, slower)
    return await vision_llm.describe(page_image,
        prompt="Describe this technical diagram in detail. "
               "Include all labels, connections, and specifications.")
```

### Custom Font Solution

```
PDF with custom fonts
  ├─ Step 1: Try text extraction (Docling)
  ├─ Step 2: Detect garbage (heuristic: >30% non-printable chars)
  └─ Step 3: If garbage → OCR fallback (Docling's EasyOCR or Surya)
```

Docling handles this automatically via its pipeline — it detects non-standard font encoding and falls back to OCR.

### Chunking Enhancement

Current NeuralMemory chunking (`doc_chunker.py`) already does section-aware markdown chunking with heading hierarchy. Docling's structured output preserves:
- Page numbers
- Section hierarchy
- Table structure (as markdown tables)
- Figure captions and references

This maps directly to NeuralMemory's `DocChunk(heading_path=...)` format.

### Long-term Applicability: **PARTIAL**

- **Text + Tables:** Docling is the right long-term solution. Actively maintained by IBM, MIT license, growing ecosystem.
- **Diagrams:** No perfect local solution exists. Vision LLMs are the state-of-the-art but require either API cost or local GPU. This is an industry-wide unsolved problem.
- **Realistic expectation:** For technical manuals like KTM 790 ADV, expect ~85-90% content extraction quality (text + tables excellent, diagrams = text descriptions only, not visual reproduction).

---

## Cross-Cutting: PostgreSQL Migration

**Roadmap item:** PostgreSQL backend when KB >5GB.

### Assessment

- Storage layer is already isolated (~4K LOC, `NeuralStorage` interface)
- sqlite-vec → pgvector is a natural migration path
- FTS5 → PostgreSQL tsvector is straightforward
- Spreading activation queries would benefit from PostgreSQL's recursive CTE optimization

**Recommendation:** Don't migrate yet. SQLite + sqlite-vec handles up to ~500K neurons (est. 5-10GB with embeddings) comfortably. Migration justified only when:
1. Multi-user concurrent access needed (SQLite = single writer)
2. >500K neurons (ANN index needed → pgvector's HNSW)
3. Real-time sync across multiple MCP server instances

**Effort when needed:** 2-3 weeks (interface already isolated).

---

## Strategic Implementation Roadmap

### Sprint 1: Cross-Language Foundation (Week 1-2)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| BGE-M3 model swap | `brain.py` config | 30min | Enables cross-lingual embeddings |
| RRF merge combiner | `retrieval.py` | 2hrs | Fuses keyword + vector results |
| Query language detection | `tool_handlers.py` | 1hr | Enables translation routing |
| Answer pre-translation | `tool_handlers.py` | 3hrs | Fixes agent language bias |
| sqlite-vec integration | `sqlite_schema.py`, `retrieval.py` | 5hrs | 10x vector search speedup |

### Sprint 2: PDF Extraction Upgrade (Week 2-3)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| Replace MarkItDown with Docling | `doc_trainer.py`, `doc_chunker.py` | 8hrs | Better tables, OCR fallback |
| Custom font detection + OCR fallback | `doc_trainer.py` | 3hrs | Fixes garbled text |
| Diagram detection + VLM description | `doc_trainer.py` (new pipeline) | 8hrs | Partial diagram extraction |

### Sprint 3: Integration & Testing (Week 3-4)

| Task | Files | Effort | Impact |
|------|-------|--------|--------|
| Re-embed existing KB with BGE-M3 | migration script | 2hrs | Backfill vectors |
| Cross-language recall tests | new test file | 4hrs | Validation |
| Re-train KTM manual with Docling | manual test | 2hrs | Validation |
| Metadata hints in MCP output | `tool_handlers.py` | 2hrs | Better agent interpretation |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BGE-M3 too large for CI (1.4GB) | Medium | Low | Mock in tests, download in CI cache |
| sqlite-vec not available on all platforms | Low | Medium | Fallback to Python cosine (current behavior) |
| Docling heavy dependency tree | Medium | Medium | Optional import, keep MarkItDown as fallback |
| Translation API rate limits | Low | Low | Cache translations, use local opus-mt |
| VLM diagram descriptions inaccurate | High | Medium | Mark as "AI-described", include confidence |

---

## Unresolved Questions

1. **Hardware:** Is GPU available for local inference? Affects BGE-M3 speed (CPU: 150ms, GPU: 30ms).
2. **KB scale:** How many neurons expected? Determines sqlite-vec vs pgvector timing.
3. **Diagram fidelity:** Is "text description of diagram" acceptable, or do users need visual reproduction?
4. **Agent control:** Can MCP server influence agent's system prompt? (Affects metadata hint ROI)
5. **Cost tolerance:** $0/mo strict, or $15/mo acceptable for translation API?
