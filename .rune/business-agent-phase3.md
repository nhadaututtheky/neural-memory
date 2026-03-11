# Phase 3: Data Import + Citation Deep Links

## Goal

Users can import documents (legal articles, accounting records) directly from the chat UI. Citations become first-class: clickable, expandable, with source provenance. The agent refuses to answer without citing sources.

## Tasks

### 3.1: DataImportModal Component
- Create `dashboard/src/features/business-agent/components/DataImportModal.tsx`
- Modal triggered by "Import Data" button in AgentPicker area
- Two import modes:
  - **Paste text**: textarea for pasting law articles or records
  - **File upload**: drag-and-drop zone accepting .txt, .md, .csv, .xlsx, .pdf
- Fields: source name (required), tags (comma-separated, optional)
- File upload: read file client-side, send content as string to POST /api/agent/import
- For CSV/XLSX: parse rows client-side, each row becomes a separate import call (batch)
- Progress indicator: "Importing... 15/50 records"
- Success: show neuron count created, close modal, refresh sources list
- Error: show inline error message, don't close modal

### 3.2: CSV/XLSX Client-Side Parser
- Add parsing logic in `dashboard/src/features/business-agent/utils/parsers.ts`
- CSV: simple split-based parser (no new deps) — handle quoted fields, newlines
- XLSX: use existing SheetJS if in deps, otherwise skip XLSX and only support CSV
- Each row -> formatted as "Column1: value1 | Column2: value2 | ..."
- Auto-detect headers from first row
- For accounting: expect columns like Date, Description, Amount, Category
- For legal: expect Article Number, Title, Content columns
- Return array of { content: string, source_name: string } ready for import

### 3.3: Source List Sidebar
- Create `dashboard/src/features/business-agent/components/SourceViewer.tsx`
- Collapsible right sidebar (or bottom panel on narrow screens)
- Shows imported sources grouped by source_name
- Each source: name, neuron count, import date, tags
- Click source -> expand to show first 5 neuron contents (previews)
- Delete source button (calls DELETE /api/agent/sources/:name — add backend route)
- Pull to refresh / auto-refresh after import

### 3.4: Backend Source Management
- Add to `src/neural_memory/server/routes/agent.py`:
  - `DELETE /api/agent/sources/{source_name}` — delete all neurons with metadata.source_name match
  - `GET /api/agent/sources/{source_name}/neurons` — list neurons for a source (paginated, limit 50)
- Source deletion: find neurons by tag + metadata.source_name, delete each + their synapses
- Validate source_name with safe path regex (no path traversal in metadata queries)

### 3.5: Enhanced Citation Rendering
- Update `CitationCard.tsx`:
  - Show full source chain: source_name -> fiber -> neuron content
  - For legal: highlight article number in bold, show clause hierarchy
  - For accounting: show record as key-value table (Date | Amount | Description)
  - "Copy citation" button -> copies formatted reference text
- Update `ChatMessage.tsx`:
  - Hover over [cite:N] badge -> tooltip with source name + snippet preview
  - Badge color: green for high-confidence match, yellow for moderate, gray for low
  - Confidence from query result's neuron activation level

### 3.6: Strict Citation Mode (Backend)
- Update system prompts in `agent.py`:
  - Legal agent: add "If no relevant source is found in the context, say 'I don't have that article in my knowledge base. Please import the relevant document.' Do NOT fabricate citations."
  - Accounting agent: add "Only report numbers that appear verbatim in sources. If unsure, say 'I need more data to answer accurately.'"
- Add `no_context_fallback` in `_call_llm()`: if `_retrieve_context()` returns 0 chunks, return a system message suggesting data import instead of calling LLM
- Log citation usage: which neurons were cited per conversation (for future analytics)

## Acceptance Criteria

- Click "Import Data" -> modal opens with paste/upload options
- Paste a law article -> neurons created, visible in source list
- Upload a CSV of transactions -> each row becomes a neuron, progress shown
- Ask agent about imported data -> response includes accurate citations
- Citations show source name, stored date, and exact content snippet
- Agent declines to answer when no relevant sources found (no hallucination)
- Delete a source -> neurons removed, no longer cited
- Dark mode compatible, no new npm deps (CSV parser is custom)

## Files Touched

### New Files
- `dashboard/src/features/business-agent/components/DataImportModal.tsx`
- `dashboard/src/features/business-agent/components/SourceViewer.tsx`
- `dashboard/src/features/business-agent/utils/parsers.ts`

### Modified Files
- `dashboard/src/features/business-agent/BusinessAgentPage.tsx` (add import button + source sidebar)
- `dashboard/src/features/business-agent/components/AgentPicker.tsx` (import button)
- `dashboard/src/features/business-agent/components/CitationCard.tsx` (enhanced rendering)
- `dashboard/src/features/business-agent/components/ChatMessage.tsx` (tooltip, confidence color)
- `dashboard/src/features/business-agent/api/agent.ts` (delete source, list neurons)
- `src/neural_memory/server/routes/agent.py` (DELETE + GET neurons endpoints, strict citation)
- `dashboard/src/i18n/en.json` (import modal strings)
- `dashboard/src/i18n/vi.json` (import modal strings)

## Dependencies

- Phase 1 (backend routes) and Phase 2 (chat UI) must be complete
- Existing NM encode/query pipeline handles the actual storage and retrieval
