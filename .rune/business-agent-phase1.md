# Phase 1: Backend Chat API + NM Integration

## Goal

Create the FastAPI backend that receives a user message, retrieves relevant context from Neural Memory, calls Claude API with that context, and streams the response back. No frontend yet — testable via curl/httpie.

## Tasks

### 1.1: Agent Route Scaffold + Models
- Create `src/neural_memory/server/routes/agent.py` with `agent_router`
- Register router in `src/neural_memory/server/routes/__init__.py` and `app.py`
- Define Pydantic models in `src/neural_memory/server/models.py`:
  - `ChatRequest`: message (str), agent_type ("legal" | "accounting"), brain (str | None), conversation_history (list[ChatTurn])
  - `ChatTurn`: role ("user" | "assistant"), content (str)
  - `ChatResponse`: answer (str), citations (list[Citation]), context_used (int neurons), latency_ms (float)
  - `Citation`: source_id (str), content_snippet (str), neuron_type (str), tags (list[str]), metadata (dict)
  - `ImportRequest`: agent_type (str), content (str), source_name (str), tags (list[str] | None)
  - `ImportResponse`: neurons_created (int), fiber_id (str)
  - `AgentSourcesResponse`: sources (list[SourceEntry]), total (int)
  - `SourceEntry`: id (str), name (str), neuron_count (int), imported_at (str)
- Route stubs: POST /api/agent/chat, POST /api/agent/import, GET /api/agent/sources

### 1.2: NM Context Retrieval Helper
- Create `src/neural_memory/server/routes/agent.py::_retrieve_context()`
- Takes: query (str), brain_name (str), agent_type (str), max_tokens (int)
- Uses existing `ReflexPipeline.query()` internally (same as /memory/query)
- For legal: depth=2, tags=["legal"], max_tokens=2000
- For accounting: depth=1, tags=["accounting"], max_tokens=1500
- Returns: list of context chunks with neuron metadata (id, content, type, tags, created_at)
- Each chunk becomes a potential citation

### 1.3: Claude API Integration
- Add `anthropic` to pyproject.toml optional deps: `agent = ["anthropic>=0.40"]`
- Create `src/neural_memory/server/routes/agent.py::_call_llm()`
- Reads ANTHROPIC_API_KEY from env (raise clear error if missing)
- System prompt per agent_type:
  - Legal: "You are a legal consultant. ALWAYS cite exact article numbers. Use [cite:N] format for references. Never paraphrase law text — quote verbatim."
  - Accounting: "You are an accounting assistant. ALWAYS include exact numbers. Use [cite:N] format. Include timestamps and audit context from sources."
- Builds messages: system + context block + conversation history + user message
- Context block format: each chunk as `[Source N] (tags: ..., stored: ...)\n{content}`
- Calls `client.messages.create()` with model="claude-sonnet-4-20250514", max_tokens=1024
- Returns: response text + which [cite:N] references were used

### 1.4: Chat Endpoint Implementation
- POST /api/agent/chat — full flow:
  1. Validate request, resolve brain name (default: agent_type name)
  2. Call `_retrieve_context()` to get NM chunks
  3. Call `_call_llm()` with context + history
  4. Parse [cite:N] from response, map to Citation objects
  5. Return ChatResponse with answer, citations, latency
- Error handling: missing API key -> 503, empty brain -> hint to import data
- Cap conversation_history to last 20 turns

### 1.5: Import Endpoint Implementation
- POST /api/agent/import — encode content into agent brain:
  1. Validate content length (max 100k chars)
  2. Auto-create brain if not exists (brain name = agent_type)
  3. Use existing MemoryEncoder.encode() to store content
  4. Add tags: [agent_type, source_name] + user tags
  5. Store source_name in metadata for citation tracking
  6. Return ImportResponse with neuron count + fiber_id
- GET /api/agent/sources — list imported sources:
  1. Query neurons with tag filter for agent_type
  2. Group by metadata.source_name
  3. Return sorted by import date

### 1.6: Basic Tests
- Create `tests/unit/test_agent_routes.py`
- Test chat endpoint with mocked Claude API (no real API calls)
- Test import endpoint stores into correct brain
- Test context retrieval returns proper citation format
- Test missing API key returns 503
- Test conversation history capped at 20

## Acceptance Criteria

- `curl -X POST /api/agent/chat -d '{"message": "What does Article 5 say?", "agent_type": "legal"}'` returns JSON with answer + citations
- `curl -X POST /api/agent/import -d '{"content": "Article 5: ...", "agent_type": "legal", "source_name": "civil_code.txt"}'` stores neurons tagged "legal"
- `curl /api/agent/sources?agent_type=legal` lists imported sources
- All tests pass, no mypy errors
- Claude API key missing -> clear 503 error, not a crash

## Files Touched

### New Files
- `src/neural_memory/server/routes/agent.py`
- `tests/unit/test_agent_routes.py`

### Modified Files
- `src/neural_memory/server/routes/__init__.py` (add agent_router export)
- `src/neural_memory/server/app.py` (register agent_router)
- `src/neural_memory/server/models.py` (add Chat/Import/Citation models)
- `pyproject.toml` (add `agent` optional dep group)

## Dependencies

- None (Phase 1 is standalone backend)
- Requires `ANTHROPIC_API_KEY` env var for real chat (tests mock it)
