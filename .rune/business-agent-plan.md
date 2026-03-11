# NM Business Agent — Master Plan

Demo chatbot showing Neural Memory powering small business AI assistants.
Two verticals: Legal Consulting (citation-driven) and Accounting (verbatim recall with audit trail).

## Architecture

```
dashboard/src/features/business-agent/
  BusinessAgentPage.tsx       -- Main page: chat + sidebar
  components/
    ChatPanel.tsx             -- Message list + input
    ChatMessage.tsx           -- Single message (user/assistant/system)
    CitationCard.tsx          -- Expandable source reference card
    DataImportModal.tsx       -- Upload/paste docs for training
    AgentPicker.tsx           -- Switch between Legal / Accounting agent
    SourceViewer.tsx          -- Full-text source preview panel
  hooks/
    useChatAgent.ts           -- Chat state + streaming via REST
    useAgentBrain.ts          -- Brain switching + stats for agent vertical
  api/
    agent.ts                  -- REST client for /api/agent/* routes
  types.ts                    -- ChatMessage, Citation, AgentConfig types

src/neural_memory/server/routes/
  agent.py                    -- NEW: /api/agent/* routes (chat, import, sources)
```

## Tech Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM Backend | Claude API (Anthropic SDK) | Best reasoning, user already has key |
| Chat transport | REST POST (streaming via SSE) | Simple, no WebSocket needed for demo |
| Memory backend | NM REST API (encode + query) | Already exists, just wire up |
| Data import | NM train handler (file upload) | Already supports md/txt/pdf/csv/xlsx |
| New deps | `anthropic` Python SDK only | Minimal, well-maintained |
| Frontend | New dashboard page | Reuse existing React/Vite/api client |
| Brain per vertical | Separate brains (legal / accounting) | Clean data isolation |

## Data Flow

```
User message
  -> POST /api/agent/chat { message, agent_type, brain }
  -> Backend: query NM for relevant context (POST /memory/query)
  -> Backend: build prompt with context + citations
  -> Backend: call Claude API with augmented prompt
  -> Backend: stream response back via SSE
  -> Frontend: render message with inline [citation] links
  -> Frontend: citation cards expand to show source + article/record
```

## Vertical Configs

### Legal Consulting Agent
- Brain: "legal" — trained on law articles (structured: Article X, Clause Y)
- Behavior: NEVER paraphrase law text. Always cite exact article number.
- NM usage: query with depth=2 (context-level), tags=["legal", "article"]
- Import: .txt/.md files with law articles, structured headings
- Demo data: 20-50 Vietnamese law articles (sample)

### Accounting Agent
- Brain: "accounting" — trained on transaction records, salary data
- Behavior: Remember exact numbers. Always show audit trail (when stored, by whom).
- NM usage: query with depth=1 (instant), include metadata timestamps
- Import: .csv/.xlsx with transaction records
- Demo data: 50-100 sample transactions

---

## PHASE 1: Backend Chat API + NM Integration
## PHASE 2: React Chat UI + Message Rendering
## PHASE 3: Data Import + Citation System
## PHASE 4: Vertical Polish + Demo Data

---

## Status

- [ ] Plan created (2026-03-11)
- [ ] Phase 1 — Backend
- [ ] Phase 2 — Chat UI
- [ ] Phase 3 — Import + Citations
- [ ] Phase 4 — Demo Polish
