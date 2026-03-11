# Phase 4: Vertical Polish + Demo Data + Landing

## Goal

Make the demo presentable. Pre-load sample data for both verticals, add visual polish, create a "demo mode" that works without an API key (canned responses), and add a simple landing/intro screen explaining the value prop.

## Tasks

### 4.1: Demo Data — Legal Vertical
- Create `src/neural_memory/server/demo_data/legal_samples.json`
- 30 sample Vietnamese law articles (Civil Code, Labor Code, Enterprise Law):
  - Each entry: article_number, title, full_text (verbatim), source_document, tags
  - Mix of commonly-asked topics: employment contracts, company formation, property rights
- Create `src/neural_memory/server/routes/agent.py::_seed_demo_data()`
  - POST /api/agent/demo/seed?agent_type=legal
  - Idempotent: skip if brain already has data (check neuron count)
  - Imports all samples into "legal" brain with proper tags
  - Returns count of neurons created

### 4.2: Demo Data — Accounting Vertical
- Create `src/neural_memory/server/demo_data/accounting_samples.json`
- 80 sample transactions for a fictional small business "Cafe Saigon":
  - Each entry: date, description, amount, category, payee, notes
  - Mix: salary payments (5 employees x 6 months), rent, supplies, revenue, tax payments
  - Realistic Vietnamese business context (VND amounts, local suppliers)
- Seed endpoint: POST /api/agent/demo/seed?agent_type=accounting
- Same idempotent pattern as legal

### 4.3: Demo Mode (No API Key Fallback)
- When ANTHROPIC_API_KEY is not set, enable demo mode automatically
- Demo mode behavior in `agent.py::_call_llm()`:
  - Still queries NM for real context (the retrieval pipeline works without LLM)
  - Returns a template response: "Based on {N} relevant memories found: {top 3 context snippets formatted as citations}"
  - No LLM call, but still demonstrates NM's retrieval + citation capability
  - Add banner in ChatPanel: "Demo mode — LLM responses simulated. Set ANTHROPIC_API_KEY for full experience."
- This lets anyone run the demo without paying for API calls

### 4.4: Agent Intro Screen
- Create `dashboard/src/features/business-agent/components/IntroScreen.tsx`
- Shown when user first visits /business-agent (before any conversation)
- Content:
  - Hero: "NM Business Agent" title + one-line value prop
  - Two cards (legal / accounting) with:
    - Icon + name + 2-sentence description of what it does
    - "Try it" button -> selects agent + shows sample question
    - Brain stats: neurons loaded, sources imported
  - Bottom: "Import your own data" CTA
- After first message sent, IntroScreen hides, ChatPanel takes over
- "New conversation" button resets to IntroScreen

### 4.5: Visual Polish
- Update `ChatMessage.tsx`:
  - Agent avatar: scale icon for legal (gavel/book), calculator for accounting
  - Typing indicator: animated dots with fade
  - Message appear animation: slide-up + fade-in (CSS only, respect prefers-reduced-motion)
- Update `CitationCard.tsx`:
  - Legal citations: monospace article numbers, indented clause text
  - Accounting citations: compact table row (date | desc | amount)
  - Expand/collapse animation: height transition
- Update `AgentPicker.tsx`:
  - Active agent: subtle glow border + brain stats badge
  - Inactive agent: muted, hover to preview
- Overall: consistent with dashboard dark theme, no jarring style differences

### 4.6: Sample Conversations
- Create `dashboard/src/features/business-agent/data/sample-questions.ts`
- Legal samples:
  - "What are the requirements for forming an LLC in Vietnam?"
  - "What does Article 153 of the Labor Code say about overtime?"
  - "Can an employer terminate a contract during maternity leave?"
- Accounting samples:
  - "What was our total salary expense in January?"
  - "Show me all transactions with supplier Nguyen Van A"
  - "What's the rent payment history for the last 6 months?"
- Show as clickable chips in empty chat state
- Clicking a chip sends it as the first message

## Acceptance Criteria

- Run `nmem serve`, visit /business-agent -> see IntroScreen with two agent cards
- Click "Legal Consulting" -> "Seed Demo Data" button loads 30 law articles
- Click sample question "What does Article 153 say?" -> agent responds with exact article text + citation
- Switch to Accounting -> seed 80 transactions -> ask "total salary January" -> correct number with audit trail
- Demo mode works without ANTHROPIC_API_KEY (retrieval still works, response is templated)
- All visual animations respect prefers-reduced-motion
- Dark mode looks polished, consistent with rest of dashboard
- Full demo walkthrough takes < 3 minutes for a new user

## Files Touched

### New Files
- `src/neural_memory/server/demo_data/legal_samples.json`
- `src/neural_memory/server/demo_data/accounting_samples.json`
- `dashboard/src/features/business-agent/components/IntroScreen.tsx`
- `dashboard/src/features/business-agent/data/sample-questions.ts`

### Modified Files
- `src/neural_memory/server/routes/agent.py` (seed endpoints, demo mode)
- `dashboard/src/features/business-agent/BusinessAgentPage.tsx` (IntroScreen toggle)
- `dashboard/src/features/business-agent/components/ChatPanel.tsx` (demo banner, sample chips)
- `dashboard/src/features/business-agent/components/ChatMessage.tsx` (avatars, animations)
- `dashboard/src/features/business-agent/components/CitationCard.tsx` (vertical-specific rendering)
- `dashboard/src/features/business-agent/components/AgentPicker.tsx` (glow, stats, seed button)
- `dashboard/src/i18n/en.json` (intro + demo strings)
- `dashboard/src/i18n/vi.json` (intro + demo strings)

## Dependencies

- Phases 1-3 must be complete
- Demo data JSON files can be authored in parallel with Phase 3
- No external dependencies beyond what Phase 1 added (anthropic SDK)
