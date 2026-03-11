# Phase 2: React Chat UI + Message Rendering

## Goal

Build the chat interface as a new dashboard page. Users can type messages, see streaming responses, and view which NM neurons were used. Two agent modes (legal/accounting) selectable via tab.

## Tasks

### 2.1: Types + API Client
- Create `dashboard/src/features/business-agent/types.ts`:
  - `ChatMessage`: id, role ("user"|"assistant"|"system"), content, citations?, timestamp
  - `Citation`: source_id, content_snippet, neuron_type, tags, metadata
  - `AgentType`: "legal" | "accounting"
  - `AgentConfig`: type, brain_name, display_name, description, icon, color
- Create `dashboard/src/features/business-agent/api/agent.ts`:
  - `sendMessage(message, agentType, history)` -> POST /api/agent/chat -> ChatResponse
  - `importData(content, agentType, sourceName, tags)` -> POST /api/agent/import
  - `getSources(agentType)` -> GET /api/agent/sources
- Use existing `dashboard/src/api/client.ts` fetch wrapper

### 2.2: Chat State Hook
- Create `dashboard/src/features/business-agent/hooks/useChatAgent.ts`
- Zustand store or useState-based hook:
  - `messages: ChatMessage[]`
  - `isLoading: boolean`
  - `agentType: AgentType`
  - `sendMessage(text)` — appends user msg, calls API, appends assistant msg
  - `switchAgent(type)` — clears conversation, switches brain
  - `clearChat()` — reset messages
- Convert API ChatResponse -> ChatMessage with citations attached
- Optimistic: show user message immediately, loading indicator for assistant
- Error state: show error as system message (red styling)

### 2.3: ChatMessage Component
- Create `dashboard/src/features/business-agent/components/ChatMessage.tsx`
- User messages: right-aligned, bg-blue/indigo
- Assistant messages: left-aligned, bg-card
- System messages: centered, muted text
- Parse `[cite:N]` in assistant content -> render as clickable superscript badges
- Clicking citation badge scrolls to / highlights the CitationCard below the message
- Timestamp shown on hover
- Markdown rendering for assistant content (use existing deps or simple regex for bold/code/lists)

### 2.4: ChatPanel Component
- Create `dashboard/src/features/business-agent/components/ChatPanel.tsx`
- Layout: messages list (scrollable, auto-scroll to bottom) + input bar at bottom
- Input: textarea with placeholder per agent type, Ctrl+Enter or button to send
- Send button disabled when loading or empty input
- Loading indicator: 3-dot animation in assistant message bubble
- Empty state: centered illustration + "Start by importing data or asking a question"
- Messages list: map ChatMessage components, citations inline under each assistant message

### 2.5: CitationCard Component
- Create `dashboard/src/features/business-agent/components/CitationCard.tsx`
- Compact card under assistant messages showing cited sources
- Shows: [N] badge, source name, snippet (truncated to 200 chars), tags
- Click to expand: full content, neuron type, stored timestamp, metadata
- Visual style: left border color coded by neuron type (concept=blue, entity=green, etc.)
- "View in Graph" link (opens /graph?highlight=neuron_id) — optional, if easy

### 2.6: AgentPicker + Page Composition
- Create `dashboard/src/features/business-agent/components/AgentPicker.tsx`
  - Two tabs/cards: Legal Consulting | Accounting
  - Each shows: icon, description, brain stats (neuron count)
  - Active tab highlighted
- Create `dashboard/src/features/business-agent/BusinessAgentPage.tsx`
  - Layout: AgentPicker on top, ChatPanel below (flex column, full height)
  - Sidebar entry: add to AppShell with chat-bubble icon + "Business Agent"
- Wire into routing:
  - `dashboard/src/App.tsx`: add lazy route for /business-agent
  - `dashboard/src/components/layout/AppShell.tsx`: add sidebar link

### 2.7: i18n + Styling
- Add i18n keys to `dashboard/src/i18n/en.json` and `vi.json`:
  - businessAgent.title, businessAgent.legal.name, businessAgent.accounting.name
  - businessAgent.placeholder.legal, businessAgent.placeholder.accounting
  - businessAgent.empty, businessAgent.importHint
- Styling: dark mode compatible, use existing CSS variables from dashboard
- Responsive: chat usable on tablet (768px+), stack on mobile

## Acceptance Criteria

- Navigate to /business-agent in dashboard, see agent picker + chat panel
- Type a message, see it appear right-aligned, loading indicator, then assistant response
- Assistant response shows [1] [2] citation badges inline
- Click citation badge -> CitationCard expands with source details
- Switch agent tab clears conversation
- Works in dark mode
- No new npm dependencies (use existing React + Tailwind)

## Files Touched

### New Files
- `dashboard/src/features/business-agent/BusinessAgentPage.tsx`
- `dashboard/src/features/business-agent/types.ts`
- `dashboard/src/features/business-agent/api/agent.ts`
- `dashboard/src/features/business-agent/hooks/useChatAgent.ts`
- `dashboard/src/features/business-agent/components/ChatPanel.tsx`
- `dashboard/src/features/business-agent/components/ChatMessage.tsx`
- `dashboard/src/features/business-agent/components/CitationCard.tsx`
- `dashboard/src/features/business-agent/components/AgentPicker.tsx`

### Modified Files
- `dashboard/src/App.tsx` (add route)
- `dashboard/src/components/layout/AppShell.tsx` (add sidebar entry)
- `dashboard/src/i18n/en.json` (add keys)
- `dashboard/src/i18n/vi.json` (add keys)

## Dependencies

- Phase 1 backend must be complete (chat API endpoint)
- Existing dashboard must build successfully
