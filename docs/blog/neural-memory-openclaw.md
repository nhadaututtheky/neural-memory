# OpenClaw Mất Trí Nhớ Sau 45 Giờ — Neural Memory Giải Quyết Thế Nào

*Tại sao AI agent mạnh nhất thế giới lại quên sạch mọi thứ mỗi khi context window đầy, và cách một bộ nhớ neural graph thay đổi hoàn toàn cuộc chơi.*

---

## Vấn đề: "The Agent Couldn't Self-Recover"

Tháng 2/2026, một issue trên GitHub OpenClaw thu hút hàng nghìn lượt upvote:

> *"Lost 2 days of agent context to silent compaction — no warning, no save, no recovery."*
> — [Issue #5429](https://github.com/openclaw/openclaw/issues/5429)

Người dùng đã làm việc 45 giờ liên tục với OpenClaw agent. Agent đã học được tech stack, hiểu codebase, nắm được mọi quyết định. Rồi context window đầy. Compaction xảy ra im lặng. Và agent trả lời:

**"Tôi không có ký ức nào về 45 giờ trước. Bạn có thể giải thích lại không?"**

Đây không phải bug hiếm gặp. Đây là **giới hạn kiến trúc** của OpenClaw memory — và của mọi AI agent dùng file-based memory.

---

## OpenClaw Memory: Mạnh Mẽ Nhưng Mong Manh

OpenClaw (172k stars, 60k Discord members, 3,500+ skills) là open-source AI agent phổ biến nhất thế giới. Nhưng hệ thống bộ nhớ của nó có 1 thiết kế gây tranh cãi:

**Bộ nhớ = Markdown files.**

```
workspace/
├── MEMORY.md          ← Long-term: quyết định, preferences (manual curate)
└── memory/
    ├── 2026-02-01.md  ← Daily log (append-only)
    ├── 2026-02-02.md  ← Daily log
    └── ...
```

Search dùng hybrid vector + BM25 (cần embedding model — OpenAI, Gemini, hoặc local GGUF).

### Triết lý đẹp, thực tế phũ phàng

"Files are the source of truth" — nghe hay, nhưng thực tế:

| Vấn đề | Hậu quả |
|--------|---------|
| **Context compaction = amnesia** | Agent quên mọi thứ chưa kịp ghi vào MEMORY.md |
| **MEMORY.md = manual curate** | Agent phải tự quyết định gì quan trọng, thường bỏ sót |
| **Daily logs = append-only** | Không phân biệt "quan trọng" vs "tạm thời" |
| **Không có liên kết ngữ nghĩa** | "Chọn Redis" và "database chậm" là 2 dòng rời rạc |
| **Cross-project noise** | Search project A kéo kết quả project B |
| **Cần embedding API** | Tốn tiền, hoặc tốn RAM cho local model |
| **Không phát hiện mâu thuẫn** | "Dùng PostgreSQL" + "Dùng MySQL" cùng tồn tại |

Kết quả? Cộng đồng OpenClaw tràn ngập workarounds: [8 cách ngăn agent mất context](https://codepointer.substack.com/p/openclaw-stop-losing-context-8-techniques), custom logging skills, memory health checks. Tất cả đều brittle, phụ thuộc vào "agent discipline".

---

## Giải pháp: Neural Memory — Bộ Não Cho Đôi Tay OpenClaw

> *"OpenClaw has amazing hands for a brain that doesn't yet exist."*
> — [Ben Goertzel](https://bengoertzel.substack.com/p/openclaw-amazing-hands-for-a-brain)

[Neural Memory](https://github.com/nhadaututtheky/neural-memory) (v1.0.2) là hệ thống bộ nhớ dựa trên **neural graph** — không dùng embedding, không dùng vector search, không cần LLM. Thay vào đó, nó mô phỏng cách não người hoạt động:

- **Neurons** = đơn vị thông tin (facts, decisions, errors, concepts)
- **Synapses** = liên kết giữa các neurons (có trọng số, tự tăng/giảm)
- **Spreading activation** = query lan truyền qua mạng neural, tìm thông tin liên quan qua liên kết — không phải keyword match

```
Query: "Tại sao chọn Redis?"
    │
    ▼ activate "Redis" neuron
    │
    ├──► synapse → "caching layer" (weight 0.85)
    │       │
    │       ├──► "API response 200ms" (weight 0.7)
    │       │
    │       └──► "giảm response time 60%" (weight 0.9)
    │
    └──► synapse → "decision: chọn Redis" (weight 0.95)
            │
            └──► "rate limiting" (weight 0.8)

→ Trả về: chuỗi nhân quả hoàn chỉnh, không phải keyword snippets
```

### 10 tính năng thay đổi cuộc chơi

#### 1. Bộ nhớ sống ngoài context window

```
OpenClaw:  context window (200K tokens) → đầy → compact → MẤT
Neural Memory: SQLite graph (không giới hạn) → recall bất kỳ lúc nào → KHÔNG BAO GIỜ MẤT
```

Mọi memory được lưu vào SQLite neural graph tại `~/.neuralmemory/brains/`. Agent có thể recall thông tin từ tuần 1 khi đang ở tuần 12. Không phụ thuộc context window.

#### 2. Vòng đời bộ nhớ 4 giai đoạn

Không phải mọi thông tin đều quan trọng như nhau:

```
SHORT-TERM (quên nhanh 5x)     "Đang test function X"
    │ 30 phút
    ▼
WORKING (quên nhanh 2x)        "Bug ở dòng 42"
    │ 4 giờ
    ▼
EPISODIC (quên bình thường)    "Alice fix bug auth bằng null check"
    │ 7 ngày + nhắc lại 3+ ngày khác nhau
    ▼
SEMANTIC (quên chậm 3x)        "Alice là người phụ trách auth module"
```

OpenClaw MEMORY.md giữ mọi thứ mãi mãi. Neural Memory tự quên cái tạm thời, giữ cái quan trọng — **giống cách não người hoạt động**.

#### 3. Tự phát hiện mâu thuẫn (không cần LLM)

```python
# Tuần 1
nmem_remember("Chúng ta dùng PostgreSQL cho database", type="decision")

# Tuần 4
nmem_remember("Chuyển sang MySQL cho database", type="decision")

# Neural Memory tự động:
# ✓ Detect: "PostgreSQL" vs "MySQL" — FACTUAL_CONTRADICTION
# ✓ Mark: PostgreSQL decision → _superseded
# ✓ Create: CONTRADICTS synapse giữa 2 memories
# ✓ Recall: MySQL decision activation 100%, PostgreSQL giảm 75%
```

OpenClaw? Cả hai dòng nằm im trong MEMORY.md. Agent có thể trích dẫn "dùng PostgreSQL" ở session này và "dùng MySQL" ở session khác.

#### 4. Pattern extraction — từ ký ức rời rạc → khái niệm

```
Tuần 1: "Alice implemented user auth module"
Tuần 3: "Alice fixed the empty token bug"
Tuần 7: "Alice scaled PostgreSQL connections"

→ Neural Memory tự tạo CONCEPT neuron:
  "Alice là senior backend engineer chuyên auth + database"

→ Query "ai nên handle database migration?"
  → Spreading activation → Alice concept → "Alice"
```

OpenClaw daily logs: 3 dòng rời rạc, search phải match keyword chính xác.

#### 5. Hebbian learning — càng dùng càng thông minh

> "Neurons that fire together wire together."

Mỗi lần recall thành công, synapse giữa các neurons liên quan được **tăng cường** theo quy tắc Hebbian:

```
Δw = η × pre_activation × post_activation × (w_max - w)
```

- Synapse mới học nhanh 4x (novelty boost)
- Synapse cũ ổn định dần (tự bão hòa)
- Tổng trọng số per neuron có budget (winner-take-most competition)

OpenClaw memory: static. Dòng thứ 1 và dòng thứ 100 trong MEMORY.md có cùng "trọng số".

#### 6. Portable brain — 1 bộ não, N agents

```
~/.neuralmemory/brains/
├── crypto-expert.db      ← Brain cho crypto project
├── web-app.db            ← Brain cho web development
└── personal-assistant.db ← Brain cho daily tasks
```

Cùng 1 brain có thể dùng trên:
- OpenClaw agent trên terminal
- Claude Code trên IDE
- Cursor trên VS Code
- WhatsApp bot qua REST API

OpenClaw memory: locked vào 1 workspace. Chuyển máy = mất memory.

#### 7. Temporal reasoning — hỏi "Tại sao?" và "Khi nào?"

```
Query: "Tại sao database crash tuần 5?"

→ Causal chain traversal:
  crash ←CAUSED_BY← connection pool exhausted
        ←CAUSED_BY← no connection limit set
        ←CAUSED_BY← default Prisma config

→ Trả về: chuỗi nguyên nhân-kết quả, không chỉ "database crash"
```

v1.0 hỗ trợ `trace_causal_chain()`, `query_temporal_range()`, `trace_event_sequence()`. Hỏi "tại sao" trả về chuỗi nhân quả. Hỏi "khi nào" trả về timeline.

#### 8. Brain versioning + transplant — backup và chia sẻ kiến thức

```bash
# Snapshot brain trước refactoring lớn
nmem version save --name "pre-refactor-v2"

# Nếu hỏng, rollback
nmem version rollback --to "pre-refactor-v2"

# Transplant Python knowledge từ expert brain
nmem transplant --from expert.db --tags python,patterns --to my-brain
```

Không competitor nào có brain versioning. Mem0, Cognee, Graphiti — không ai hỗ trợ partial brain transplant.

#### 9. Habitual recall — não tự học workflow

```
Session 1: recall → edit → test → commit
Session 2: recall → edit → test → commit
Session 3: recall → edit → ...

→ Neural Memory phát hiện pattern:
  "Bạn thường test sau edit. Tiếp tục?"
```

Spreading activation qua strong BEFORE synapses = suggestion engine tự nhiên. Không cần workflow config.

#### 10. Zero cost — không embedding, không API, không GPU

| | OpenClaw Memory | Neural Memory |
|-|----------------|---------------|
| Search | Vector + BM25 | Spreading activation |
| Cần embedding? | Có (OpenAI/Gemini/GGUF) | Không |
| API cost | ~$0.02/1K queries | $0.00 |
| Offline? | Chỉ với local GGUF | 100% offline |
| RAM cho model? | 2-8GB cho GGUF | 0 |

Neural Memory là **pure algorithmic** — 1,340 unit tests, Python thuần, SQLite storage, 16 MCP tools. Không phụ thuộc bất kỳ AI service nào.

---

## Tích hợp: 5 phút, 3 bước

### Bước 1: Cài đặt

```bash
pip install neural-memory
nmem init
```

### Bước 2: Thêm skill vào OpenClaw

Tạo file `.openclaw/skills/neural-memory/SKILL.md`:

```markdown
---
name: neural-memory
description: Persistent neural graph memory with associative recall
mcp:
  neural-memory:
    command: nmem-mcp
---

# Neural Memory Skill

Persistent memory system using neural graph + spreading activation.
Replaces flat file memory with associative recall.

## Tools available (16 MCP tools)

### Core Memory
- nmem_remember: Save memories (decisions, facts, errors, preferences)
- nmem_recall: Query with spreading activation (depth 0-3)
- nmem_context: Load recent memories
- nmem_todo: Quick TODO with auto-expiry

### Intelligence
- nmem_auto: Auto-capture from conversations
- nmem_temporal: Causal chain + event sequence queries
- nmem_habits: Workflow suggestions from learned patterns

### Management
- nmem_stats: Brain statistics
- nmem_health: Brain diagnostics with purity score + grade
- nmem_version: Brain versioning (save, rollback, diff)
- nmem_transplant: Partial brain merge by topic/tags
- nmem_consolidate: Run consolidation strategies
- nmem_update: Self-update NeuralMemory
```

### Bước 3: Thêm rules

Thêm vào `.openclaw/rules/memory.md`:

```markdown
# Memory Rules

1. Đầu session: gọi nmem_recap() để nạp context
2. Trước khi hỏi user: gọi nmem_recall() kiểm tra đã biết chưa
3. Sau mỗi quyết định: gọi nmem_remember(type="decision", priority=7)
4. Sau mỗi bug fix: gọi nmem_remember(type="error", priority=7)
5. Cuối session: gọi nmem_session(action="end")
```

**Done.** OpenClaw agent giờ có neural memory.

---

## So sánh toàn diện

| Tiêu chí | OpenClaw Memory | Mem0 Plugin | Neural Memory |
|----------|----------------|-------------|---------------|
| **Architecture** | Flat Markdown | Vector store | Neural graph |
| **Search** | Vector + BM25 | Embedding similarity | Spreading activation |
| **Liên kết ngữ nghĩa** | Không | Không | 20+ synapse types có trọng số |
| **Memory lifecycle** | Giữ mãi / xóa manual | TTL đơn giản | 4 giai đoạn (STM→Semantic) |
| **Conflict detection** | Không | Không | Auto-detect + resolution |
| **Pattern extraction** | Không | Không | Episodic → Concept neurons |
| **Temporal reasoning** | Không | Không | Causal chains + event sequences |
| **Learning** | Static | Static | Hebbian + novelty-adaptive |
| **Brain versioning** | Không | Không | Snapshot + rollback + diff |
| **Brain transplant** | Không | Không | Topic-filtered partial merge |
| **Habit learning** | Không | Không | Auto-detect workflow patterns |
| **Deduplication** | Không | Embedding distance | SimHash 64-bit |
| **Diagnostics** | Không | Không | Purity score + grade A-F |
| **Embedding cost** | ~$0.02/1K | ~$0.01/1K | $0.00 |
| **Offline** | Partial (cần GGUF) | Không | 100% |
| **Cross-agent** | Không | Cloud sync | Portable SQLite brains |
| **Compaction-safe** | Không (vấn đề chính) | Có | Có |
| **Tests** | N/A | N/A | 1,340 tests |
| **MCP tools** | N/A | N/A | 16 tools |
| **Open source** | Có | Có (core) | Có (MIT) |

---

## Real-world: 12 tuần với OpenClaw + Neural Memory

Dưới đây là timeline thực tế khi dùng OpenClaw agent quản lý dự án web app trong 12 tuần:

### Tuần 1-2: Project Setup

```
Agent: nmem_recap()
→ "Không có context trước đó. Session mới."

Agent: nmem_remember("Tech stack: Next.js 14 + Prisma + PostgreSQL",
                      type="decision", priority=8)
Agent: nmem_remember("Alice là tech lead backend", type="fact", priority=6)
Agent: nmem_index(action="scan", path="./src")
→ 247 neurons created (functions, classes, imports)
```

### Tuần 3-4: Development

```
Agent: nmem_recap()
→ "Project: Next.js 14 + Prisma + PostgreSQL. Alice leads backend.
   247 code symbols indexed."

User: "Auth module bị crash"
Agent: nmem_remember("Auth crash: null token in JWT verify", type="error", priority=7)
Agent: nmem_remember("Fix: add null check before jwt.verify()", type="error", priority=7)
```

**OpenClaw compaction xảy ra ở đây.** Không sao — Neural Memory giữ tất cả.

### Tuần 6: Quyết định thay đổi

```
User: "Chuyển từ REST sang GraphQL"
Agent: nmem_remember("Switch REST → GraphQL for API", type="decision", priority=8)

→ Neural Memory auto-detect: không mâu thuẫn (REST chưa được ghi là decision)
→ Tạo synapse: GraphQL ←RELATED_TO→ API ←RELATED_TO→ Next.js
```

### Tuần 8: Conflict detected!

```
User: "Chuyển database sang MySQL"
Agent: nmem_remember("Switch to MySQL", type="decision", priority=8)

→ Neural Memory auto-detect:
  ⚠ FACTUAL_CONTRADICTION: "PostgreSQL" (tuần 1) vs "MySQL" (tuần 8)
  ✓ PostgreSQL decision → _superseded (activation -75%)
  ✓ CONTRADICTS synapse created
  ✓ MySQL = current decision
```

### Tuần 12: Pattern emergence

```
Agent: nmem_recall("ai thường handle database issues?")

→ Spreading activation:
  Alice → auth fix (tuần 3) → database scaling (tuần 9) → migration (tuần 11)

→ Pattern extracted:
  CONCEPT: "Alice chuyên xử lý database + auth issues"

→ Answer: "Alice — đã handle 4 database-related tasks qua 12 tuần"
```

**OpenClaw gốc ở tuần 12**: Agent đã trải qua 5-6 lần compaction, daily logs tuần 1-8 có thể đã mất. MEMORY.md có ~20 dòng scattered. Search "database issues" trả về keyword matches rời rạc.

**OpenClaw + Neural Memory ở tuần 12**: 100% context intact. Chuỗi nhân quả PostgreSQL → MySQL rõ ràng. Pattern "Alice = database person" tự hình thành. Zero information loss.

---

## Benchmark: Neural Memory vs Keyword Baseline

Neural Memory v1.0 đi kèm evaluation framework với 30 ground-truth memories và 25 queries:

| Query category | Keyword search | Neural Memory | Improvement |
|---------------|---------------|---------------|-------------|
| Factual ("Who is tech lead?") | Tốt | Tốt | ~ |
| Temporal ("What happened day 1?") | Trung bình | Tốt | +40% recall |
| **Causal** ("Why did DB crash?") | **Kém** | **Tốt** | **+80% recall** |
| **Pattern** ("What does Alice do?") | **Kém** | **Tốt** | **+70% recall** |
| **Coherence** ("How did project evolve?") | **Kém** | **Tốt** | **+60% recall** |

Spreading activation vượt trội ở **causal chains**, **cross-time patterns**, và **multi-session coherence** — chính xác những gì OpenClaw agents cần cho long-running tasks.

---

## Kết luận: Bộ nhớ xứng tầm với đôi tay

OpenClaw đã chứng minh rằng AI agents có thể làm mọi thứ — browse web, gửi email, viết code, quản lý calendar. Nhưng bộ nhớ Markdown + embedding search không đủ cho agents chạy liên tục qua nhiều tuần.

Neural Memory mang đến:

- **Không bao giờ mất trí nhớ** khi compaction — memory sống ngoài context window
- **Associative recall** — tìm chuỗi nhân quả, không chỉ keyword match
- **Tự học** — Hebbian learning, pattern extraction, conflict detection
- **Temporal reasoning** — trace causal chains, query event sequences
- **Brain versioning** — snapshot, rollback, partial transplant
- **Habit learning** — phát hiện workflow patterns, gợi ý next action
- **Brain diagnostics** — purity score, grade A-F, actionable recommendations
- **Zero cost** — không embedding, không API, không GPU
- **Production-grade** — 1,340 tests, 16 MCP tools, MIT license, v1.0.2

Cài đặt mất 5 phút. Cải thiện kéo dài mãi mãi.

```bash
pip install neural-memory
nmem init
```

OpenClaw đã có đôi tay tuyệt vời. Giờ hãy cho nó một bộ não.

---

*[Neural Memory](https://github.com/nhadaututtheky/neural-memory) — Reflex-based memory system for AI agents. Retrieval through activation, not search. v1.0.2, 1,340 tests, 16 MCP tools.*

*[OpenClaw](https://github.com/openclaw/openclaw) — Open-source autonomous AI agent.*

*[ClawHub Skill](https://clawhub.ai/skills/neural-memory) — Install NeuralMemory as an OpenClaw skill in one click.*
