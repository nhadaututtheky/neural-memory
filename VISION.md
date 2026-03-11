# NeuralMemory — Vision & North Star

> NeuralMemory không tìm memory — nó **kích hoạt** memory.
> Recall Through Activation, Not Search.

---

## The Key: Associative Reflex (Phản xạ Liên tưởng)

NeuralMemory không phải database. Không phải vector store. Không phải RAG pipeline.

NeuralMemory là **bộ nhớ sinh học cho AI** — nó sống, nó quên, nó liên tưởng,
và nó là chuẩn mở cho mọi AI tool chia sẻ ký ức.

Khi bạn hỏi một câu, NeuralMemory không "tìm kiếm" — nó **kích hoạt lan truyền**
qua mạng neuron, giống cách bộ não thật phản xạ. Ký ức liên quan tự nổi lên,
không cần keyword match, không cần cosine similarity.

---

## 4 Trụ Cột Cốt Lõi

### 1. Recall Through Activation (Kích hoạt, không tìm kiếm)

```
Query → Spreading Activation → Related memories surface naturally
```

- Không phải keyword search, không phải vector similarity
- Là **lan truyền kích hoạt** qua graph — hỏi "API format" thì nhớ luôn
  "authentication decision" vì chúng liên kết qua synapse
- Depth levels (instant → context → habit → deep) giống cách não nhớ nhanh vs nhớ sâu
- Query càng chi tiết (thêm keyword, time) → recall càng nhanh, dùng ít context hơn
- Đây là lợi thế lớn nhất của associative vs search

**Nếu bỏ hết vector embeddings và semantic search, hệ thống vẫn phải recall được.**
Nếu không → đã lệch khỏi key cốt lõi.

### 2. Temporal & Causal Topology (Cấu trúc Thời gian & Nhân quả)

```
"Vì chuyện A xảy ra chiều qua, nên sáng nay tôi mới làm chuyện B"
```

- Ký ức không phẳng (flat) — mọi ký ức KHÔNG có giá trị ngang nhau
- Ký ức phải có **chiều sâu thời gian** và **logic nguyên nhân - kết quả**
- Hệ thống phải trả lời được "**Tại sao?**" và "**Khi nào?**" một cách tự nhiên
  nhờ duyệt các sợi thần kinh thời gian, không chỉ "Cái gì?" như RAG
- Não nhớ theo chuỗi: sự kiện → nguyên nhân → hệ quả → quyết định

### 3. Portable Consciousness (Tính Di động của Ý thức)

```
Brain chuyên gia Crypto → lắp vào Agent A (hỗ trợ khách hàng)
                        → lắp vào Agent B (trading bot)
Toàn bộ phản xạ và liên tưởng được chuyển giao nguyên vẹn.
```

- Bộ nhớ KHÔNG dính chặt vào một Agent duy nhất
- "Bộ não" là module có thể tháo lắp — **Brain-as-a-Service**
- Export/Import/Merge giữ nguyên toàn bộ cấu trúc graph
- Tập trung vào Packaging và Standardization để "Swap não" mượt mà
- MCP protocol → bất kỳ AI agent nào cũng plug-in được

### 4. Source-Aware Memory (Nhớ Nguồn, Không Chỉ Nhớ Nội Dung)

```
Luật sư không nhớ nguyên văn 10,000 điều luật.
Luật sư nhớ: "lãi suất" → Điều 468 BLDS → mở sách ra đọc.
Kế toán nhớ chính xác: "Lương tháng 3 = 25 triệu, vì hợp đồng quy định."
```

Não thật có **2 chế độ nhớ** — NeuralMemory phải hỗ trợ cả hai:

#### Chế độ 1: Verbatim Memory (Nhớ Nguyên Văn)

- Nhớ chính xác: con số, giá trị, dữ kiện cụ thể
- Kèm audit trail: ai lưu, khi nào, tại sao, từ đâu ra
- Use case: kế toán, dữ liệu tài chính, thông số kỹ thuật
- **Recall trả đúng giá trị gốc**, không summarize

#### Chế độ 2: Navigational Memory (Nhớ Đường Đi)

- Không nhớ nguyên văn — nhớ **graph path** đến nguồn gốc
- Neuron chứa pointer (source, article, page) thay vì full text
- Khi cần exact quote → resolve pointer → fetch từ source gốc
- Use case: luật pháp, tài liệu tham khảo, knowledge base lớn
- **Recall trả con đường + metadata**, source resolution trả nguyên văn

#### Source = First-Class Concept

```
Source (e.g. "Bộ luật Dân sự 2015")
  ├── version, effective_date, status (active/repealed)
  ├── file_hash (integrity verification)
  └── Neurons linked via SOURCE_OF synapse
      ├── Neuron: "Điều 468 — Lãi suất" (pointer + summary)
      ├── Neuron: "Điều 466 — Nghĩa vụ trả nợ" (pointer + summary)
      └── Synapse: Điều 468 REFERENCES → Điều 466
```

- Mọi neuron phải trả lời được: **"Thông tin này từ đâu ra?"**
- Source tracking không phải metadata tùy chọn — nó là **thuộc tính cốt lõi**
- Agent cite source thay vì bịa → trustworthy, verifiable
- **Brain test**: Não biết "tôi đọc cái này ở đâu" (source memory) → Yes ✅

---

## Công Thức Kiểm Tra Mỗi Khi Update

Trước khi thêm feature, refactor, hoặc mở rộng, **tự hỏi 5 câu này**:

### Câu 1: Activation hay Search?

> Feature này giúp recall giống **phản xạ** hơn hay chỉ giúp **search** tốt hơn?

Nếu chỉ giúp search → **lệch hướng**.

### Câu 2: Spreading Activation vẫn là trung tâm?

> Tính năng này có còn giữ nguyên cơ chế "kích hoạt lan truyền" là trung tâm không?

Nếu "không, giờ chủ yếu search rồi" → **lệch hướng**.

### Câu 3: Bỏ embeddings vẫn chạy?

> Nếu bỏ hết vector embeddings và semantic search, hệ thống vẫn recall được không?

Nếu "không" → **lệch khỏi key cốt lõi**.

### Câu 4: Query chi tiết hơn = nhanh hơn?

> Khi query chi tiết hơn (thêm keyword/time), recall có nhanh hơn & dùng ít context hơn không?

Đây là lợi thế lớn nhất của associative vs search. Nếu mất lợi thế này → **lệch hướng**.

### Câu 5: Truy xuất nguồn được không?

> Agent có thể trả lời "thông tin này từ đâu ra?" với source + timestamp + confidence không?

Nếu "không" → **thiếu accountability**. Memory không có nguồn = hallucination risk.

---

## Brain Test

Ngoài 4 câu trên, luôn hỏi thêm:

> **"Bộ não thật có làm điều này không?"**

| Feature idea | Brain test | Verdict |
|---|---|---|
| Memory decay over time | Não quên dần | Yes |
| Consolidation (prune/merge) | Não gom ký ức khi ngủ | Yes |
| Spreading activation | Liên tưởng tự nhiên | Yes |
| Typed memories (fact, decision, todo) | Não phân loại ký ức | Yes |
| Conflict resolution on merge | Hai nguồn mâu thuẫn → não chọn | Yes |
| Temporal & causal links | "Vì A nên B" | Yes |
| Emotional valence | Ký ức gắn cảm xúc | Yes |
| Source memory ("đọc ở đâu?") | Não nhớ nguồn gốc thông tin | Yes |
| Verbatim recall (exact data) | Não nhớ chính xác số điện thoại | Yes |
| Navigational recall (know where) | "Nó ở trang 5 chương 3" | Yes |
| Structured data (tables, rows) | Não hiểu cấu trúc dữ liệu | Yes |
| Audit trail (ai nói, khi nào) | Não nhớ ai nói gì | Yes |
| Full-text search engine | Não không grep | **No** |
| Vector similarity ranking | Não không tính cosine | **Careful** |
| AI-generated summaries | Não tự tóm tắt | Yes |

---

## Memory Lifecycle (Vòng đời ký ức)

```
Create → Reinforce → Decay → Consolidate → Forget
  ↑                                           |
  └───────── Re-activate ─────────────────────┘
```

- **Create**: Ký ức mới tạo ra yếu
- **Reinforce**: Được nhắc lại → mạnh lên (Hebbian learning)
- **Decay**: Không dùng → phai dần
- **Consolidate**: "Ngủ" → gom lại, tỉa bớt, tạo schema
- **Forget**: Hết hạn hoặc quá yếu → bị xóa
- **Re-activate**: Ký ức cũ được kích hoạt lại → quay về Reinforce

Đây là thứ mà Redis, Pinecone, ChromaDB **không có**.

---

## Roadmap Định Hướng Theo Vision

> **Chi tiết đầy đủ xem [ROADMAP.md](ROADMAP.md)** — versioned roadmap v0.14.0 → v1.0.0
> với gap coverage matrix, expert feedback mapping, và VISION.md checklist per phase.

### Đã có (v2.29.0)

- [x] Spreading activation retrieval (4 depth levels + RRF score fusion)
- [x] Hebbian learning (reinforcement through use)
- [x] Memory decay over time (type-aware)
- [x] Sleep & Consolidate (13 strategies: prune/merge/summarize/mature/infer/dream/...)
- [x] Typed memories (14 types) with priorities and expiry
- [x] Brain export/import/merge (portable consciousness)
- [x] Conflict resolution (4 strategies)
- [x] MCP protocol (40 tools, standard memory layer)
- [x] VS Code extension
- [x] REST API + WebSocket sync + Multi-device hub
- [x] Cognitive layer (hypothesize/evidence/predict/verify/gaps/schema)
- [x] SimHash deduplication + graph query expansion
- [x] Score breakdown (activation, freshness, frequency, intersection)
- [x] Auto-tags (entity + keyword extraction, Vietnamese NLP)
- [x] Personalized PageRank activation (opt-in)
- [x] Multi-format KB training (PDF/DOCX/PPTX/HTML/JSON/XLSX/CSV)
- [x] Fernet encryption + sensitive content auto-detect

### Chưa có — Cần cho Pillar 4 (Source-Aware Memory)

| Feature | Why | Brain Analogy |
|---------|-----|---------------|
| **Source Registry** | Mọi neuron trả lời "từ đâu ra?" | Não nhớ "tôi đọc ở sách X" |
| **Exact Recall Mode** | Trả raw content, không summarize | Não đọc lại nguyên văn khi cần |
| **Structured Encoding** | Schema-aware (tabular data, key-value) | Não hiểu cấu trúc, không chỉ text |
| **Citation Engine** | Built-in `[cite:source/article]` output | Não trích dẫn có nguồn |
| **Audit Synapse** | STORED_BY, VERIFIED_AT, APPROVED_BY | Não nhớ ai nói, khi nào |
| **Source Resolution** | Pointer → external DB/file lookup | Não biết "mở sách ra trang X" |

### Hướng phát triển tiếp theo

| Version | Theme | Key Deliverable |
|---------|-------|-----------------|
| v3.0 | Source-Aware Memory | Source registry, exact recall, structured encoding |
| v3.x | Brain Intelligence | Adaptive retrieval, predictive priming, drift detection |
| v4.0 | Scale & Performance | Tiered storage, ANN index, brain sharding |
| v5.0 | Platform & Ecosystem | Brain Protocol spec, plugin architecture, federation |

See [ROADMAP.md](ROADMAP.md) for full details.

---

## What NeuralMemory Is NOT

| NeuralMemory is NOT | It IS |
|---|---|
| A database | A living memory system |
| A vector store | An associative reflex engine |
| A search engine | An activation network |
| A RAG pipeline | A biological memory model |
| A cache | A consciousness module |
| Flat storage | Temporal-causal topology |
| Vendor-locked | A portable open standard |
| A black box | A source-traceable knowledge graph |
| A text blob store | A structured, typed memory with provenance |

---

## Two Memory Modes (Hai Chế Độ Nhớ)

NeuralMemory phải hỗ trợ cả hai cách não thật nhớ thông tin:

```
┌──────────────────────┐     ┌──────────────────────┐
│  VERBATIM MODE       │     │  NAVIGATIONAL MODE   │
│  "Nhớ chính xác"     │     │  "Biết đường đi"     │
│                      │     │                      │
│  Kế toán nhớ:        │     │  Luật sư nhớ:        │
│  "Lương A = 25tr"    │     │  "lãi suất → Đ.468"  │
│  → recall trả exact  │     │  → recall trả pointer │
│  → kèm audit trail   │     │  → resolve trả exact  │
│                      │     │                      │
│  Data IN brain       │     │  Graph IN brain       │
│  (episodic memory)   │     │  Data IN source       │
│                      │     │  (semantic + pointer)  │
└──────────────────────┘     └──────────────────────┘
```

Cả hai chế độ đều dùng **spreading activation** — khác nhau ở:
- Verbatim: neuron content = exact data, recall trả content trực tiếp
- Navigational: neuron content = summary + pointer, recall trả path, resolution trả content

---

## One-Line North Star

> **NeuralMemory: Bộ nhớ sinh học cho AI — kích hoạt thay vì tìm kiếm,
> liên tưởng thay vì truy vấn, truy xuất nguồn thay vì bịa đặt,
> di động thay vì gắn chặt.**

---

*Last updated: 2026-03-11*
