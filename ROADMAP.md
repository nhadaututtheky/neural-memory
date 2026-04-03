# NeuralMemory Roadmap

> Forward-looking vision. What's next, what's possible, where we're going.
> Every item passes the VISION.md 4-question test + brain test.
> ZERO LLM dependency — pure algorithmic, regex, graph-based.

**Current state**: v4.27.1 — 55 MCP tools, 5900+ tests, schema v38, SQLite + PostgreSQL + InfinityDB backends, neuroscience engine (10 brain-inspired algorithms), tiered memory (HOT/WARM/COLD + auto-tier + domain boundaries), decision intelligence, 7-module handler architecture, Pro bundled in main package (license-gated).
**Architecture**: Spreading activation reflex engine, biological memory model, MCP standard.

---

## What We've Built (v1.0 → v4.11)

| Capability | Version | Brain Test |
|------------|---------|------------|
| Spreading activation (4 depth levels + RRF score fusion) | v1.0–v2.29 | Associative reflex |
| 14 memory types, 24 synapse types | v1.0 | Typed memory |
| Hebbian learning + memory decay (type-aware) | v1.0 | Use it or lose it |
| Sleep consolidation (13 strategies: prune/merge/dream/mature/infer/...) | v1.0 | Sleep replay |
| Multi-format KB training (PDF/DOCX/PPTX/HTML/JSON/XLSX/CSV) | v2.0 | Learning from documents |
| Pinned KB memories (skip decay/prune/compress) | v2.0 | Core knowledge |
| Tool memory (PostToolUse → neuron clusters) | v2.25 | Procedural memory |
| Error resolution learning (RESOLVED_BY synapses) | v2.0 | Learning from mistakes |
| Multi-device sync (hub-spoke, 4 conflict strategies) | v2.0 | — |
| Fernet encryption + sensitive content auto-detect | v2.0 | — |
| VS Code extension (status bar, graph explorer, CodeLens) | v2.10 | — |
| REST API + WebSocket dashboard (7 pages) | v2.0 | — |
| Telegram backup integration | v2.0 | — |
| Brain versioning + transplant + merge | v2.0 | Portable consciousness |
| Algorithmic sufficiency gate (8-gate retrieval validator) | v2.0 | Attention filter |
| Codebase indexing + code-aware recall | v2.0 | — |
| SimHash deduplication + graph query expansion | v2.29 | — |
| Personalized PageRank activation (opt-in) | v2.29 | Hub dampening |
| RRF multi-retriever score fusion | v2.29 | — |
| Cognitive reasoning (hypothesize/evidence/predict/verify/gaps/schema) | v2.27 | Scientific reasoning |
| Source-Aware Memory (registry, exact recall, citations, audit) | v3.1 | Source memory |
| Structured encoding (tables, CSV, JSON arrays) | v3.1 | Structured recall |
| Cloud Sync Hub (Cloudflare Workers + D1, API key auth) | v3.3 | — |
| Session intelligence (topic EMA, auto-expiry, SQLite persist) | v3.2 | Working memory |
| Adaptive depth selection (calibration-driven, session-aware) | v3.4 | Efficient recall |
| Predictive priming (4-source: cache, topic, habit, co-activation) | v3.5 | Priming |
| Semantic drift detection (tag co-occurrence, Union-Find clustering) | v4.0 | Concept merging |
| Diminishing returns gate (stop traversal when no new signal) | v4.11 | Attention economy |
| Brain Quality Track A: Smart instructions, Knowledge Surface (.nm), reflection engine | v4.8–v4.9 | Proactive memory |
| Brain Quality Track B: Auto-consolidation, Hebbian retrieval, IDF keywords, adaptive decay | v4.8 | Graph quality |
| Lazy entity promotion (2+ mentions before neuron creation) | v4.8 | Selective encoding |
| Auto-importance scoring (heuristic priority from content signals) | v4.8 | Salience detection |
| PostgreSQL + pgvector backend | v4.7 | — |
| Context merger (structured `context` dict in remember) | v4.5 | — |
| Quality scorer (per-memory quality hints) | v4.5 | — |
| Onboarding overhaul (`nmem init --full`, `nmem doctor`) | v4.10 | — |
| IDE rules generator (Cursor, Windsurf, Cline, Gemini, AGENTS.md) | v4.6 | — |
| Cascading retrieval with fiber summary tier | v4.3 | — |
| HuggingFace Spaces chatbot (ReflexPipeline, no LLM) | v4.3 | — |
| Auto-tier engine, decision intelligence, domain boundaries | v4.24–v4.25 | Intelligent memory management |
| Milestone analysis (`nmem_milestone`) | v4.25 | Brain growth tracking |
| Pro merged into main package (license-gated, zero-friction install) | v4.27 | One install, key unlocks |
| Pay-hub (Cloudflare Workers + D1, SePay + Polar) | v4.27 | Payment infrastructure |
| File watcher ingestion (`nmem_watch` MCP + CLI) | v4.14 | Environmental learning |
| Agent visualization (`nmem_visualize` Vega-Lite/ASCII) | v4.15 | Brain introspection |
| Brain Oracle (card-based memory fortune teller, 3 modes) | v4.16 | Playful brain exploration |
| Markdown brain export (`snapshot_to_markdown`) | v4.17 | Human-readable export |
| OpenClaw plugin (v1.16.0, ClawHub marketplace) | v4.6 | IDE integration |

---

## Phase A: Production Hardening (v5.0)

> Ship quality. Fix gaps. Make existing features bulletproof.

### A1. PostgreSQL Backend Parity — Issues #83, #84, #86

**Problem**: PostgreSQL backend missing cognitive layer tables, `nmem_edit` persistence, and `pin_fibers` method. Users who chose PostgreSQL hit runtime errors.

**Scope**:
- [ ] Cognitive tables in PostgreSQL schema (cognitive_state, hot_index, knowledge_gaps)
- [ ] `pin_fibers()` implementation for PostgreSQL storage
- [ ] `nmem_edit` type/priority changes persisted in PostgreSQL
- [ ] Parity test suite: run SQLite test matrix against PostgreSQL

### A2. File Watcher Ingestion — Issue #66 ✅

**Problem**: Users manually run `nmem train` on files. Should be automatic: drop file → auto-memorize.

**Scope**: 3 phases (plan: `.rune/plan-file-watcher.md`)
- [x] Phase 1: Core FileWatcher class, watchdog integration, state tracking (mtime + simhash)
- [x] Phase 2: `nmem watch` CLI + `nmem_watch` MCP tool + config
- [x] Phase 3: `nmem serve` integration, debounce (2s), metrics
- **Brain test**: Não tự hấp thụ thông tin từ môi trường → Yes

### A3. Brain Quality Track C — Vertical Intelligence ✅

**Problem**: Brain treats all content the same. Domain-specific entities (financial amounts, legal references) deserve specialized extraction and encoding.

**Scope**: 3 sub-phases (plan: `.rune/plan-brain-quality.md`) — **all shipped**
- [x] C1+C2: Domain entity types + structured data encoding (regex-based, no LLM) — shipped v4.25
- [x] C3: Cross-encoder reranking (optional `bge-reranker` post-SA refinement) — shipped v4.25
- [x] C4: Agent visualization (`nmem_visualize` → Vega-Lite/markdown/ASCII charts) — shipped v4.25
- **Brain test**: Kế toán nhớ "ROE" khác "Paris" → Yes

### A4. Stability & Polish

- [x] Pre-ship smoke test automation (`scripts/pre_ship.py` → CI)
- [ ] E2E test coverage for dashboard (Playwright)
- [ ] Schema migration rollback testing (v29 → v28 → v27)
- [ ] Performance benchmarks: recall latency at 10K/50K/100K neurons
- [x] Consolidation performance fixes (v4.20.2-v4.20.4: timeouts, O(N²) caps, async yields)
- [x] InfinityDB integration fixes (7 bugs: singleton, list_brains, set_brain, WAL fallback, migrator)

### A5. Neuroscience Engine ✅ (plan: `.rune/plan-neuro-engine.md`)

**Problem**: Brain metaphor stops at storage/retrieval. Real brains have lateral inhibition, reconsolidation, prediction error, context-dependent recall, and tiered access patterns. NM treats all memories equally at encoding and retrieval time.

**Scope**: 4 phases, 10 improvements (~1600 LOC total) — **shipped v4.21.0**
- [x] Phase 1: Lateral Inhibition + Temporal Binding + Emotional Valence (~250 LOC)
- [x] Phase 2: Prediction Error Encoding + Retrieval Reconsolidation (~350 LOC)
- [x] Phase 3: Context-Dependent Retrieval + Hippocampal Replay + Working Memory Chunking (~470 LOC)
- [x] Phase 4: Schema Assimilation + Interference Forgetting (~550 LOC)
- [x] v4.21.1: Multilingual support (en/vi + agnostic fallback), input firewall noise stripping, `clean_for_prompt` recall mode
- **Brain test**: ALL 10 improvements map to documented neuroscience principles → Yes
- **Zero LLM**: Pure algorithmic (regex, SimHash, graph ops). No embeddings required.
- 107 new tests, post-encode hooks, paginated tag fetch, real activation scores

### A6. Tiered Memory Loading — Issue #111 ✅ (plan: `.rune/plan-tiered-memory.md`)

**Problem**: All memories have equal access priority. Real brains have fast-access working memory vs long-term storage. Safety rules and user preferences should always be available, not just when semantically matched.

**Scope**: Logical tiers on neurons (prerequisite for C1 physical storage tiers) — **shipped v4.22.0, fixes v4.22.1**
- [x] Schema migration v37: `tier TEXT DEFAULT 'warm'` on typed_memories + index
- [x] HOT tier: always injected into context, decay floor = 0.5, MAX_HOT_CONTEXT_MEMORIES = 50
- [x] WARM tier: default behavior (semantic match, normal decay)
- [x] COLD tier: explicit `nmem_recall` only, 2× decay rate, excluded from auto-context
- [x] `nmem_remember(..., tier="hot")` + `nmem_edit` + `nmem_recall(tier=...)` filter
- [x] `nmem_pin` → auto-promote to HOT
- [x] Safety boundaries: `type=boundary` → always HOT, enforced in create/edit/pin/decay
- [x] Context optimizer: HOT +0.3 score boost, COLD excluded by default
- [x] Dashboard: TierDistribution card (progress bars), `count_typed_memories()` SQL COUNT
- [x] v4.22.1: 6 review fixes — with_priority data loss, boundary migration v38, case-insensitive tier, broader exception handling
- **42 new tests** across 4 phase files
- **Brain test**: Não có working memory (nhanh) vs long-term memory (chậm) → Yes
- **Backward compatible**: default `warm` → existing memories unchanged

### A7. Recall Intelligence — Smarter Context, Fewer Mistakes

**Problem**: Brain stores both mistakes and corrections, but recall doesn't know which supersedes which. Agents repeat old wrong approaches because:
1. RESOLVED_BY/EVOLVES_FROM synapses are stored but **ignored during recall ranking**
2. Error demotion only reduces activation 50-75% — errors still appear in context
3. Workflow habits (BEFORE synapses) don't boost recall — treated like any other synapse
4. `report_outcome` updates metadata but creates no synapses → outcomes don't feed back into retrieval
5. No "follow the correction chain" logic — agent sees the bug, not the fix

**Scope**: 4 phases (plan: `.rune/plan-recall-intelligence.md`)

#### Phase 1: Supersession-Aware Recall
- [ ] During activation spread, if neuron has `_resolved_by` → auto-inject the resolver with boosted score
- [ ] Superseded neurons demoted to ghost-level (show as "outdated: see X" hint, not full content)
- [ ] EVOLVES_FROM chain following: always surface the LATEST version of a decision/workflow
- [ ] "Correction chain" metadata: when recalling error, append "Fixed by: {resolver content}"
- **Brain test**: Não nhớ cách sửa chứ không lặp lại lỗi → Yes

#### Phase 2: Outcome-Driven Learning
- [ ] `report_outcome` creates VERIFIED_BY (success) or FALSIFIED_BY (failure) synapses → feeds back into graph
- [ ] Success rate affects recall ranking: high success_rate → boost, low → demote
- [ ] Failure modes stored as synapses (LEADS_TO error type) → agent warned before repeating
- [ ] "Confidence score" per memory: `success_count / execution_count` visible in context
- **Brain test**: Não học từ kết quả, không chỉ từ kinh nghiệm → Yes

#### Phase 3: Workflow Recall Boost
- [ ] Habit synapses (BEFORE) get activation boost proportional to `_habit_frequency`
- [ ] When recalling step N of a workflow, auto-prime step N+1 (predictive priming v2)
- [ ] Workflow version tracking: old workflow → new workflow via EVOLVES_FROM → recall latest
- [ ] "Workflow context" mode: recall all steps of a detected workflow, not just keyword matches
- **Brain test**: Thói quen lặp đi lặp lại trở thành phản xạ → Yes

#### Phase 4: Cross-Session Context Bridge
- [ ] Session-end auto-summary: corrections made, decisions taken, patterns discovered
- [ ] Session-start injection: most recent corrections/decisions for the active domain
- [ ] "Correction memory" type: explicitly stores "old way → new way" with RESOLVED_BY
- [ ] Inter-session learning: if Agent A fixes bug X, Agent B's next session recalls the fix
- **Brain test**: Ngủ dậy nhớ bài học hôm qua, không lặp sai → Yes

**Target**: v5.0 = "production-ready for teams" release.

---

## Phase B: Monetization & Growth (v5.x → v6.0)

> From open-source tool to sustainable product. Revenue enables long-term development.

### B1. Sync Hub: Landing + Payment ✅ (plan: `.rune/plan-sync-hub-phase3.md`)

**Problem**: Cloud sync works but has no billing. Need landing page + payment flow.

**Scope**: **Shipped** — landing at `neuralmemory.theio.vn`, pay-hub at `pay.theio.vn`
- [x] Landing page (GitHub Pages) — features, pricing, pro-landing.html
- [x] SePay integration (Vietnam, 0% fee) — VietQR checkout
- [x] Polar integration (international) — Card/PayPal checkout
- [x] Pro tier ($9/mo or $89/yr, 219k VND/tháng)
- [x] Dashboard UpgradeModal — purchase + license activation
- [x] Pay-hub D1 orders table, webhook fulfillment, /verify endpoint
- [ ] Usage dashboard: sync history, storage used, device count (deferred)

### B2. Sync Hub: Team Sharing (plan: `.rune/plan-sync-hub-phase4.md`)

**Problem**: Each agent has its own brain. Knowledge doesn't flow between team members.

**Scope**:
- [ ] Team brain: shared namespace with per-user attribution
- [ ] Roles: owner, editor, viewer
- [ ] Activity feed: "Agent B learned about React hooks 2 hours ago"
- [ ] Audit log: who changed what, when
- **Brain test**: Collective memory (team knowledge) → Yes

### B3. Distribution & Discoverability

**Problem**: Users don't know NeuralMemory exists. Need to be where they search.

**Scope**:
- [x] MCP Registry listing (modelcontextprotocol.io)
- [x] awesome-mcp-servers PR (punkpeye/awesome-mcp-servers)
- [x] PyPI package optimization (description, classifiers, keywords)
- [x] npm package for OpenClaw plugin (v1.16.0, ClawHub marketplace)
- [ ] Blog posts: "NeuralMemory vs Mem0", "Why spreading activation beats RAG"
- [ ] HuggingFace Spaces demo polished + promoted

### B4. Brain Marketplace v1

**Problem**: Expert knowledge is siloed. A React expert's brain could help thousands of developers.

**Scope**:
- [ ] `nmem brain publish --name "react-19-patterns" --tags react,hooks,rsc`
- [ ] `nmem brain install react-19-patterns --merge`
- [ ] Brain packages: versioned, with metadata (description, tags, size, neuron count)
- [ ] Discovery: browse/search on sync hub landing page
- [ ] Free tier: publish up to 3 brains. Premium: unlimited + featured listing
- **Brain test**: Humans learn from books/teachers (external knowledge) → Yes

### B5. Pro: Smart Tiers & Decision Intelligence ✅ — Issue #112

**Problem**: Free tier gives manual HOT/WARM/COLD control (#111). Pro makes it intelligent — auto-promote/demote by usage, structured decision matching, domain-scoped safety boundaries.

**Scope**: Requires #111 (A6) first — plan: `.rune/plan-smart-tiers.md`
- [x] Phase 1: Auto-tier Engine — promote/demote by access patterns, Pro-gated, TierEngine + MCP tool (v4.24.0)
- [x] Phase 2: Decision Intelligence — structured components, overlap scoring, EVOLVES_FROM synapse (v4.24.0)
- [x] Phase 3: Domain Boundaries — `domain:` tag convention, domain-filtered HOT context, `nmem_boundaries` tool (v4.25.0)
- [x] Phase 4: Tier Analytics — dashboard API, distribution charts, promotion history (v4.26.0)
- **Foundation**: A5 hippocampal replay (LTP/LTD) already provides strengthen/weaken mechanism
- **Monetization gate**: Intelligence, not data access — users always see all memories, Pro makes management smarter
- **Brain test**: Não tự điều chỉnh độ ưu tiên theo thói quen → Yes

**Target**: v6.0 = "NeuralMemory as a service" with revenue stream.

---

## Phase C: Scale & Enterprise (v6.x → v7.0)

> From laptop brain to production brain. Handle millions of neurons.

### C1. Tiered Storage Architecture

**Problem**: SQLite great for <500K neurons. Beyond that, graph queries slow down.

**Vision**: Hybrid storage — hot data in memory/FalkorDB, warm in SQLite WAL, cold in compressed archives.
**Prerequisite**: A6 Tiered Memory Loading (logical tiers) + B5 Auto-tier (intelligent placement). C1 adds the *physical* storage layer underneath.

```
Hot tier (in-memory + optional FalkorDB) — recent + frequently activated
  ↕ auto-promote/demote
Warm tier (SQLite WAL)                   — moderate activity, queryable
  ↕ auto-archive
Cold tier (SQLite read-only, compressed) — archived, rarely accessed
```

- Access frequency drives tier placement (already tracked in NeuronState)
- KB (pinned) memories stay in hot tier permanently
- Single query interface — storage layer handles tier routing transparently
- **Target**: Sub-100ms recall at 1M+ neurons
- **Brain test**: Não có vùng nhớ nhanh (working memory) vs nhớ dài hạn → Yes

### C2. Approximate Nearest Neighbor Index

**Problem**: SimHash dedup is O(n) scan. At 500K+ neurons, embedding-based recall bottlenecks.

**Vision**: ANN index (sqlite-vec or HNSW) for embedding pre-filtering, spreading activation refines within candidate set.

- ANN narrows 500K → 500 candidates, SA refines final ranking
- Index rebuilds async during consolidation (not on hot path)
- **Important**: Acceleration, not replacement. Spreading activation remains central.
- **Brain test**: Não có vùng chuyên lọc nhanh trước khi phản xạ sâu (thalamus) → Yes

### C3. Partitioned Brain Sharding

**Problem**: Single brain file grows unbounded. At GB scale, VACUUM takes minutes.

**Vision**: Auto-shard by domain_tag or time window. Each shard is independent SQLite file.

- Domain shards: `brain-kb-react.db`, `brain-kb-python.db`, `brain-organic-2026-Q1.db`
- Query router fans out to relevant shards only
- Cross-shard synapses: `(shard_id, neuron_id)` tuple reference
- **Target**: Individual shard stays <200MB, total brain can be 10GB+

### C4. Self-Hosted Brain Hub (Production Docker)

**Problem**: Current sync is Cloudflare-hosted. Enterprises need self-hosted option.

**Scope**:
- [ ] Docker one-liner: `docker run -p 8080:8080 neuralmemory/hub`
- [ ] Admin dashboard: connected devices, sync status, brain health
- [ ] Backup: automatic daily snapshots to configurable storage (S3/GCS/local)
- [ ] Rate limiting + connection pooling
- **Deployment targets**: Docker, Kubernetes, Railway, Fly.io

**Target**: v7.0 = "enterprise-ready" with million-neuron scale.

---

## Phase D: Platform & Ecosystem (v7.0+)

> From tool to platform. NeuralMemory as the memory standard for AI.

### D1. Brain Protocol Specification

**Vision**: Publish formal spec for how AI memory systems should work. Any vendor can implement it.

- Core spec: neuron/synapse/fiber model, spreading activation algorithm, consolidation rules
- Transport: MCP (primary), REST, gRPC
- Serialization: brain export format (JSON + binary embeddings)
- Compliance test suite: "Does your memory system pass Brain Protocol tests?"

### D2. Plugin Architecture

**Vision**: Plugin hooks at every lifecycle stage. Community extends NM without forking.

```
Lifecycle hooks:
  on_encode    → custom extraction, enrichment, tagging
  on_recall    → custom ranking, filtering, augmentation
  on_consolidate → custom pruning, merging, summarization
  on_decay     → custom decay curves, preservation rules
  on_sync      → custom conflict resolution, transformation
```

- Plugin registry: `nmem plugin install sentiment-boost`
- Sandboxed execution: plugins can't break core

### D3. Multi-Modal Memory

**Vision**: Extend neuron types beyond text — images, code AST, audio.

- Image neurons: store image embeddings, activate on visual similarity
- Code neurons: AST-aware storage, activate on structural similarity
- Audio neurons: voice memo → transcription + audio embedding
- Cross-modal synapses: screenshot → error message → fix
- **Brain test**: Não lưu đa phương thức → Yes

### D4. Federation Protocol

**Vision**: Brain Hubs peer with each other. Selective knowledge sharing across organizations.

- Federation handshake: Hub A ↔ Hub B establish trust
- Selective sync: share only neurons tagged with specific domains
- Discovery: brain directory service (like DNS for brains)

---

## Phase E: Intelligence Frontier (v8.0+)

> Where NeuralMemory goes beyond current AI memory paradigms.

### E1. Dream Engine v2 (Insight Generation) — *partially pulled to A5 Phase 3*

**Vision**: During consolidation, detect patterns across unrelated memories → surface non-obvious connections.

- Cross-domain pattern detection: "auth tokens expire" + "memory decay" → pattern
- Anomaly detection: memories that should be connected but aren't
- Weekly "dream report": "Your brain discovered 3 new connections this week"
- **Pulled forward**: Hippocampal Replay (biased LTP/LTD) → A5 Phase 3
- **Brain test**: Dreams create unexpected associations → Yes

### E2. Forgetting Curves & Spaced Repetition — *partially pulled to A5 Phase 4*

**Vision**: Integrate Ebbinghaus curves into recall loop. Foundation exists (`nmem_review` with Leitner boxes).

- Auto-schedule review for important memories approaching decay threshold
- Agent hints: "You haven't recalled 'deployment checklist' in 14 days. Review?"
- Memories surviving multiple reviews → lower decay rate automatically
- **Pulled forward**: Interference Forgetting (retroactive/proactive/fan effect) → A5 Phase 4
- **Brain test**: Não cần ôn lại để nhớ lâu → Yes

### E3. Contextual Personality — *partially pulled to A5 Phase 3*

**Vision**: Brain adapts retrieval based on agent persona, task context, user preferences.

- "Security expert" persona → boost security-related synapses
- "Quick chat" context → shallow depth; "code review" → deep
- Personality profiles stored as brain metadata
- **Pulled forward**: Context-Dependent Retrieval (project/topic fingerprint) → A5 Phase 3
- **Brain test**: Context ảnh hưởng cách não nhớ → Yes

### E4. Causal Reasoning Engine — *partially pulled to A5 Phase 2*

**Vision**: Detect implicit causality from temporal patterns. "X always happens before Y" → auto-create causal synapse.

- Temporal co-occurrence mining (existing sequence_mining foundation)
- Confidence scoring (correlation ≠ causation guard)
- Counterfactual queries: "What would have happened if X didn't occur?"
- Causal graph visualization in dashboard
- **Pulled forward**: Prediction Error Encoding (surprise signal) → A5 Phase 2
- **Brain test**: Não suy luận nhân quả từ kinh nghiệm → Yes

---

## Stretch Goals (Exploratory)

> Ideas worth tracking. May never ship, but inform direction.

| Idea | Brain Test | Feasibility | Impact |
|------|-----------|-------------|--------|
| **Voice interface** — speak memories, hear recalls | Yes (auditory) | Medium | High UX |
| **Spatial memory** — memories tied to locations/projects | Yes (hippocampus) | Medium | Medium |
| **Sleep mode** — agent idle → deep consolidation | Yes (sleep cycle) | Easy | High quality |
| **Brain aging** — long-lived brains develop "wisdom" | Yes (wisdom) | Hard | High value |
| **Memory palace** — spatial organization of knowledge | Yes (method of loci) | Hard | Novel |
| **Neuroplasticity** — brain structure adapts to usage | Yes (plasticity) | Medium | High |
| **Mirror neurons** — learn by observing other agents | Yes (mirror system) | Hard | Team AI |

---

## Guiding Principles

Every roadmap item must pass:

1. **Activation, not search** — Does this make recall more like reflex, not query?
2. **Spreading activation stays central** — Is graph traversal still the core mechanism?
3. **Works without embeddings** — Would this work with pure graph + SimHash?
4. **Detailed query = faster recall** — Does specificity still help?
5. **Brain test** — Does a real brain do something analogous?
6. **Zero LLM dependency** — Pure algorithmic. LLM is optional enhancement, never requirement.

---

## Priority Signal

| Phase | Timeline | Risk | Value |
|-------|----------|------|-------|
| Phase A: Production Hardening | Now → v5.0 | Low | High — stability unlocks adoption |
| Phase B: Monetization & Growth | v5.x → v6.0 | Medium | Critical — revenue sustains development |
| Phase C: Scale & Enterprise | v6.x → v7.0 | Medium | High — unlocks enterprise use cases |
| Phase D: Platform & Ecosystem | v7.0+ | High | Transformative — memory standard for AI |
| Phase E: Intelligence Frontier | v8.0+ | High | Moonshot — novel AI memory paradigm |

---

*See [VISION.md](VISION.md) for the north star guiding all decisions.*
*Last updated: 2026-04-03*
