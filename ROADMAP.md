# NeuralMemory Roadmap

> Forward-looking vision. What's next, what's possible, where we're going.
> Every item passes the VISION.md 4-question test + brain test.
> ZERO LLM dependency — pure algorithmic, regex, graph-based.

**Current state**: v2.17.0 — 28 MCP tools, 2830+ tests, schema v20, SQLite + FalkorDB backends.
**Architecture**: Spreading activation reflex engine, biological memory model, MCP standard.

---

## Where We Are Now

| Capability | Status |
|------------|--------|
| Spreading activation (4 depth levels) | Shipped |
| 11 memory types, 24 synapse types | Shipped |
| Hebbian learning + memory decay | Shipped |
| Sleep consolidation (ENRICH/DREAM/PRUNE) | Shipped |
| Multi-format KB training (PDF/DOCX/PPTX/HTML/JSON/XLSX/CSV) | Shipped |
| Pinned KB memories (skip decay/prune/compress) | Shipped |
| Tool memory (PostToolUse → neuron clusters) | Shipped |
| Error resolution learning (RESOLVED_BY synapses) | Shipped |
| Multi-device sync (hub-spoke, 4 conflict strategies) | Shipped |
| Fernet encryption + sensitive content auto-detect | Shipped |
| FalkorDB graph backend (opt-in) | Shipped |
| VS Code extension (preview) | Shipped |
| REST API + WebSocket dashboard | Shipped |
| Brain versioning + transplant + merge | Shipped |
| Algorithmic sufficiency gate (8-gate retrieval validator) | Shipped |
| Codebase indexing + code-aware recall | Shipped |

---

## Phase 1: Brain Intelligence (v2.18–v2.20)

> Make the brain smarter without adding LLM dependency.

### 1.1 Adaptive Retrieval

**Problem**: Fixed depth levels (instant/context/habit/deep) don't adapt to query complexity. User asks simple question → wastes cycles on deep traversal.

**Vision**: Brain learns optimal depth per query pattern. Track which depth level produces useful results for each query type → auto-select depth.

- EMA-calibrated depth selection from `retrieval_calibration` table (foundation exists)
- Per-query-type thresholds (fact queries → shallow, "why" queries → deep)
- Diminishing returns gate — stop traversal when new hops add no new signal
- **Brain test**: Não thật không duyệt toàn bộ ký ức cho mỗi câu hỏi → Yes ✅

### 1.2 Predictive Activation (Priming)

**Problem**: Brain only activates when queried. Real brains prime related memories before they're needed.

**Vision**: Based on session context + habits, pre-activate likely-needed memories. When agent starts working on "auth" → KB memories about auth are already warm.

- Session topic detection from recent `nmem_remember` / `nmem_recall` calls
- Habit-based priming: "after recall('deployment'), user usually recalls('env vars')"
- Pre-warm activation levels for predicted queries
- Reduces recall latency for anticipated needs
- **Brain test**: Não thật prime ký ức liên quan trước khi cần → Yes ✅

### 1.3 Semantic Drift Detection

**Problem**: Over time, same concept gets different tags: "API" vs "REST" vs "endpoint" vs "route". Brain fragments.

**Vision**: Detect tag clusters that refer to the same concept → auto-merge or suggest merge.

- Tag co-occurrence matrix from fiber metadata
- Cluster detection (tags that always appear together = likely synonyms)
- Auto-normalize or prompt user: "API, REST, endpoint appear to mean the same thing. Merge?"
- Prevents brain fragmentation without LLM
- **Brain test**: Não thật gom khái niệm tương tự → Yes ✅

---

## Phase 2: Scale & Performance (v2.21–v2.25)

> From laptop brain to production brain. Handle millions of neurons.

### 2.1 Tiered Storage Architecture

**Problem**: SQLite great for <500K neurons. Beyond that, graph queries slow down. FalkorDB exists but is all-or-nothing switch.

**Vision**: Hybrid storage — hot data in FalkorDB (fast graph traversal), cold data in SQLite (cheap storage), automatic tiering.

```
Hot tier (FalkorDB/Redis)     — recent + frequently activated neurons
  ↕ auto-promote/demote
Warm tier (SQLite WAL)        — moderate activity, still queryable
  ↕ auto-archive
Cold tier (SQLite read-only)  — archived, compressed, rarely accessed
```

- Access frequency drives tier placement (already tracked in NeuronState)
- KB (pinned) memories stay in hot tier permanently
- Organic memories flow naturally: hot → warm → cold → pruned
- Single query interface — storage layer handles tier routing transparently
- **Target**: Sub-100ms recall at 1M+ neurons

### 2.2 Approximate Nearest Neighbor Index

**Problem**: SimHash dedup is O(n) scan. At 500K+ neurons, embedding-based recall becomes bottleneck.

**Vision**: Add optional ANN index (HNSW or IVF) for embedding-based pre-filtering, while keeping spreading activation as the core mechanism.

- ANN narrows candidates from 500K → 500
- Spreading activation refines within candidate set
- Index rebuilds async during consolidation (not on hot path)
- **Important**: This is acceleration, not replacement. Spreading activation remains central.
- **Brain test**: Não có vùng chuyên lọc nhanh trước khi phản xạ sâu → Yes ✅ (thalamus)

### 2.3 Partitioned Brain Sharding

**Problem**: Single brain file grows unbounded. At GB scale, even SQLite VACUUM takes minutes.

**Vision**: Auto-shard brain by domain_tag or time window. Each shard is independent SQLite file, cross-shard synapses use lightweight references.

- Domain shards: `brain-kb-react.db`, `brain-kb-python.db`, `brain-organic-2026-Q1.db`
- Query router fans out to relevant shards only
- Cross-shard synapses: `(shard_id, neuron_id)` tuple reference
- Merge shards when needed (transplant already supports this)
- **Target**: Individual shard stays <200MB, total brain can be 10GB+

---

## Phase 3: Cloud & Collaboration (v2.26–v3.0)

> From local brain to shared brain. Multi-agent, multi-device, multi-user.

### 3.1 Brain Hub Server (Production-Ready)

**Problem**: Current sync is basic hub-spoke. Need production-grade deployment for teams.

**Vision**: Self-hostable Brain Hub that multiple agents/devices connect to. Real-time sync with conflict resolution.

- Docker one-liner: `docker run -p 8080:8080 neuralmemory/hub`
- WebSocket real-time sync (not polling)
- Auth: API key per device/agent
- Rate limiting + connection pooling
- Admin dashboard: see connected devices, sync status, brain health
- Backup: automatic daily snapshots to configurable storage (S3/GCS/local)
- **Deployment targets**: VPS, Docker, Kubernetes, Railway, Fly.io

### 3.2 Collaborative Brain

**Problem**: Each agent has its own brain. Knowledge doesn't flow between team members' agents.

**Vision**: Shared brain with namespaced contributions. Agent A's memories visible to Agent B, with attribution.

- Namespace per contributor: `agent-a/`, `agent-b/`, `shared/`
- Visibility rules: private (only me), team (my team), public (all agents)
- Merge strategies when knowledge conflicts across contributors
- Activity feed: "Agent B learned about React hooks 2 hours ago"
- **Brain test**: Collective memory (team knowledge) → Yes ✅

### 3.3 Brain Marketplace v2

**Problem**: Expert knowledge is siloed. A React expert's brain could help thousands of developers.

**Vision**: Publish & subscribe to brain packages. Install a "React 19" brain, merge into yours, get expert-level recall.

- Publish: `nmem brain publish --name "react-19-patterns" --tag react,hooks,rsc`
- Install: `nmem brain install react-19-patterns --merge`
- Versioned: brain packages have semver, changelog, compatibility matrix
- Revenue share: premium brains with subscription model
- Quality: community ratings, download counts, verified authors
- **Brain test**: Humans learn from books/teachers (external knowledge) → Yes ✅

### 3.4 Automated Backup & Disaster Recovery

**Problem**: Brain is precious data. Loss = catastrophic. No automated backup story.

**Vision**: Built-in backup with zero configuration.

- Local: automatic daily snapshots in `~/.neuralmemory/backups/` (rolling 7 days)
- Cloud: optional push to S3/GCS/Backblaze B2 (encrypted at rest)
- Point-in-time recovery: restore brain to any snapshot
- Integrity check: SHA-256 verification on restore
- CLI: `nmem backup create`, `nmem backup restore 2026-03-01`, `nmem backup list`

---

## Phase 4: Platform & Ecosystem (v3.0+)

> From tool to platform. NeuralMemory as the memory standard for AI.

### 4.1 Brain Protocol Specification

**Problem**: NeuralMemory is the only implementation. Need open spec for interoperability.

**Vision**: Publish "Brain Protocol" — formal spec for how AI memory systems should work. Any vendor can implement it.

- Core spec: neuron/synapse/fiber model, spreading activation algorithm, consolidation rules
- Transport: MCP (primary), REST, gRPC
- Serialization: brain export format (JSON + binary embeddings)
- Compliance test suite: "Does your memory system pass the Brain Protocol tests?"
- Submit to standardization body (or de facto standard via adoption)

### 4.2 Plugin Architecture

**Problem**: Adding new features requires core changes. Community can't extend NM easily.

**Vision**: Plugin hooks at every lifecycle stage. Community builds extensions without forking.

```
Lifecycle hooks:
  on_encode    → custom extraction, enrichment, tagging
  on_recall    → custom ranking, filtering, augmentation
  on_consolidate → custom pruning, merging, summarization
  on_decay     → custom decay curves, preservation rules
  on_sync      → custom conflict resolution, transformation
```

- Plugin registry: `nmem plugin install sentiment-boost`
- Plugin API: typed interfaces, versioned contracts
- Sandboxed execution: plugins can't break core

### 4.3 Multi-Modal Memory

**Problem**: NM only stores text. Real brains store images, sounds, spatial relationships.

**Vision**: Extend neuron types to support multi-modal content with cross-modal synapses.

- Image neurons: store image embeddings, activate on visual similarity
- Code neurons: AST-aware storage, activate on structural similarity
- Audio neurons: voice memo → transcription + audio embedding
- Cross-modal synapses: "this code screenshot" → "this error message" → "this fix"
- **Brain test**: Não lưu đa phương thức (hình ảnh, âm thanh, cảm giác) → Yes ✅

### 4.4 Federation Protocol

**Problem**: Brain Hubs are isolated. Knowledge doesn't flow between organizations.

**Vision**: Brain Hubs can peer with each other. Selective knowledge sharing across organizations.

- Federation handshake: Hub A ↔ Hub B establish trust
- Selective sync: only share neurons tagged with specific domains
- Privacy: encrypted transit, no plaintext on wire
- Discovery: brain directory service (like DNS for brains)
- **Use case**: Company A's "kubernetes-ops" brain peers with Company B's "cloud-infra" brain

---

## Phase 5: Intelligence Frontier (v4.0+)

> Where NeuralMemory goes beyond current AI memory paradigms.

### 5.1 Dream Engine v2 (Insight Generation)

**Problem**: Current DREAM consolidation creates associative synapses. But doesn't generate novel insights.

**Vision**: During consolidation, detect patterns across unrelated memories → surface non-obvious connections.

- Cross-domain pattern detection: "auth tokens expire" + "memory decay" → "use token expiry pattern for memory cleanup"
- Anomaly detection: memories that should be connected but aren't
- Weekly "dream report": "Your brain discovered 3 new connections this week"
- **Brain test**: Dreams create unexpected associations → Yes ✅

### 5.2 Forgetting Curves & Spaced Repetition

**Problem**: Current decay is time-based. Real forgetting follows Ebbinghaus curves — memories need reinforcement at increasing intervals to become permanent.

**Vision**: Integrate spaced repetition into the recall loop. Foundation exists (`nmem_review` with Leitner boxes).

- Auto-schedule review for important memories approaching decay threshold
- Agent receives hints: "You haven't recalled 'deployment checklist' in 14 days. Review?"
- Memories that survive multiple review cycles → lower decay rate automatically
- KB memories (pinned) skip this — they're already permanent
- **Brain test**: Não cần ôn lại để nhớ lâu → Yes ✅

### 5.3 Contextual Personality

**Problem**: Brain is static — same retrieval behavior regardless of who's asking or what context.

**Vision**: Brain adapts retrieval behavior based on agent persona, task context, and user preferences.

- Agent persona affects activation weights: "security expert" → boost security-related synapses
- Task context affects depth: "quick chat" → shallow, "code review" → deep
- User preferences: some users want brief recall, others want comprehensive
- Personality profiles stored as brain metadata, not hardcoded
- **Brain test**: Context ảnh hưởng cách não nhớ → Yes ✅

### 5.4 Causal Reasoning Engine

**Problem**: Current causal traversal follows existing CAUSED_BY/LEADS_TO synapses. Can't infer new causal relationships.

**Vision**: Detect implicit causality from temporal patterns. "X always happens before Y" → auto-create causal synapse.

- Temporal co-occurrence mining (existing sequence_mining foundation)
- Confidence scoring for inferred causality (correlation ≠ causation guard)
- "Counterfactual" queries: "What would have happened if X didn't occur?"
- Causal graph visualization in dashboard
- **Brain test**: Não suy luận nhân quả từ kinh nghiệm → Yes ✅

---

## Stretch Goals (Exploratory)

> Ideas worth tracking. May never ship, but inform direction.

| Idea | Brain Test | Feasibility | Impact |
|------|-----------|-------------|--------|
| **Voice interface** — speak memories, hear recalls | Yes (auditory memory) | Medium | High UX |
| **Spatial memory** — memories tied to locations/projects | Yes (hippocampus) | Medium | Medium |
| **Sleep mode** — agent idle → trigger deep consolidation | Yes (sleep cycle) | Easy | High quality |
| **Brain aging** — long-lived brains develop "wisdom" (meta-patterns) | Yes (wisdom) | Hard | High value |
| **Memory palace** — spatial organization of knowledge domains | Yes (method of loci) | Hard | Novel |
| **Neuroplasticity** — brain structure adapts to usage patterns | Yes (plasticity) | Medium | High |
| **Mirror neurons** — learn by observing other agents' actions | Yes (mirror system) | Hard | Team AI |
| **Emotional context** — mood affects recall (already partial) | Yes (affect) | Easy | Medium |

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
| Phase 1: Brain Intelligence | Next | Low | High — smarter recall without complexity |
| Phase 2: Scale & Performance | Medium | Medium | Critical — unlocks enterprise use |
| Phase 3: Cloud & Collaboration | Medium-Long | Medium | High — multi-agent is the future |
| Phase 4: Platform & Ecosystem | Long | High | Transformative — memory standard for AI |
| Phase 5: Intelligence Frontier | Exploratory | High | Moonshot — novel AI memory paradigm |

---

*See [VISION.md](VISION.md) for the north star guiding all decisions.*
*Last updated: 2026-03-02*
