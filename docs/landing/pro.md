# Neural Memory Pro

> Your agent's brain, upgraded. From keyword search to semantic understanding.

Free Neural Memory is a complete, production-ready memory system — **you never have to pay**. But if your agent's brain is growing past 10K memories and you're noticing missed recalls, slow consolidation, or ballooning storage, Pro is the upgrade path.

**One command. No migration. No breaking changes.**

```bash
pip install neural-memory              # Pro features included
nmem pro activate YOUR_LICENSE_KEY    # activate with your key
```

All 52 free tools keep working. Your existing memories are preserved. Pro adds 3 new tools and upgrades the engine underneath.

> **New to Pro?** Start with the [Pro Quickstart Guide →](../guides/pro-quickstart.md)

---

## The Problem

Free Neural Memory stores memories as text and retrieves them by **keyword matching** (FTS5 BM25). It works great for small brains (<10K memories). But as your agent accumulates knowledge:

- **Recall degrades** — keyword search misses semantically related memories that use different words
- **Consolidation slows** — O(N²) pairwise comparison crawls past 1K neurons
- **Storage grows unbounded** — every neuron keeps full-precision vectors forever

Neural Memory Pro fixes all three.

---

## What Changes

### Before (Free — SQLite)

```
User: "What did we decide about the auth system?"
Agent: [FTS5 search: "decide" AND "auth" AND "system"]
       → Finds 2 of 7 relevant memories (keyword match only)
```

### After (Pro — InfinityDB)

```
User: "What did we decide about the auth system?"
Agent: [HNSW vector search: semantic embedding of query]
       → Finds 7 of 7 relevant memories (meaning match)
       → Ranked by: similarity × 0.7 + activation × 0.3
```

The difference isn't speed. It's **recall quality**. Your agent remembers by meaning, not by words.

---

## Feature Comparison

| | Free (SQLite) | Pro (InfinityDB) |
|--|---------------|-------------------|
| **Recall method** | Keyword match (FTS5 BM25) | Semantic similarity (HNSW) |
| **Search speed** | ~500ms at 10K neurons | **~5ms** at 10K neurons |
| **Search quality** | Exact/fuzzy word match | Meaning-based match |
| **Scale tested** | ~50K neurons | 2M+ neurons |
| **Vector storage** | Not stored | Persistent mmap on disk |
| **Compression** | Text-level (sentence trimming) | Vector-level (5-tier adaptive) |
| **Consolidation** | O(N²) brute-force | O(N×k) HNSW neighbor clustering |
| **Graph traversal** | SQL JOINs per hop | Native adjacency BFS, <1ms |
| **Crash recovery** | SQLite WAL | Custom WAL + idempotent replay |
| **MCP tools** | 55 tools | 52 + 3 Pro-exclusive |
| **Storage per 1M neurons** | ~5 GB | **~1 GB** (with tier compression) |

---

## Pro-Exclusive Features

### 1. Cone Queries — Semantic Recall

Instead of matching keywords, Cone Queries search a **semantic cone** around your query embedding.

```
nmem_cone_query(
  query="authentication architecture decisions",
  threshold=0.7,    # 0.65 = wide cone, 0.95 = narrow
  max_results=10
)
```

**How scoring works:**

```
combined_score = similarity × 0.7 + activation_level × 0.3
```

A memory can be semantically relevant (high similarity) even if rarely accessed. A frequently accessed memory (high activation) gets a boost even at moderate similarity. Both signals matter.

**Threshold controls precision:**

| Threshold | Behavior | Use case |
|-----------|----------|----------|
| 0.60–0.70 | Wide cone — more results, broader context | Exploration, brainstorming |
| 0.75–0.85 | Balanced — relevant results | Default recall |
| 0.90–0.95 | Narrow cone — only near-exact matches | Precise lookup |

---

### 2. Directional Compression — Smarter Memory Trimming

Free compression cuts sentences by entity density — it doesn't know *which direction* matters.

Pro uses **multi-axis directional compression**: it scores each sentence against multiple semantic directions (the memory's own embedding + up to 3 related neuron embeddings), keeping sentences that preserve **all** relevant directions.

**Example:**

A memory about "React performance optimization" relates to both React AND performance. Free compression might keep only React-heavy sentences (losing the performance angle). Pro keeps sentences scoring high on both axes.

```
Score per sentence = primary_similarity × 0.6 + max(reference_similarities) × 0.4
```

**Compression levels:**

| Level | Content kept | When applied |
|-------|-------------|--------------|
| FULL | 100% | Active memories |
| SUMMARY | 66% | Warm tier (7-30 days) |
| ESSENCE | 33% | Cool tier (30-90 days) |
| GHOST | First 5 words | Frozen tier (>90 days) |

---

### 3. Smart Merge — Consolidation That Scales

Free consolidation compares every pair of neurons → O(N²). At 5K neurons, that's 12.5M comparisons. Slow.

Pro uses HNSW neighbor search to find merge candidates in O(N×k):

1. For each neuron, find k=10 nearest HNSW neighbors
2. **Mutual similarity check**: A is near B AND B is near A (threshold: 0.82)
3. Cluster mutually similar neurons together
4. Rank by priority × activation → highest becomes anchor
5. Merge low-ranked into anchor, preserving unique information

```
nmem_pro_merge(
  similarity_threshold=0.85,
  max_merges=20,
  dry_run=true        # Preview before committing
)
```

**Scale comparison:**

| Neurons | Free (brute-force) | Pro (HNSW) |
|---------|-------------------|------------|
| 1,000 | ~2s | ~0.1s |
| 10,000 | ~3 min | ~1s |
| 100,000 | ~5 hours | ~10s |

---

### 4. Five-Tier Vector Compression — Automatic Lifecycle

Memories age. Pro automatically manages their storage footprint:

```
ACTIVE   float32   1,536 bytes/neuron   Recently accessed, high priority
   ↓
WARM     float16     768 bytes          7-30 days old           (-50%)
   ↓
COOL     int8        384 bytes          30-90 days old          (-75%)
   ↓
FROZEN   binary       48 bytes          >90 days old            (-97%)
   ↓
CRYSTAL  metadata      0 bytes          Archived, vector gone   (-100%)
```

**Smart rules prevent over-compression:**

- Priority ≥ 8 → always ACTIVE (critical memories never compress)
- Recent access → auto-promote back to higher tier
- Ephemeral memories → CRYSTAL immediately (scratch notes don't waste space)

**Storage impact at scale:**

| Brain size | Free (all float32) | Pro (mixed tiers) | Savings |
|-----------|-------------------|-------------------|---------|
| 10K neurons | 15 MB | 5 MB | 67% |
| 100K neurons | 150 MB | 25 MB | 83% |
| 1M neurons | 1.5 GB | 120 MB | 92% |

---

### 5. InfinityDB Engine — Purpose-Built for Neural Graphs

Not a wrapper around SQLite. A custom storage engine designed for one job: neural memory graphs.

**Architecture:**

```
brain.inf      Header (magic + version + dimensions + count)
brain.vec      Memory-mapped vectors (numpy mmap, zero-copy read)
brain.idx      HNSW index (hnswlib, M=16, ef=200)
brain.graph    Directed synapse edges (msgpack adjacency lists)
brain.meta     Neuron metadata (msgpack, O(1) ID lookup)
brain.fibers   Fiber collections (msgpack, bidirectional index)
brain.wal      Write-ahead log (max 50MB, idempotent replay)
```

**Why this matters:**

- **Zero SQL overhead** — binary access, no query parsing
- **Memory-mapped vectors** — OS handles caching, zero-copy reads
- **HNSW index** — O(log N) approximate nearest neighbor, not O(N) scan
- **Crash-safe** — WAL with idempotent replay, survives mid-write interruptions
- **Batch operations** — vectorized bulk insert + search

---

## Pro MCP Tools

Three new tools added to your agent's toolkit:

### `nmem_cone_query`

Semantic search with adjustable precision cone.

```json
{
  "query": "what helps with memory retention",
  "threshold": 0.7,
  "max_results": 10
}
```

Returns: neuron_id, content, similarity, activation, combined_score, type.

### `nmem_tier_info`

View and manage storage tier distribution.

```json
{ "action": "stats" }
→ { "tiers": { "active": 150, "warm": 200, "cool": 500, "frozen": 1000 },
    "estimated_savings_bytes": 1240000 }

{ "action": "sweep" }
→ Demotes stale memories to lower tiers automatically
```

### `nmem_pro_merge`

Smart consolidation with preview mode.

```json
{
  "similarity_threshold": 0.9,
  "dry_run": true,
  "max_merges": 50
}
→ { "clusters_found": 25, "merge_actions": 18, "details": [...] }
```

---

## Installation

One command. No config changes needed.

```bash
pip install neural-memory
nmem pro activate YOUR_LICENSE_KEY
```

Pro features are bundled in the main package — all dependencies included. Just activate your license key. Your existing 55 MCP tools keep working unchanged. Three new tools appear automatically.

To enable InfinityDB (semantic search engine), set `storage_backend = "infinitydb"` in your `config.toml`. On next startup, existing memories are auto-migrated from SQLite. Both databases coexist — downgrade is safe.

**Downgrade is safe:** remove Pro dependencies and everything reverts to free SQLite. No data loss — SQLite database is preserved alongside InfinityDB files.

---

## Who Should Upgrade

| Use case | Free is enough | Pro is better |
|----------|---------------|---------------|
| Personal agent, <5K memories | ✅ | |
| Keyword recall is sufficient | ✅ | |
| Single project, light usage | ✅ | |
| Growing brain, >10K memories | | ✅ |
| Need semantic recall (meaning, not words) | | ✅ |
| Storage is a concern | | ✅ |
| Frequent consolidation | | ✅ |
| Team/production deployment | | ✅ |

---

## Pricing

### Free — $0 forever

Everything you have today. 55 MCP tools, SQLite storage, spreading activation, 14 consolidation strategies, FTS5 search, cloud sync (100 neurons). **No features removed, ever.**

### Pro — $9/month (219,000 VND)

All free features plus:

- InfinityDB engine (HNSW vector search)
- Cone Queries (semantic recall)
- Smart Merge (scalable consolidation)
- Directional Compression (multi-axis)
- 5-tier vector compression (auto lifecycle)
- 3 Pro MCP tools
- Unlimited cloud sync
- Priority support

### Team — $29/month per seat (719,000 VND)

Everything in Pro plus:

- Shared brain hub (team knowledge graph)
- Brain-to-brain sync
- Role-based access control
- Audit log
- Self-hosted option

---

## Payment Methods

### International

Powered by [Polar.sh](https://polar.sh) — GitHub-native checkout.

- Credit/debit card (Visa, Mastercard, Amex)
- GitHub Sponsors integration
- Annual billing: 2 months free

### Vietnam

Powered by [Sepay](https://sepay.vn) — local payment gateway.

- Bank transfer (QR code — Vietcombank, Techcombank, MB Bank, etc.)
- MoMo, ZaloPay, VNPay
- Annual billing: 2 months free

---

## FAQ

**Does Pro require an internet connection?**
No. InfinityDB runs 100% locally. Cloud sync is optional.

**Will my existing memories transfer?**
Yes. After you enable InfinityDB (`storage_backend = "infinitydb"` in config.toml), SQLite data is automatically migrated on next startup. Both databases coexist — nothing is deleted.

**What happens if I cancel Pro?**
Your agent falls back to free SQLite storage. InfinityDB files remain on disk (in case you resubscribe). No data loss.

**Is Pro open source?**
Yes. The Pro plugin is open source on GitHub. InfinityDB engine code is included — no black boxes. A license key unlocks Pro features.

**Can I self-host the sync hub?**
Team plan includes a Docker image for self-hosted deployment. Pro plan uses the managed Cloudflare hub.

**How do I verify Pro is active?**
```bash
nmem pro status
# or via MCP:
nmem_tier_info(action="stats")
```

---

## Technical Specifications

| Spec | Value |
|------|-------|
| Python | ≥ 3.11 |
| Vector dimensions | 384 (default, configurable) |
| HNSW params | M=16, ef_construction=200 |
| Max WAL size | 50 MB |
| Max BFS traversal | 1,000 nodes |
| Max cone results | 500 |
| Batch insert | Vectorized, atomic rollback |
| File format | Custom binary (.inf, .vec, .idx, .graph, .meta, .fibers, .wal) |
| Crash recovery | Idempotent WAL replay |

---

*Neural Memory is MIT licensed. Pro is a paid add-on — the free version is complete and fully functional on its own.*
