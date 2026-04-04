# Pro Quickstart

You installed Pro. Here's how to make it work for you in 5 minutes.

---

## 1. Install and activate

```bash
pip install neural-memory
nmem pro activate YOUR_LICENSE_KEY
```

> Don't have a license key? [Purchase here →](https://nhadaututtheky.github.io/neural-memory/landing/pricing/)

Verify activation:

```bash
nmem pro status
```

```
Pro: Active
Backend: SQLite (InfinityDB available)
License: valid (expires 2026-04-26)
Features: cone_query, smart_merge, directional_compress, auto_tier
```

At this point, Pro tools (cone queries, smart merge, directional compression) are already available. But recall still uses SQLite keyword matching. To unlock semantic recall, you need to enable InfinityDB.

---

## 2. Enable InfinityDB

InfinityDB is Pro's semantic search engine — it's what makes recall work by **meaning** instead of keywords. Enabling it requires an explicit opt-in because it migrates your data.

Edit your config:

```toml
# ~/.neuralmemory/config.toml
storage_backend = "infinitydb"
```

Or via CLI:

```bash
nmem config set storage_backend infinitydb
```

**Restart your MCP server** (or CLI session). On first startup, Neural Memory automatically migrates your existing neurons from SQLite to InfinityDB. This may take a few minutes for large brains (>50K neurons).

After migration, verify:

```bash
nmem pro status
```

```
Pro: Active
Backend: InfinityDB
License: valid (expires 2026-04-26)
Features: cone_query, smart_merge, directional_compress, auto_tier
```

> **Your SQLite database is preserved** — both databases coexist. If you ever downgrade, Neural Memory falls back to SQLite automatically. No data loss.

---

## 3. Your first semantic recall

Free tier matches keywords. Pro with InfinityDB matches **meaning**.

```bash
# Store some memories
nmem remember "We chose PostgreSQL over MySQL for better JSON support"
nmem remember "JWT rotation was added to fix the session hijack vulnerability"
nmem remember "Alice suggested rate limiting after the DDoS incident"

# Free would need exact keywords. Pro finds semantic matches:
nmem recall "database decisions"       # finds PostgreSQL memory
nmem recall "security improvements"    # finds JWT + rate limiting
```

### Cone Queries — adjustable precision

Narrow the cone for exact matches, widen it for exploration:

```bash
# Via MCP tool
nmem_cone_query(query="auth", threshold=0.85)   # precise — only strong matches
nmem_cone_query(query="auth", threshold=0.60)   # exploratory — cast a wide net
```

Default threshold is `0.75`. Lower = more results, higher = more relevant.

---

## 4. Check your storage tiers

Pro automatically manages memory lifecycle across 5 tiers:

| Tier | Format | Size | When |
|------|--------|------|------|
| 1 | float32 | 100% | Fresh memories (< 7 days) |
| 2 | float16 | 50% | Maturing (7–30 days) |
| 3 | int8 | 25% | Stable (30–90 days) |
| 4 | binary | 3% | Archived (90+ days) |
| 5 | metadata | <1% | Ghost tier (rarely accessed) |

Memories auto-promote back to higher tiers when accessed. Check your distribution:

```bash
nmem_tier_info
```

```
Tier distribution:
  float32:  1,234 neurons (12%)
  float16:  3,456 neurons (34%)
  int8:     4,567 neurons (45%)
  binary:      890 neurons (9%)
  metadata:     12 neurons (<1%)

Total storage: 1.2 GB (vs ~5.1 GB without tiering)
Savings: 76%
```

---

## 5. Run Smart Merge

Standard consolidation is O(N²) — it slows down past 10K neurons. Smart Merge uses HNSW neighbor clustering for O(N x k):

```bash
# Dry run first — see what would be merged
nmem consolidate --strategy smart_merge --dry-run

# Run it
nmem consolidate --strategy smart_merge
```

Or via MCP:

```
nmem_pro_merge(dry_run=true)    # preview
nmem_pro_merge()                 # execute
```

Smart Merge finds semantically similar memories (not just keyword duplicates) and consolidates them while preserving causal links.

---

## 6. Connect Cloud Sync

Sync your brain across all your machines:

```bash
# First time: deploy your sync hub (Cloudflare Workers, free tier)
# See: https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/

# Configure sync
nmem_sync_config(hub_url="https://your-hub.workers.dev", api_key="your-key")

# Initial seed (uploads full brain)
nmem sync --seed

# After that: incremental sync
nmem sync              # manual
nmem sync --auto       # auto after every remember/recall
```

Pro sync uses **Merkle delta** — only changes are transmitted. A brain with 100K neurons syncs in under 2 seconds.

---

## What changed from Free

| Aspect | Before (Free) | After (Pro) |
|--------|---------------|-------------|
| Storage engine | SQLite + FTS5 | InfinityDB (HNSW vectors) |
| Recall method | Keyword matching | Semantic similarity |
| Consolidation | O(N²) brute force | O(N x k) Smart Merge |
| Compression | Text-level trimming | 5-tier vector lifecycle |
| New MCP tools | — | `nmem_cone_query`, `nmem_tier_info`, `nmem_pro_merge` |

**Everything else stays the same.** All 52 free tools still work. Your existing memories are preserved — when you enable InfinityDB (step 2), they're auto-migrated on first startup.

---

## Troubleshooting

### "Pro: Inactive" after install

```bash
# Check if Pro deps are installed
python -c "from neural_memory.pro import is_pro_deps_installed; print(is_pro_deps_installed())"

# Re-activate license
nmem pro activate YOUR_LICENSE_KEY

# Check license status
nmem pro status --verbose
```

### Recall quality didn't improve

Make sure you've enabled InfinityDB (step 2). If `nmem pro status` still shows `Backend: SQLite`, the config change didn't take effect. After enabling, InfinityDB needs to index your existing neurons — this may take a few minutes for large brains (>50K neurons). Check progress:

```bash
nmem_tier_info    # shows indexing progress
```

### Want to downgrade?

```bash
pip install neural-memory  # reinstall without [pro] extra
```

Your data stays intact. Neural Memory falls back to SQLite + FTS5 automatically. No data loss, no migration needed.

---

## Next steps

- [Full Pro comparison →](https://nhadaututtheky.github.io/neural-memory/landing/pro/)
- [Cloud Sync setup →](https://nhadaututtheky.github.io/neural-memory/guides/cloud-sync/)
- [Brain Health guide →](https://nhadaututtheky.github.io/neural-memory/guides/brain-health/)
- [Pricing & plans →](https://nhadaututtheky.github.io/neural-memory/landing/pricing/)
