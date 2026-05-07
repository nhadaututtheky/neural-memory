# Temporal Recall Recipes

The `nmem_causal` MCP tool exposes two temporal actions — `temporal_range` and `temporal_neighborhood` — that let agents reason about *when* things happened, not just *what* happened. This guide shows the patterns most agents actually need.

Both actions are read-only and cost-bounded (limit clamped to 50 by default, max 200).

## Action Reference (Quick)

| Action | Input | Output |
|---|---|---|
| `temporal_range` | `start`, `end` (ISO-8601) | All fibers anchored within the window |
| `temporal_neighborhood` | `fiber_id`, `window_hours` (default 24) | Fibers temporally adjacent to the anchor |

Use `temporal_range` for absolute windows ("last week"). Use `temporal_neighborhood` for relative windows around a known event ("everything that happened around the time we shipped X").

---

## Recipe 1 — Review the last week

**Use case**: weekly retro, standup prep, "what shipped this week?".

### MCP

```python
nmem_causal(
  action="temporal_range",
  start="2026-04-25T00:00:00",
  end="2026-05-02T00:00:00",
  limit=50,
)
```

### CLI

```bash
nmem causal temporal_range \
  --start 2026-04-25T00:00:00 \
  --end 2026-05-02T00:00:00 \
  --limit 50
```

### Returns

```json
{
  "start": "2026-04-25T00:00:00",
  "end": "2026-05-02T00:00:00",
  "fibers": [
    { "id": "fbr_...", "summary": "...", "time_start": "..." }
  ],
  "count": 12
}
```

**Tip**: Pair with `nmem_recall` afterwards if you need to chase a specific thread — `temporal_range` gives you the surface, recall gives you the depth.

---

## Recipe 2 — Memories around a specific decision

**Use case**: "show me everything that happened around the time we picked PostgreSQL". You have the decision's `fiber_id` (from a prior recall or `nmem_remember` response), but you want the surrounding context.

### MCP

```python
nmem_causal(
  action="temporal_neighborhood",
  fiber_id="fbr_pg_decision_xyz",
  window_hours=12,
  limit=20,
)
```

### CLI

```bash
nmem causal temporal_neighborhood \
  --fiber-id fbr_pg_decision_xyz \
  --window-hours 12 \
  --limit 20
```

The window is symmetric — 12 hours before and 12 hours after `time_start` of the anchor fiber.

**Tip**: Use a tight window (1–6 hours) when investigating a specific work session. Use a wide window (24–72 hours) when reconstructing a multi-day arc.

---

## Recipe 3 — Sibling events that happened in parallel

**Use case**: "I'm looking at this bugfix — what else was happening at the same time? Did it correlate with anything?"

This is `temporal_neighborhood` with a deliberately narrow window (1–4 hours).

```python
nmem_causal(
  action="temporal_neighborhood",
  fiber_id="fbr_bugfix_xyz",
  window_hours=2,
  limit=10,
)
```

If multiple unrelated decisions happened in the same 2-hour window, that's a signal worth following — especially if you're root-causing a regression and looking for the *real* trigger.

---

## Recipe 4 — Audit trail: what led up to an outage

**Use case**: incident postmortem. You know when the alert fired (e.g., 2026-04-30 14:30 UTC). You want everything in the 6 hours before.

```python
nmem_causal(
  action="temporal_range",
  start="2026-04-30T08:30:00",
  end="2026-04-30T14:30:00",
  limit=100,
)
```

Then chain with `nmem_causal action="trace"` on any suspicious fiber to follow `CAUSED_BY` / `LEADS_TO` synapses.

---

## Recipe 5 — Combining temporal + causal

**Use case**: "Show me decisions made last quarter, then trace which ones led to current open issues."

Two-step pattern:

```python
# Step 1 — get last quarter's decisions
range_result = nmem_causal(
  action="temporal_range",
  start="2026-01-01T00:00:00",
  end="2026-04-01T00:00:00",
  limit=100,
)

# Step 2 — for each decision fiber, trace forward
for fiber in range_result["fibers"]:
    trace = nmem_causal(
        action="trace",
        seed=fiber["anchor_neuron_id"],
        direction="forward",
    )
    # ... process trace
```

This is the canonical "long-tail audit" pattern. It works because `temporal_range` is cheap (single index scan) and `trace` is bounded per call.

---

## Common pitfalls

- **Mixing timezones.** Always pass UTC ISO-8601 (`2026-05-02T00:00:00`). Naive datetimes are interpreted as UTC; aware datetimes are converted before the query.
- **Forgetting `start <= end`.** The handler returns `{"error": "start must be <= end"}` rather than swapping silently.
- **Window too wide on `temporal_neighborhood`.** Default is 24 hours; max is 168 (1 week). For long arcs use `temporal_range` with explicit dates instead.
- **Limit not high enough.** Default `limit=50`; bump to 100–200 only when you genuinely need a wide sweep — the response cost grows linearly.

## Related

- [`nmem_causal` action reference](../api/mcp-tools.md#nmem_causal) — full parameter table
- [Cognitive Reasoning](cognitive-reasoning.md) — how causal traces interact with spreading activation
- [Agent Memory Governance](agent-memory-governance.md) — when to *store* memories that you'll later retrieve temporally
