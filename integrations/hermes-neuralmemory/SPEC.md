---
title: "NeuralMemory Hermes Plugin Port"
status: in-progress
created: 2026-08-06
updated: 2026-08-06
workflow: /build
scope: standard
---

# Spec: NeuralMemory OpenClaw Plugin → Hermes Port

## Source of truth (canon)
Upstream: https://github.com/nhadaututtheky/neural-memory — `integrations/neuralmemory/` (v1.17.0, TypeScript, OpenClaw).
Local reference copy: `openclaw-src/` in this workspace.

## Principle
FAITHFUL PORT, not rewrite. Every function, regex, default, and hook behavior maps 1:1 to the
OpenClaw source unless Hermes forces an adaptation. Adaptations are listed and justified below.

## Feature parity table (canon → port)

| # | OpenClaw feature | Hermes equivalent | Status |
|---|---|---|---|
| 1 | 5 fallback tools (nmem_remember/recall/context/stats/health) | `ctx.register_tool` | PORT 1:1 |
| 2 | 2 compat shims (memory_search, memory_get) | `ctx.register_tool` | PORT 1:1 |
| 3 | Dynamic tool discovery via `tools/list` | Discovered tools LOGGED (Hermes freezes tool list at register()) | ADAPTED |
| 4 | MCP stdio client (JSON-RPC, newline-delimited, 10MB buffer cap, env allowlist, timeouts 30s/90s) | `subprocess.Popen` + threading lock | PORT 1:1 |
| 5 | Singleton client pool keyed by (pythonPath, brain) | `plugins.plugin_utils.lazy_singleton` | PORT 1:1 |
| 6 | Schema normalization (strip constraint keys, integer→number, additionalProperties:false, ensure properties) | same algorithm | PORT 1:1 |
| 7 | `before_prompt_build` → systemPrompt (tool instructions) | `pre_llm_call` injection — Hermes injects into USER message to preserve prompt cache | ADAPTED (documented) |
| 8 | `before_prompt_build` → prependContext (auto-recall) | `pre_llm_call` `{"context": ...}` | PORT 1:1 |
| 9 | `agent_end` → auto-capture (last 5 assistant msgs, sanitize, nmem_auto) | `post_llm_call` (fires on success only) | PORT 1:1 |
| 10 | `before_compaction` → emergency flush | NO EQUIVALENT in Hermes VALID_HOOKS | DROPPED (documented) |
| 11 | `before_reset` → session boundary flush | `on_session_reset` | PORT 1:1 |
| 12 | `gateway_start` → consolidation (nmem_consolidate enrich) | NO EQUIVALENT hook in plugin API | DROPPED (documented; users can run nmem_consolidate manually) |
| 13 | `stripPromptMetadata` (7 regex passes) | same regexes | PORT 1:1 |
| 14 | `sanitizeAutoCapture` | same regexes, incl. the stricter `\]\s` neuron-bullet variant (canon index.ts:134) | PORT 1:1 |
| 15 | Config validation (brain name regex, ranges for depth/tokens/timeouts) | same validators | PORT 1:1 |
| 16 | `service.stop()` → close MCP + evict pool entry | `on_session_end` hook → `mcp.close()` + pool eviction | PORT 1:1 (added after reviewer MINOR-2) |
| 17 | `agent_end` reads only `ev.messages` | `post_llm_call` falls back to `assistant_response` kwarg when history has no assistant strings | ADAPTED (documented; Hermes guarantees the kwarg, history shape differs) |

## Hermes contract facts (verified in source)
- Manifest: `plugin.yaml` with `name`, `version`, `description`, `provides_tools`, `provides_hooks`
- Entry: `__init__.py` with `register(ctx)` (sync)
- Tool handler signature: `handler(params, **kwargs)` → JSON string
- Hook signatures: `pre_llm_call(session_id, user_message, conversation_history, is_first_turn, model, platform, **kwargs)` → `{"context": str}` | None; `post_llm_call(session_id, user_message, assistant_response, conversation_history, model, platform, **kwargs)` → ignored
- Thread safety: `plugins.plugin_utils.lazy_singleton` (Hermes is multi-threaded)
- Plugin config: `plugins.entries.neuralmemory` in config.yaml, read via `hermes_cli.config.load_config()`
- Opt-in: `plugins.enabled: [neuralmemory]` in config.yaml
- MCP child env: ALLOWED_ENV_KEYS allowlist ported verbatim (least privilege)

## Files
```
neuralmemory-hermes/
├── plugin.yaml          # manifest
├── __init__.py          # register(ctx): tools + hooks
├── mcp_client.py        # NeuralMemoryMcpClient (port of mcp-client.ts)
├── tools_proxy.py       # schemas, normalization, fallback/compat tools (port of tools.ts)
├── hooks.py             # stripPromptMetadata, sanitizeAutoCapture, hook callbacks (port of index.ts)
├── config.py            # resolveConfig + defaults (port of index.ts config section)
├── openclaw-src/        # READ-ONLY reference (do not ship)
└── SPEC.md / PLAN.md
```

## Non-goals
- No MemoryProvider subclass (separate exclusive-slot path; would replace Hermes' built-in memory entirely — different blast radius; not what canon does)
- No from-scratch features
- No changes to upstream repo
