## Summary

Adds `integrations/hermes-neuralmemory/` — a faithful Python port of the OpenClaw NeuralMemory plugin (`integrations/neuralmemory/`, v1.17.0) to the [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin API.

Hermes agents currently have no NeuralMemory integration; this gives them the same brain-inspired persistent memory (neurons, synapses, fibers, spreading activation) the OpenClaw plugin provides, against the same `nmem-mcp` server and brain databases.

## Architecture

```
Hermes Agent → neuralmemory plugin (Python) → MCP stdio → nmem-mcp → ~/.neuralmemory/brains/<brain>.db
```

## What's included (parity with the OpenClaw plugin)

- **7 tools registered synchronously**: `nmem_remember`, `nmem_recall`, `nmem_context`, `nmem_stats`, `nmem_health` + compat shims `memory_search`, `memory_get`
- **MCP stdio client**: JSON-RPC 2.0 newline-delimited, 10 MB buffer cap, least-privilege env allowlist (identical key set), 30s request / 90s init timeouts, auto-reconnect on first tool call, thread-safe (Hermes is multi-threaded)
- **Singleton client** per (pythonPath, brain) via Hermes' official `lazy_singleton` helper
- **Schema normalization**: identical algorithm (strip constraint keywords, integer→number, `additionalProperties: false`, ensure `properties`)
- **Hooks**:
  - `pre_llm_call` ← `before_prompt_build` (tool instructions on session first turn + auto-recall context every turn)
  - `post_llm_call` ← `agent_end` (auto-capture: last 5 assistant messages, sanitized, `nmem_auto`)
  - `on_session_reset` ← `before_reset` (boundary flush)
- **Prompt hygiene**: `stripPromptMetadata` and `sanitizeAutoCapture` regex pipelines ported with identical passes and order
- **Config**: same keys/defaults/validation ranges (`pythonPath`, `brain`, `autoContext`, `autoCapture`, `autoFlush`, `autoConsolidate`, `contextDepth` 0–3, `maxContextTokens` 100–10000, `timeout` 5s–120s, `initTimeout` 10s–300s), read from `plugins.entries.neuralmemory.config`

## Documented adaptations (Hermes API differences)

| OpenClaw | Hermes port | Why |
|---|---|---|
| `before_prompt_build` systemPrompt | `pre_llm_call` user-message injection, first turn only | Hermes injects into the user message to preserve prompt caching |
| `before_compaction` flush | dropped | no Hermes plugin-hook equivalent |
| `gateway_start` consolidation | dropped (cron workaround documented) | no Hermes plugin-hook equivalent |
| dynamic tool registration | discovery logged, 7 stable tools registered | Hermes freezes the tool list after `register()` (same constraint as OpenClaw) |

## Verification performed

- `py_compile` clean on all modules
- Load test: 7 tools + 3 hooks registered via mock context
- **Live Hermes run** (`HERMES_PLUGINS_DEBUG=1 hermes chat`): plugin discovered, loaded, registered all 7 tools + 3 hooks from real config
- **E2E MCP round-trip**: connect → `tools/list` (63 tools discovered) → `nmem_remember` (fiber_id returned) → `nmem_recall` (memory retrieved) → `nmem_stats`
- Hook unit checks: both regex pipelines produce exact expected outputs; first-turn/later-turn injection paths verified
- Mechanical parity audit vs the TypeScript source: all checks pass

## Install (from plugin README)

1. `pip install neural-memory` (Python 3.11+)
2. Copy `integrations/hermes-neuralmemory/` → `~/.hermes/plugins/neuralmemory/`
3. Enable in `config.yaml`: `plugins.enabled: [neuralmemory]` + optional `plugins.entries.neuralmemory.config`
4. Restart Hermes
