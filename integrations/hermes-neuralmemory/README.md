# NeuralMemory — Hermes Plugin

Faithful port of the [OpenClaw NeuralMemory plugin](https://github.com/nhadaututtheky/neural-memory/tree/main/integrations/neuralmemory) (v1.17.0, TypeScript) to the Hermes plugin API (Python).

Brain-inspired persistent memory for AI agents: neurons, synapses, fibers,
spreading-activation recall, Hebbian learning, memory decay, contradiction
detection — zero LLM dependency.

```
Hermes Agent
    |
    v
NeuralMemory Plugin (this package)
    |  Spawns + manages lifecycle
    v
nmem-mcp (Python MCP server, stdio transport)
    |
    v
~/.neuralmemory/brains/<brain>.db (SQLite)
```

## Prerequisites

```bash
pip install neural-memory   # Python 3.11+
nmem-mcp --help             # verify
```

## Install

1. Copy this directory to `~/.hermes/plugins/neuralmemory/`
   (on this machine: `G:\hermes\profiles\<profile>\plugins\neuralmemory\`).
2. Enable it in `config.yaml` (Hermes user plugins are opt-in):

```yaml
plugins:
  enabled:
    - neuralmemory
  entries:
    neuralmemory:
      config:
        pythonPath: "python"     # Windows default — NOT python3
        brain: "default"
        autoContext: true
        autoCapture: true
        autoFlush: true
        autoConsolidate: true
        contextDepth: 1
        maxContextTokens: 500
        timeout: 30000
        initTimeout: 90000
```

3. Restart Hermes. The 7 tools (`nmem_remember`, `nmem_recall`,
   `nmem_context`, `nmem_stats`, `nmem_health`, `memory_search`,
   `memory_get`) appear immediately; the MCP process connects in the
   background and tools auto-reconnect on first call.

## Port parity with the OpenClaw plugin

| OpenClaw feature | Hermes port |
|---|---|
| 5 fallback tools + 2 compat shims (sync registration) | identical — registered via `ctx.register_tool` |
| MCP stdio client (JSON-RPC, newline-delimited, 10 MB buffer cap, env allowlist, 30s/90s timeouts) | identical (`mcp_client.py`) |
| Singleton client per (pythonPath, brain) | `plugins.plugin_utils.lazy_singleton` |
| Schema normalization (strip constraint keys, integer→number, additionalProperties:false) | identical (`tools_proxy.py`) |
| `before_prompt_build` → systemPrompt + prependContext | `pre_llm_call` — tool instructions on first turn of each session; recall context every turn. Hermes injects into the user message to preserve prompt cache |
| `agent_end` → auto-capture (last 5 assistant msgs, sanitize, nmem_auto) | `post_llm_call` (fires on successful turns only — same guard) |
| `before_reset` → boundary flush | `on_session_reset` |
| `before_compaction` → emergency flush | no Hermes plugin-hook equivalent — dropped (documented) |
| `gateway_start` → consolidation | no Hermes plugin-hook equivalent — dropped (documented; run `nmem_consolidate` manually or via cron) |
| `stripPromptMetadata` / `sanitizeAutoCapture` regex pipelines | identical, same pass order |
| Config validation (brain name regex, depth/tokens/timeouts ranges) | identical |

### Known adaptation notes

- **Dynamic tool discovery**: Hermes freezes the tool list after
  `register()` returns (same as OpenClaw). The plugin connects in a
  background thread, runs `tools/list`, and LOGS the full discovered tool
  count. The 7 sync-registered tools auto-reconnect and proxy to MCP, so
  core memory works even before the background connect completes.
- **Auto-consolidation**: Hermes has no gateway-startup plugin hook; use a
  Hermes cron job calling `nmem_consolidate` if you want the upstream
  startup-consolidation behavior.

## Troubleshooting

- `"python" not found` → set `pythonPath` in the plugin config entry.
- `MCP initialize failed` → verify `python -m neural_memory.mcp` works
  manually; check `~/.hermes/logs/` for `[mcp stderr]` lines.
- Slow cold start → raise `initTimeout` (max 300000).
