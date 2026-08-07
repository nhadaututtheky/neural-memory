# NeuralMemory — post-install

## 1. Install the memory engine

The plugin shells out to `python -m neural_memory.mcp`, so `neural-memory` must be
installed in the Python that `pythonPath` resolves to (Python 3.11+):

```bash
pip install neural-memory
nmem-mcp --help   # verify
```

If it's installed in a venv or a different interpreter, set `pythonPath` in the
plugin config entry below to that interpreter's `python`/`python.exe`.

## 2. Enable the plugin

```bash
hermes plugins enable neuralmemory
```

## 3. Optional config

Add to `config.yaml` (all values have sane defaults — skip if you don't need to change them):

```yaml
plugins:
  entries:
    neuralmemory:
      config:
        pythonPath: "python"
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

## 4. Restart

Restart Hermes. The 7 tools (`nmem_remember`, `nmem_recall`, `nmem_context`,
`nmem_stats`, `nmem_health`, `memory_search`, `memory_get`) appear immediately;
the MCP process connects in the background and tools auto-reconnect on first call.

Verify with:

```bash
hermes plugins list
```

## Troubleshooting

- `"python" not found` → set `pythonPath` to your interpreter.
- `MCP initialize failed` → verify `python -m neural_memory.mcp` works manually,
  then check `~/.hermes/logs/` for `[mcp stderr]` lines.
- Slow cold start → raise `initTimeout` (max 300000).
