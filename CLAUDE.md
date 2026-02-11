# NeuralMemory — AI Coding Standards

Project-level rules that Claude Code reads automatically.

## Architecture

```
src/neural_memory/
  core/       — Frozen dataclasses (Neuron, Synapse, Fiber, Brain)
  engine/     — Encoding, retrieval, consolidation, diagnostics
  storage/    — SQLite persistence (async via aiosqlite)
  mcp/        — MCP server (stdio transport for Claude Code)
  server/     — FastAPI REST API + dashboard
  integration/— External source adapters (Mem0, ChromaDB, Graphiti, …)
  safety/     — Sensitive content detection, freshness evaluation
  utils/      — Config, time utilities, simhash
```

## Immutability Rules

- **Never mutate function parameters.** Create new objects with `replace()` or spread (`{**d, key: val}`).
- **Core models are frozen dataclasses.** Use `replace()` to derive new instances.
- **No mutable default arguments** on frozen dataclasses (use `field(default_factory=...)`).

## Datetime Rules

- Use `utcnow()` from `neural_memory.utils.timeutils` — never `datetime.now()`.
- Store **naive UTC** datetimes for SQLite (no tzinfo).
- Never mix naive and timezone-aware datetimes.

## Security Rules

- **Parameterized SQL only.** Never f-string or `.format()` into SQL queries.
- **Validate all paths** with `Path.resolve()` + `is_relative_to()` before file access.
- **Never expose internal errors** to clients. Use generic messages in HTTP/MCP responses.
- **Never include available brain names, stack traces, or exception types** in error responses.
- **Bind to `127.0.0.1` by default**, not `0.0.0.0`.
- **CORS defaults** to localhost origins, not `["*"]`.

## Bounds Rules

- **Always cap server-side limits.** Use `min(user_limit, MAX)` for any user-provided limit.
- MCP context limit: max 200. Habits fiber fetch: max 1000.
- REST neuron list: max 1000. Encode content: max 100,000 chars.

## Testing Rules

- Minimum coverage: **70%** (enforced by CI).
- Test immutability: verify that functions don't mutate their inputs.
- Use `pytest-asyncio` with `asyncio_mode = "auto"`.

## Error Handling

- Never bare `except: pass`. Always log or re-raise.
- In MCP handlers: always `logger.error(...)` before returning error dicts.
- Migration errors: halt on non-benign errors (don't advance schema version).

## Type Safety Rules

- **Always use generic type params**: `dict[str, Any]` not bare `dict`, `list[str]` not bare `list`.
- **Mixin classes must declare protocol stubs** under `if TYPE_CHECKING` for all attributes/methods used from the composing class. Use `raise NotImplementedError` for stubs with non-`None` return types.
- **Narrow Optional types before use**: `assert x is not None` or `x = x or "default"` before passing `str | None` to a parameter typed `str`.
- **No stale `# type: ignore`**: remove when the underlying issue is fixed. Always use specific error codes (`# type: ignore[attr-defined]`), never bare `# type: ignore`.
- **CI must pass `mypy src/ --ignore-missing-imports` with 0 errors.** Never merge code that adds new mypy errors.
- **Avoid variable name reuse** across different types in the same scope — rename to avoid type conflicts (e.g. `storage` / `sqlite_storage`).

## Naming Conventions

- `type` parameter is accepted in **public API** (FastAPI query params, MCP tool args).
- Use `neuron_type` in **new internal code** to avoid shadowing the builtin.
- `snake_case` for functions/variables, `PascalCase` for classes, `SCREAMING_SNAKE` for constants.

## Commit Messages

Format: `<type>: <description>` — types: feat, fix, refactor, docs, test, chore, perf, ci
