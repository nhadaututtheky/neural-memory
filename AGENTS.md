# NeuralMemory — AI Coding Standards

Project-level rules that Codex reads automatically.

## Architecture

```
src/neural_memory/
  core/       — Frozen dataclasses (Neuron, Synapse, Fiber, Brain)
  engine/     — Encoding, retrieval, consolidation, diagnostics
  storage/    — SQLite persistence (async via aiosqlite)
  mcp/        — MCP server (stdio transport for Codex)
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

- Minimum coverage: **67%** (enforced by CI via `fail_under = 67` in pyproject.toml).
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

## Migration Rules

When changing config formats, storage paths, or schema:

- **Test the upgrade path**: existing data (old format) → new code must work seamlessly.
- **Test fresh install**: no prior data → new code creates correct defaults.
- **Test mixed state**: partial migration (e.g. `config.json` exists but `config.toml` doesn't) must resolve correctly.
- **Never silently discard user state.** If a legacy file contains `current_brain = "work"`, the migration must carry it forward — not reset to `"default"`.
- **Write migration tests before merging.** Every `load()` / `migrate()` function needs tests for: old→new, fresh, already-migrated, corrupt input, and invalid values.
- **Log migrations.** Use `logger.info()` when migrating data so users can diagnose issues.

## Pre-release Smoke Test

Before tagging a release, verify these scenarios manually or via integration tests:

1. **Fresh install**: delete data dir, run MCP server, confirm default brain created.
2. **Upgrade from previous version**: keep old data dir intact, run new code, confirm brain name and memories preserved.
3. **Brain switch round-trip**: switch brain via CLI → confirm MCP reads the new brain → switch back → confirm again.
4. **Config file conflicts**: both `config.json` and `config.toml` exist → confirm `config.toml` wins.
5. **Recall after upgrade**: store a memory, upgrade, recall it — confirm it's still there with correct brain context.

## API Pitfalls

| API | Gotcha |
|-----|--------|
| `Neuron.create()` | Uses `type=` not `neuron_type=` |
| `Synapse.create()` | Uses `type=` not `synapse_type=` |
| `find_neurons()` | Uses `content_exact=` param. NO `brain_id` param |
| `get_synapses()` | NOT `find_synapses()`. Uses `source_id=`, `target_id=`, `type=` |
| `add_neuron()`, `add_synapse()` | NO `brain_id` param — uses `set_brain` context |
| `tool_events` FK | Test fixture must insert brain with ALL required columns |

## Version Bump Checklist

ALL files must be updated when bumping version:

1. `pyproject.toml` → `version = "x.y.z"`
2. `src/neural_memory/__init__.py` → `__version__ = "x.y.z"`
3. `.Codex-plugin/plugin.json` → `"version": "x.y.z"`
4. `.Codex-plugin/marketplace.json` → `"version": "x.y.z"` (TWO occurrences)
5. `tests/unit/test_health_fixes.py` → `assert neural_memory.__version__ == "x.y.z"`
6. `tests/unit/test_markdown_export.py` → `"version": "x.y.z"` in fixture

## Pre-Ship Verification (MANDATORY)

Before EVERY release, run:

```bash
python scripts/pre_ship.py        # Verify all checks pass
python scripts/pre_ship.py --fix  # Auto-fix ruff issues first
```

This checks: version consistency (6 files), ruff lint+format, mypy, import smoke test, fast unit tests, CHANGELOG entry, OpenClaw plugin consistency.

**Do NOT tag or release if pre_ship.py fails.**

## Commit Messages

Format: `<type>: <description>` — types: feat, fix, refactor, docs, test, chore, perf, ci

## Brain Oracle (Dashboard Feature)

Card-based memory fortune teller. Reads AI memories as tarot-style cards with 9 suits mapped from neuron types.

### Architecture

```
dashboard/src/features/oracle/
  engine/
    types.ts          — Card suits, OracleCard, OracleMode, reading types
    card-generator.ts — neuronsToCards(): GraphNeuron[] → OracleCard[]
    reading-engine.ts — Reading logic: daily seed, card selection, spread layouts
    templates.ts      — Template interpolation for What If collision text
  components/
    CardBack.tsx      — CSS geometric mandala pattern (conic-gradient)
    CardFace.tsx      — Suit-colored gradient, symbol, content, stats
    FlipCard.tsx      — 3D CSS flip (perspective + rotateY, 600ms)
    ModeSelector.tsx  — Daily/WhatIf/Matchup mode tabs
    DailyReading.tsx  — Past/Present/Future 3-card spread
    WhatIfMode.tsx    — Collision mode: 2 cards collide → outcome
    MatchupMode.tsx   — Compare 2 cards side-by-side
    ShareButton.tsx   — Export reading as PNG image
  hooks/
    useOracleData.ts  — useGraph(500) → neuronsToCards(), memoized
    useDaily.ts       — Daily reading persistence (localStorage)
  utils/
    share-image.ts    — Canvas-based PNG generation for sharing
  OraclePage.tsx      — Main page: mode selector + card layouts
```

### Key Conventions

- **TS 5.9 compat**: Use `as const` objects + derived types, NOT TypeScript enums
- **0 new dependencies**: Pure CSS animations + React
- **9 card suits**: decision→Architect, error→Shadow, insight→Oracle, fact→Scholar, workflow→Engineer, concept→Dreamer, entity→Keeper, pattern→Weaver, preference→Compass, unknown→Wanderer
- **3 modes**: Daily Reading (Past/Present/Future), What If (collision), Matchup (compare 2)
- **Data**: Uses existing `/api/graph?limit=500`, card stats from neuron activation/connections/age
- **Flip animation**: CSS-only 3D with `backface-visibility: hidden`, `autoFlipDelay` for stagger

### Plan & Status

Full plan: `.rune/plan-brain-oracle.md`
All 3 phases complete: Foundation, Game Modes, Polish (share PNG, daily persistence, i18n).
