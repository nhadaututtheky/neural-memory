# Phase 5: DX Sprint — Developer Experience Polish

## Goal

Lower the barrier to entry. New users get value in < 5 minutes. Embedding setup is painless. First-run experience guides users through brain setup, embedding provider, and first memory.

## Tasks

- [ ] 5.1: One-line installer verification
  - Verify `pip install neural-memory` → `nmem --help` works on clean Python 3.11+
  - Verify `nmem init` creates config, brain, and tests basic encode/recall
  - Fix any issues found in fresh install path
  - Document minimum viable setup in README "Quick Start"

- [ ] 5.2: First-run wizard (CLI)
  - `nmem init --wizard` interactive mode
  - Steps: choose brain name → select embedding provider → test embedding → store first memory → recall it
  - Auto-detect available providers (sentence-transformers installed? Ollama running? Gemini key?)
  - Skip wizard if config already exists (idempotent)
  - Non-interactive fallback: `nmem init --defaults` (no prompts)

- [ ] 5.3: Embedding auto-setup
  - `nmem setup embeddings` command
  - Lists available providers with install instructions
  - For Gemini: prompt for API key, validate with test call, save to config
  - For Ollama: check if running, pull model if needed
  - For sentence-transformers: check if installed, suggest pip install
  - Show estimated resource usage (disk, RAM) for each provider

- [ ] 5.4: Health check improvements
  - `nmem doctor` command — comprehensive system check
  - Checks: Python version, dependencies, config validity, brain accessibility,
    embedding provider, storage integrity, schema version
  - Output: green/yellow/red status per check, actionable fix suggestions
  - Reuse existing `nmem_health` logic where possible

- [ ] 5.5: Error messages improvement
  - Audit all user-facing error messages in CLI + MCP
  - Replace cryptic errors with actionable messages:
    - "No embedding provider" → "Run `nmem setup embeddings` to configure"
    - "Brain not found" → "Run `nmem brain create <name>` or `nmem init`"
    - "Schema mismatch" → "Run `nmem upgrade` to migrate your database"
  - Add `--verbose` flag for debug-level output

- [ ] 5.6: Tests
  - Fresh install simulation (temp dir, no prior config)
  - Wizard flow (mock stdin for interactive)
  - Embedding auto-setup for each provider
  - Doctor command output for healthy + broken states
  - Error message coverage for common failure modes

## Acceptance Criteria

- [ ] Fresh `pip install neural-memory && nmem init` works without errors
- [ ] `nmem init --wizard` guides through full setup in < 2 minutes
- [ ] `nmem setup embeddings` configures provider without reading docs
- [ ] `nmem doctor` identifies and explains all common issues
- [ ] No error message says "internal error" or shows a stack trace to user

## Files Touched

### New
- `src/neural_memory/cli/wizard.py`
- `src/neural_memory/cli/doctor.py`
- `tests/unit/test_dx_wizard.py`

### Modified
- `src/neural_memory/cli/main.py` — add wizard, doctor, setup commands
- `src/neural_memory/cli/embedding_setup.py` — if exists, enhance; else create
- `src/neural_memory/mcp/tool_handlers.py` — improved error messages
- `src/neural_memory/unified_config.py` — wizard config creation

## Dependencies

- Independent of Phases 0-4 (can run in parallel)
- Benefits from Phase 2 (source setup in wizard) but not required
