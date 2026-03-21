# Progress Log

## 2026-03-18
- Full 8-dimension audit completed (Rune audit skill)
  - Verdict: WARNING, Overall health: 8.0/10
  - Findings: 0 CRITICAL, 3 HIGH, 8 MEDIUM, 5 LOW, 2 INFO
  - HIGH: input firewall gap on explicit nmem_remember, tool_handlers.py 2609 LOC, magic number duplication
  - Positives: zero ruff/mypy errors, 4361 tests, comprehensive security posture
  - Report: AUDIT-REPORT.md
  - ADR: .rune/adr/ADR-001-input-firewall-explicit-path-gap.md

## 2026-03-12
- v4.1.0 released: docs engine + chatbot + brain fix + CI docs check
  - Auto-generated MCP tool reference (44 tools) + CLI reference (66 commands)
  - Documentation chatbot: Gradio UI powered by ReflexPipeline (no LLM)
  - Brain lookup fallback: `get_brain()` falls back to `find_brain_by_name()` (fixes brain.v2 dupe bug)
  - CI docs freshness check: new `docs` job in GitHub Actions
  - Docs polish: orphan pages added to nav, cross-links, CLI Guide title rename
- Stale references audit: updated tool counts (39→44), schema (v22→v26), test counts (3500→3778+) across README, SKILL.md, mcp-server.md, ROADMAP.md, plugin.json
- CLAUDE.md: updated Brain Oracle architecture (added Phase 2+3 files), fixed suit mapping, status

## 2026-03-05
- Neurodungeon feature planned: roguelike dungeon crawler in dashboard
  - Full plan: .rune/neurodungeon-plan.md (4 phases, 20 tasks)
  - Phase 1: Foundation (types, dungeon-gen, canvas, movement, HUD, page)
  - Phase 2: Interaction (dialogue, combat, items, multi-floor, events)
  - Phase 3: Polish (effects, quiz, game over, start screen, sound)
  - Phase 4: Virality (share card, achievements, brain impact, daily)
- Fixed `nmem serve`: port conflict + non-editable install (static files missing)
- Dashboard rebuilt: `cd dashboard && npm run build`
- Reinstalled editable mode: `pip install --user -e .`

## 2026-03-03
- Rune onboard: initialized .rune/ directory
- CLAUDE.md: added API pitfalls + version bump checklist + fixed coverage threshold (70%→67%)
- MEMORY.md: split into topic files (versions.md, pitfalls.md)
- Neural Memory upgraded to v2.21.0 globally

## 2026-03-02
- v2.21.0 shipped: cross-language recall hint + Vietnamese detection + embedding docs
- /ship skill enhanced: added CHANGELOG step, version grep, better release template

## 2026-03-01
- v2.20.0 shipped: Gemini embedding, parallel anchors, comprehensive audit fixes
- PR #9 merged (embedding infrastructure)
