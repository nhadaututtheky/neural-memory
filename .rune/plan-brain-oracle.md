# Feature: Brain Oracle

## Overview

Card-based fortune-telling that turns AI memories into tarot-style cards. Replaces Neurodungeon.
3 modes: Daily Reading, What If, Memory Matchup. Zero new deps, pure CSS animations, shareable PNG cards.

## Architecture

```
dashboard/src/features/oracle/
  OraclePage.tsx
  engine/
    types.ts, card-generator.ts, reading-engine.ts, templates.ts
  components/
    CardFace.tsx, CardBack.tsx, FlipCard.tsx
    DailyReading.tsx, WhatIfMode.tsx, MatchupMode.tsx
    ModeSelector.tsx, ShareButton.tsx
  hooks/
    useOracleData.ts, useDaily.ts
  utils/
    share-image.ts
```

## Data Source

Uses existing `/api/graph?limit=500` — neurons become cards, synapses provide connection counts.
No new API endpoints needed.

## Card Suits (neuron type → archetype)

| Type | Name | Color | Symbol |
|------|------|-------|--------|
| decision | The Architect | Gold #fbbf24 | ◆ |
| error | The Shadow | Crimson #ef4444 | ♠ |
| insight | The Oracle | Purple #a78bfa | ♥ |
| fact | The Scholar | Blue #60a5fa | ♣ |
| workflow | The Engineer | Emerald #34d399 | ★ |
| concept | The Dreamer | Cyan #22d3ee | ○ |
| entity | The Keeper | Amber #f59e0b | △ |
| pattern | The Weaver | Rose #fb7185 | ◇ |
| preference | The Compass | Teal #2dd4bf | ⊕ |
| (unknown) | The Wanderer | Gray #a8a29e | ? |

## Phases

| # | Name | Status | Summary |
|---|------|--------|---------|
| 1 | Foundation | ⬚ Pending | Types, card gen, CardFace/Back/Flip, page + route swap |
| 2 | Game Modes | ⬚ Pending | DailyReading, WhatIf, Matchup, templates, reading engine |
| 3 | Polish | ⬚ Pending | Share PNG, responsive, i18n, animations, delete neurodungeon |

## Key Decisions

- No AI for "What If" text — pure template interpolation from neuron content
- Daily seed = `YYYY-MM-DD` + brain name hash — consistent per day per brain
- Card art = CSS-only geometric patterns (gradients + shapes per suit)
- Delete neurodungeon in Phase 3 (keep route working until oracle is complete)
