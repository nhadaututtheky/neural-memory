export const CARD_SUITS = {
  decision: { name: "The Architect", color: "#fbbf24", symbol: "◆", bg: "from-amber-900/40 to-amber-950/60" },
  error: { name: "The Shadow", color: "#ef4444", symbol: "♠", bg: "from-red-900/40 to-red-950/60" },
  insight: { name: "The Oracle", color: "#a78bfa", symbol: "♥", bg: "from-violet-900/40 to-violet-950/60" },
  fact: { name: "The Scholar", color: "#60a5fa", symbol: "♣", bg: "from-blue-900/40 to-blue-950/60" },
  workflow: { name: "The Engineer", color: "#34d399", symbol: "★", bg: "from-emerald-900/40 to-emerald-950/60" },
  concept: { name: "The Dreamer", color: "#22d3ee", symbol: "○", bg: "from-cyan-900/40 to-cyan-950/60" },
  entity: { name: "The Keeper", color: "#f59e0b", symbol: "△", bg: "from-amber-900/40 to-orange-950/60" },
  pattern: { name: "The Weaver", color: "#fb7185", symbol: "◇", bg: "from-rose-900/40 to-rose-950/60" },
  preference: { name: "The Compass", color: "#2dd4bf", symbol: "⊕", bg: "from-teal-900/40 to-teal-950/60" },
} as const

export const DEFAULT_SUIT = {
  name: "The Wanderer",
  color: "#a8a29e",
  symbol: "?",
  bg: "from-stone-900/40 to-stone-950/60",
} as const

export type CardSuitKey = keyof typeof CARD_SUITS

export interface CardSuit {
  readonly name: string
  readonly color: string
  readonly symbol: string
  readonly bg: string
}

export interface OracleCard {
  readonly id: string
  readonly title: string
  readonly content: string
  readonly suit: CardSuit
  readonly suitKey: CardSuitKey | "unknown"
  readonly activation: number
  readonly connectionCount: number
  readonly age: string
  readonly priority: number
  readonly createdAt: string
}

export type OracleMode = "daily" | "whatif" | "matchup"

export interface DailyReading {
  readonly past: OracleCard
  readonly present: OracleCard
  readonly future: OracleCard
  readonly interpretation: string
  readonly date: string
  readonly brainName: string
}

export interface WhatIfScenario {
  readonly decisions: readonly OracleCard[]
  readonly error: OracleCard
  readonly scenario: string
}

export interface MatchupState {
  readonly cardA: OracleCard
  readonly cardB: OracleCard
  readonly round: number
  readonly score: number
  readonly totalRounds: number
}
