import type { OracleCard, DailyReading, WhatIfScenario, MatchupState } from "./types"
import { WHAT_IF_TEMPLATES, DAILY_INTERPRETATIONS, MATCHUP_PROMPTS } from "./templates"

// Deterministic seed from date + brain name
function dailySeed(date: string, brainName: string): number {
  let hash = 0
  const str = `${date}:${brainName}`
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i)
    hash |= 0
  }
  return Math.abs(hash)
}

// Combine two seeds into a distinct hash
function combineSeed(a: number, b: number): number {
  let h = (a * 2654435761) ^ b
  h = Math.imul(h ^ (h >>> 16), 2246822507)
  h = Math.imul(h ^ (h >>> 13), 3266489909)
  return (h ^ (h >>> 16)) >>> 0
}

// Seeded pseudo-random (mulberry32)
function seededRandom(seed: number): () => number {
  let s = seed | 0
  return () => {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Pick N unique items from array using seeded random
function pickN<T>(items: readonly T[], n: number, rand: () => number): T[] {
  const pool = [...items]
  const picked: T[] = []
  for (let i = 0; i < n && pool.length > 0; i++) {
    const idx = Math.floor(rand() * pool.length)
    picked.push(pool[idx])
    pool.splice(idx, 1)
  }
  return picked
}

// Pick a template string using random
function pickTemplate(templates: readonly string[], rand: () => number): string {
  if (templates.length === 0) return ""
  return templates[Math.floor(rand() * templates.length)]
}

// Interpolate {past}, {present}, {future}, {cardA}, {cardB}, {suit} etc.
function interpolate(
  template: string,
  vars: Record<string, string>,
): string {
  return Object.entries(vars).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, value),
    template,
  )
}

export function generateDailyReading(
  cards: readonly OracleCard[],
  brainName: string,
  date?: string,
): DailyReading | null {
  if (cards.length < 3) return null

  const today = date ?? new Date().toISOString().slice(0, 10)
  const seed = dailySeed(today, brainName)
  const rand = seededRandom(seed)

  // Sort by activation descending, then pick from top 60% (min 3)
  const sorted = [...cards].sort((a, b) => b.activation - a.activation)
  const topPool = sorted.slice(0, Math.max(Math.ceil(sorted.length * 0.6), 3))
  const picked = pickN(topPool, 3, rand)

  // Guard: pickN must return exactly 3
  if (picked.length < 3) return null
  const [past, present, future] = picked

  const interpretation = interpolate(
    pickTemplate(DAILY_INTERPRETATIONS, rand),
    {
      past: past.suit.name,
      present: present.suit.name,
      future: future.suit.name,
      pastContent: past.content,
      presentContent: present.content,
      futureContent: future.content,
    },
  )

  return { past, present, future, interpretation, date: today, brainName }
}

export function generateWhatIf(
  cards: readonly OracleCard[],
  seed?: number,
): WhatIfScenario | null {
  if (cards.length < 3) return null

  const rand = seededRandom(seed ?? Date.now())

  // Pick 2 decisions + 1 wildcard
  const decisions = pickN(cards, 2, rand)
  if (decisions.length < 2) return null

  const remaining = cards.filter((c) => !decisions.includes(c))
  const wildcardPick = pickN(remaining.length > 0 ? remaining : cards, 1, rand)
  if (wildcardPick.length === 0) return null
  const wildcard = wildcardPick[0]

  const scenario = interpolate(pickTemplate(WHAT_IF_TEMPLATES, rand), {
    cardA: decisions[0].content,
    cardB: decisions[1].content,
    suitA: decisions[0].suit.name,
    suitB: decisions[1].suit.name,
    wildcard: wildcard.content,
    wildcardSuit: wildcard.suit.name,
  })

  return { decisions, error: wildcard, scenario }
}

export function generateMatchup(
  cards: readonly OracleCard[],
  round = 1,
  totalRounds = 5,
  previousPicks: readonly string[] = [],
  seed?: number,
): MatchupState | null {
  if (cards.length < 2) return null

  // Use combineSeed to avoid collision between rounds in same millisecond
  const baseSeed = seed ?? Date.now()
  const rand = seededRandom(combineSeed(baseSeed, round * 7919))
  const available = cards.filter((c) => !previousPicks.includes(c.id))
  const pool = available.length >= 2 ? available : cards

  const picked = pickN(pool, 2, rand)
  if (picked.length < 2) return null
  return { cardA: picked[0], cardB: picked[1], round, score: 0, totalRounds }
}

export function getMatchupPrompt(
  cardA: OracleCard,
  cardB: OracleCard,
  seed?: number,
): string {
  // Use stable seed from card IDs so prompt doesn't change on re-render
  const stableSeed = seed ?? dailySeed(cardA.id, cardB.id)
  const rand = seededRandom(stableSeed)
  return interpolate(pickTemplate(MATCHUP_PROMPTS, rand), {
    suitA: cardA.suit.name,
    suitB: cardB.suit.name,
    contentA: cardA.content,
    contentB: cardB.content,
  })
}
