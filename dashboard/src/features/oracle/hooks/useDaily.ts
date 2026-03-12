import { useState, useEffect } from "react"
import { generateDailyReading } from "../engine/reading-engine"
import type { OracleCard, DailyReading } from "../engine/types"

const STORAGE_KEY = "oracle-daily-reading"

interface StoredReading {
  date: string
  brainName: string
  reading: DailyReading
}

function getToday(): string {
  return new Date().toISOString().slice(0, 10)
}

function loadCached(brainName: string): DailyReading | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const stored: StoredReading = JSON.parse(raw)
    if (stored.date === getToday() && stored.brainName === brainName) {
      return stored.reading
    }
    return null
  } catch {
    return null
  }
}

function saveToCache(reading: DailyReading): void {
  const stored: StoredReading = {
    date: reading.date,
    brainName: reading.brainName,
    reading,
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(stored))
}

export function useDaily(
  cards: readonly OracleCard[],
  brainName: string,
): DailyReading | null {
  const [reading, setReading] = useState<DailyReading | null>(null)

  useEffect(() => {
    if (cards.length < 3) {
      setReading(null)
      return
    }

    // Check cache first
    const cached = loadCached(brainName)
    if (cached) {
      setReading(cached)
      return
    }

    // Generate new reading
    const newReading = generateDailyReading(cards, brainName)
    if (newReading) {
      saveToCache(newReading)
      setReading(newReading)
    }
  }, [cards, brainName])

  return reading
}
