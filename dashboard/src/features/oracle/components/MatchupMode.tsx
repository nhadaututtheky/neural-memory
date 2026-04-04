import { useState, useCallback, useEffect, useRef } from "react"
import { Trophy } from "@phosphor-icons/react"
import { FlipCard } from "./FlipCard"
import { generateMatchup, getMatchupPrompt } from "../engine/reading-engine"
import type { OracleCard, MatchupState } from "../engine/types"
import { useTranslation } from "react-i18next"

interface MatchupModeProps {
  cards: OracleCard[]
}

const TOTAL_ROUNDS = 5

// Normalize activation (0-10) and connectionCount to comparable 0-10 scale
function cardScore(card: OracleCard): number {
  const activationScore = card.activation // already 0-10
  const connectionScore = Math.min(card.connectionCount, 20) / 2 // cap at 20, scale to 0-10
  return activationScore + connectionScore
}

export function MatchupMode({ cards }: MatchupModeProps) {
  const { t } = useTranslation()
  const [state, setState] = useState<MatchupState | null>(() =>
    generateMatchup(cards, 1, TOTAL_ROUNDS),
  )
  const [score, setScore] = useState(0)
  const [picks, setPicks] = useState<string[]>([])
  const [chosen, setChosen] = useState<"A" | "B" | null>(null)
  const [gameOver, setGameOver] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const choose = useCallback(
    (side: "A" | "B") => {
      if (!state || chosen) return
      setChosen(side)

      const picked = side === "A" ? state.cardA : state.cardB
      const newScore = score + cardScore(picked)
      const newPicks = [...picks, state.cardA.id, state.cardB.id]

      setScore(newScore)
      setPicks(newPicks)

      // Advance to next round after short delay
      timerRef.current = setTimeout(() => {
        if (state.round >= TOTAL_ROUNDS) {
          setGameOver(true)
          return
        }
        const next = generateMatchup(
          cards,
          state.round + 1,
          TOTAL_ROUNDS,
          newPicks,
          Date.now(),
        )
        setState(next ? { ...next, score: newScore } : null)
        setChosen(null)
      }, 1200)
    },
    [state, chosen, score, picks, cards],
  )

  const restart = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setState(generateMatchup(cards, 1, TOTAL_ROUNDS))
    setScore(0)
    setPicks([])
    setChosen(null)
    setGameOver(false)
  }, [cards])

  if (!state) return null

  const prompt = getMatchupPrompt(state.cardA, state.cardB)

  if (gameOver) {
    return (
      <div className="flex flex-col items-center gap-6 animate-in fade-in duration-500">
        <Trophy className="size-12 text-amber-400" aria-hidden="true" />
        <h2 className="font-display text-2xl font-bold">
          {t("oracle.matchupComplete")}
        </h2>
        <p className="text-4xl font-bold text-primary">
          {Math.round(score)}
        </p>
        <p className="text-sm text-muted-foreground">
          {t("oracle.matchupScoreDesc")}
        </p>
        <button
          onClick={restart}
          className="cursor-pointer rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          {t("oracle.playAgain")}
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Round indicator */}
      <div className="flex items-center gap-3">
        <div className="flex gap-1.5">
          {Array.from({ length: TOTAL_ROUNDS }, (_, i) => (
            <div
              key={i}
              className={`size-2 rounded-full transition-colors ${
                i < state.round - 1
                  ? "bg-primary"
                  : i === state.round - 1
                    ? "bg-primary animate-pulse"
                    : "bg-muted"
              }`}
            />
          ))}
        </div>
        <span className="text-xs text-muted-foreground">
          {t("oracle.round", { current: state.round, total: TOTAL_ROUNDS })}
        </span>
      </div>

      {/* Prompt */}
      <p className="max-w-lg text-center text-sm text-muted-foreground italic">
        {prompt}
      </p>

      {/* Cards */}
      <div className="flex flex-wrap justify-center gap-8">
        {(["A", "B"] as const).map((side) => {
          const card = side === "A" ? state.cardA : state.cardB
          const isChosen = chosen === side
          const isOther = chosen !== null && chosen !== side

          return (
            <div key={`${state.round}-${side}`} className="flex flex-col items-center gap-3">
              <div
                className={`transition-all duration-500 ${
                  isChosen
                    ? "scale-105 ring-2 ring-primary ring-offset-2 ring-offset-background rounded-xl"
                    : isOther
                      ? "scale-95 opacity-40"
                      : ""
                }`}
              >
                <FlipCard
                  card={card}
                  autoFlipDelay={300 + (side === "B" ? 200 : 0)}
                  className="h-[340px] w-[240px]"
                />
              </div>
              {!chosen && (
                <button
                  onClick={() => choose(side)}
                  className="cursor-pointer rounded-lg bg-accent px-5 py-2 text-sm font-medium text-foreground transition-all hover:bg-primary hover:text-primary-foreground"
                >
                  {t("oracle.choose")}
                </button>
              )}
              {isChosen && (
                <span className="text-xs font-medium text-primary animate-in fade-in">
                  {t("oracle.chosen")}
                </span>
              )}
            </div>
          )
        })}
      </div>

      {/* Score */}
      <p className="text-xs text-muted-foreground">
        {t("oracle.score")}: <span className="font-mono font-bold text-foreground">{Math.round(score)}</span>
      </p>
    </div>
  )
}
