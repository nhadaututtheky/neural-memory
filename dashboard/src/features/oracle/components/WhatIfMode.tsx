import { useState, useCallback, useRef } from "react"
import { Shuffle } from "@phosphor-icons/react"
import { FlipCard } from "./FlipCard"
import { generateWhatIf } from "../engine/reading-engine"
import type { OracleCard, WhatIfScenario } from "../engine/types"
import { useTranslation } from "react-i18next"

interface WhatIfModeProps {
  cards: OracleCard[]
}

export function WhatIfMode({ cards }: WhatIfModeProps) {
  const { t } = useTranslation()
  const [scenario, setScenario] = useState<WhatIfScenario | null>(() =>
    generateWhatIf(cards),
  )
  const [allFlipped, setAllFlipped] = useState(false)
  const flipCount = useRef(0)

  const reshuffle = useCallback(() => {
    setScenario(generateWhatIf(cards, Date.now()))
    setAllFlipped(false)
    flipCount.current = 0
  }, [cards])

  const handleFlip = useCallback(() => {
    flipCount.current += 1
    if (flipCount.current >= 3) setAllFlipped(true)
  }, [])

  if (!scenario) return null

  const allCards = [...scenario.decisions, scenario.error]

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="flex items-center gap-3">
        <p className="text-sm text-muted-foreground">
          {t("oracle.whatifHint")}
        </p>
        <button
          onClick={reshuffle}
          className="flex cursor-pointer items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          aria-label={t("oracle.reshuffle")}
        >
          <Shuffle className="size-3.5" />
          {t("oracle.reshuffle")}
        </button>
      </div>

      <div className="flex flex-wrap justify-center gap-6">
        {allCards.map((card, i) => (
          <div key={`slot-${i}`} className="flex flex-col items-center gap-2">
            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {i < 2 ? t("oracle.memory") + ` ${i + 1}` : t("oracle.wildcard")}
            </span>
            <FlipCard
              card={card}
              onFlip={handleFlip}
              className="h-[340px] w-[240px]"
            />
          </div>
        ))}
      </div>

      {/* Scenario text after all cards flipped */}
      {allFlipped && (
        <div className="max-w-2xl animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5">
            <p className="text-center text-sm leading-relaxed text-foreground/80 italic">
              {scenario.scenario}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
