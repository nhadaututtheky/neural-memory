import { useState } from "react"
import { Sparkle } from "@phosphor-icons/react"
import { ModeSelector } from "./components/ModeSelector"
import { DailyReading } from "./components/DailyReading"
import { WhatIfMode } from "./components/WhatIfMode"
import { MatchupMode } from "./components/MatchupMode"
import { useOracleData } from "./hooks/useOracleData"
import { useStats } from "@/api/hooks/useDashboard"
import type { OracleMode } from "./engine/types"
import { useTranslation } from "react-i18next"

export default function OraclePage() {
  const { t } = useTranslation()
  const [mode, setMode] = useState<OracleMode>("daily")
  const { cards, isLoading } = useOracleData()
  const { data: stats } = useStats()
  const brainName = stats?.active_brain ?? "default"

  if (isLoading) {
    return (
      <div className="space-y-6 p-6">
        <h1 className="font-display text-2xl font-bold">{t("oracle.title")}</h1>
        <p className="text-muted-foreground">{t("oracle.loading")}</p>
      </div>
    )
  }

  if (cards.length < 3) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 p-6 pt-24">
        <Sparkle className="size-12 text-muted-foreground/40" />
        <h2 className="font-display text-xl font-semibold text-muted-foreground">
          {t("oracle.needMore")}
        </h2>
        <p className="max-w-md text-center text-sm text-muted-foreground/70">
          {t("oracle.needMoreDesc")}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-bold">{t("oracle.title")}</h1>
        <ModeSelector mode={mode} onModeChange={setMode} />
      </div>

      <div className="flex flex-col items-center gap-8">
        {mode === "daily" && (
          <DailyReading cards={cards} brainName={brainName} />
        )}
        {mode === "whatif" && <WhatIfMode cards={cards} />}
        {mode === "matchup" && <MatchupMode cards={cards} />}
      </div>
    </div>
  )
}
