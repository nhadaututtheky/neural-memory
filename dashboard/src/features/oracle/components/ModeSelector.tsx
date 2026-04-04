import { Sparkle, Shuffle, Sword } from "@phosphor-icons/react"
import type { OracleMode } from "../engine/types"
import { useTranslation } from "react-i18next"

const MODES: { key: OracleMode; labelKey: string; icon: typeof Sparkle }[] = [
  { key: "daily", labelKey: "oracle.dailyReading", icon: Sparkle },
  { key: "whatif", labelKey: "oracle.whatif", icon: Shuffle },
  { key: "matchup", labelKey: "oracle.matchup", icon: Sword },
]

interface ModeSelectorProps {
  mode: OracleMode
  onModeChange: (mode: OracleMode) => void
}

export function ModeSelector({ mode, onModeChange }: ModeSelectorProps) {
  const { t } = useTranslation()
  return (
    <div className="flex gap-2">
      {MODES.map(({ key, labelKey, icon: Icon }) => (
        <button
          key={key}
          onClick={() => onModeChange(key)}
          className={`flex cursor-pointer items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
            mode === key
              ? "bg-primary/15 text-primary ring-1 ring-primary/30"
              : "text-muted-foreground hover:bg-accent hover:text-foreground"
          }`}
        >
          <Icon className="size-4" />
          {t(labelKey)}
        </button>
      ))}
    </div>
  )
}
