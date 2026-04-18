import { useTranslation } from "react-i18next"
import { useCountUp } from "../hooks/useCountUp"

interface StatsBarProps {
  neurons: number
  synapses: number
  activeNeurons: number
  totalPulses: number
}

export function StatsBar({ neurons, synapses, activeNeurons, totalPulses }: StatsBarProps) {
  const { t } = useTranslation()
  const n = useCountUp(neurons)
  const s = useCountUp(synapses)
  const a = useCountUp(activeNeurons, 300)
  const p = useCountUp(totalPulses, 300)

  return (
    <div
      className="pointer-events-none absolute bottom-3 left-3 flex items-center gap-3 rounded-md border border-border/60 bg-card/85 px-3 py-2 text-[11px] font-medium shadow-sm backdrop-blur-md"
      aria-label={t("livingBrain.stats.label")}
      role="status"
    >
      <Stat label={t("livingBrain.stats.neurons")} value={n} />
      <Divider />
      <Stat label={t("livingBrain.stats.synapses")} value={s} />
      <Divider />
      <Stat label={t("livingBrain.stats.active")} value={a} accent />
      <Divider />
      <Stat label={t("livingBrain.stats.pulses")} value={p} />
    </div>
  )
}

function Stat({ label, value, accent }: { label: string; value: number; accent?: boolean }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span
        className={`font-mono tabular-nums ${accent ? "text-primary" : "text-foreground"}`}
      >
        {value.toLocaleString()}
      </span>
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
    </div>
  )
}

function Divider() {
  return <span aria-hidden className="h-3 w-px bg-border/80" />
}
