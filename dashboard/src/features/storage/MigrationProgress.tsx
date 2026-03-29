import type { MigrationJobStatus } from "@/api/types"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"

interface MigrationProgressProps {
  job: MigrationJobStatus
}

function ProgressBar({
  label,
  done,
  total,
}: {
  label: string
  done: number
  total: number
}) {
  const pct = total > 0 ? Math.min((done / total) * 100, 100) : 0
  const isDone = total > 0 && done >= total

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-medium tabular-nums">
          {total > 0 ? `${done.toLocaleString()} / ${total.toLocaleString()}` : "—"}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500 ease-out",
            isDone ? "bg-green-500" : "bg-primary",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export function MigrationProgress({ job }: MigrationProgressProps) {
  const { t } = useTranslation()

  return (
    <div className="space-y-3">
      <ProgressBar
        label={t("storage.neurons")}
        done={job.neurons_done}
        total={job.neurons_total}
      />
      <ProgressBar
        label={t("storage.synapses")}
        done={job.synapses_done}
        total={job.synapses_total}
      />
      <ProgressBar
        label={t("storage.fibers")}
        done={job.fibers_done}
        total={job.fibers_total}
      />
    </div>
  )
}
