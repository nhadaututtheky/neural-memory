import type { TierDistribution } from "@/api/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTranslation } from "react-i18next"
import { Stack } from "@phosphor-icons/react"

interface TierDistributionCardProps {
  data: TierDistribution
}

function TierBar({ label, count, total, color }: {
  label: string
  count: number
  total: number
  color: string
}) {
  const pct = total > 0 ? (count / total) * 100 : 0

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="flex items-center gap-2">
          <span
            className="inline-block size-2.5 rounded-full"
            style={{ backgroundColor: color }}
            aria-hidden="true"
          />
          {label}
        </span>
        <span className="font-mono font-semibold tabular-nums">{count}</span>
      </div>
      <div className="h-2 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

export function TierDistributionCard({ data }: TierDistributionCardProps) {
  const { t } = useTranslation()

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Stack className="size-5" aria-hidden="true" />
          {t("storage.tierDistribution")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <TierBar
          label={t("storage.tierHot")}
          count={data.hot}
          total={data.total}
          color="#ef4444"
        />
        <TierBar
          label={t("storage.tierWarm")}
          count={data.warm}
          total={data.total}
          color="#f59e0b"
        />
        <TierBar
          label={t("storage.tierCold")}
          count={data.cold}
          total={data.total}
          color="#3b82f6"
        />
        <div className="pt-2 border-t text-sm text-muted-foreground">
          {t("storage.totalMemories")}: <span className="font-mono font-semibold text-foreground">{data.total}</span>
        </div>
      </CardContent>
    </Card>
  )
}
