import { useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"
import { useTranslation } from "react-i18next"
import {
  ArrowUp,
  ArrowDown,
  Archive,
  Stack,
  Lightning,
} from "@phosphor-icons/react"
import { useTierAnalytics, useTierHistory, type TierChangeEvent } from "./useTierAnalytics"

const TIER_COLORS = {
  hot: "#ef4444",
  warm: "#f59e0b",
  cold: "#3b82f6",
} as const

function VelocityCard({
  label,
  promoted,
  demoted,
  archived,
  loading,
}: {
  label: string
  promoted: number
  demoted: number
  archived: number
  loading: boolean
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {loading ? (
          <Skeleton className="h-16 w-full" />
        ) : (
          <div className="grid grid-cols-3 gap-3">
            <div className="flex items-center gap-2">
              <ArrowUp className="size-4 text-green-500" aria-hidden="true" />
              <div>
                <p className="font-mono text-lg font-bold tabular-nums">{promoted}</p>
                <p className="text-xs text-muted-foreground">Promoted</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <ArrowDown className="size-4 text-orange-500" aria-hidden="true" />
              <div>
                <p className="font-mono text-lg font-bold tabular-nums">{demoted}</p>
                <p className="text-xs text-muted-foreground">Demoted</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Archive className="size-4 text-blue-500" aria-hidden="true" />
              <div>
                <p className="font-mono text-lg font-bold tabular-nums">{archived}</p>
                <p className="text-xs text-muted-foreground">Archived</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function TierBadge({ tier }: { tier: string }) {
  const color =
    tier === "hot"
      ? "bg-red-500/15 text-red-500"
      : tier === "cold"
        ? "bg-blue-500/15 text-blue-500"
        : "bg-amber-500/15 text-amber-500"

  return (
    <span className={`inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-medium ${color}`}>
      {tier.toUpperCase()}
    </span>
  )
}

export default function TierAnalyticsPage() {
  const { t } = useTranslation()
  const { data: analytics, isLoading: analyticsLoading } = useTierAnalytics()
  const { data: history, isLoading: historyLoading } = useTierHistory(50)

  const chartData = useMemo(() => {
    if (!analytics?.breakdown_by_type) return []
    return Object.entries(analytics.breakdown_by_type).map(([type, tiers]) => ({
      type,
      hot: (tiers as Record<string, number>).hot ?? 0,
      warm: (tiers as Record<string, number>).warm ?? 0,
      cold: (tiers as Record<string, number>).cold ?? 0,
    }))
  }, [analytics])

  const events: TierChangeEvent[] = history?.events ?? []

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center gap-3">
        <Stack className="size-7 text-primary" aria-hidden="true" />
        <h1 className="font-display text-2xl font-bold">
          {t("tierAnalytics.title", "Tier Analytics")}
        </h1>
      </div>

      {/* Velocity KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <VelocityCard
          label={t("tierAnalytics.velocity7d", "Velocity (7 days)")}
          promoted={analytics?.velocity_7d?.promoted ?? 0}
          demoted={analytics?.velocity_7d?.demoted ?? 0}
          archived={analytics?.velocity_7d?.archived ?? 0}
          loading={analyticsLoading}
        />
        <VelocityCard
          label={t("tierAnalytics.velocity30d", "Velocity (30 days)")}
          promoted={analytics?.velocity_30d?.promoted ?? 0}
          demoted={analytics?.velocity_30d?.demoted ?? 0}
          archived={analytics?.velocity_30d?.archived ?? 0}
          loading={analyticsLoading}
        />
      </div>

      {/* Breakdown by Type Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Lightning className="size-5" aria-hidden="true" />
            {t("tierAnalytics.breakdownByType", "Tier Distribution by Memory Type")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {analyticsLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : chartData.length === 0 ? (
            <p className="text-center text-sm text-muted-foreground py-8">
              {t("tierAnalytics.noData", "No typed memories yet")}
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
                <XAxis
                  dataKey="type"
                  tick={{ fontSize: 12 }}
                  stroke="var(--muted-foreground)"
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fontSize: 12 }}
                  stroke="var(--muted-foreground)"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: "8px",
                    fontSize: "13px",
                  }}
                />
                <Legend />
                <Bar dataKey="hot" name="HOT" fill={TIER_COLORS.hot} radius={[4, 4, 0, 0]} />
                <Bar dataKey="warm" name="WARM" fill={TIER_COLORS.warm} radius={[4, 4, 0, 0]} />
                <Bar dataKey="cold" name="COLD" fill={TIER_COLORS.cold} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* Recent Tier Changes Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {t("tierAnalytics.recentChanges", "Recent Tier Changes")}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {historyLoading ? (
            <Skeleton className="h-48 w-full" />
          ) : events.length === 0 ? (
            <p className="text-center text-sm text-muted-foreground py-8">
              {t("tierAnalytics.noChanges", "No tier changes recorded yet")}
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-2 pr-4 font-medium">Memory</th>
                    <th className="pb-2 pr-4 font-medium">Type</th>
                    <th className="pb-2 pr-4 font-medium">Change</th>
                    <th className="pb-2 pr-4 font-medium">Reason</th>
                    <th className="pb-2 font-medium">When</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {events.map((ev, i) => (
                    <tr key={`${ev.fiber_id}-${ev.at}-${i}`} className="hover:bg-muted/50">
                      <td className="py-2 pr-4">
                        <code className="text-xs font-mono">{ev.fiber_id.slice(0, 12)}…</code>
                      </td>
                      <td className="py-2 pr-4 text-xs">{ev.memory_type}</td>
                      <td className="py-2 pr-4">
                        <span className="flex items-center gap-1">
                          <TierBadge tier={ev.from_tier} />
                          <span className="text-muted-foreground">→</span>
                          <TierBadge tier={ev.to_tier} />
                        </span>
                      </td>
                      <td className="py-2 pr-4 text-xs text-muted-foreground max-w-[200px] truncate">
                        {ev.reason}
                      </td>
                      <td className="py-2 text-xs text-muted-foreground whitespace-nowrap">
                        {ev.at ? new Date(ev.at).toLocaleDateString() : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
