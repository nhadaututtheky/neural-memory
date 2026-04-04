import { useMemo, useState } from "react"
import { useTimeline, useDailyStats } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts"
import { useTranslation } from "react-i18next"
import { Brain, Stack, Lightning, CalendarBlank } from "@phosphor-icons/react"

const RANGE_OPTIONS = [
  { days: 7, key: "range7d" },
  { days: 30, key: "range30d" },
  { days: 90, key: "range90d" },
  { days: 365, key: "rangeAll" },
] as const

const NEURON_TYPE_COLORS: Record<string, string> = {
  entity: "var(--color-chart-1)",
  concept: "var(--color-chart-2)",
  action: "var(--color-chart-3)",
  intent: "var(--color-chart-4)",
  time: "var(--color-chart-5, #8b5cf6)",
  spatial: "#f59e0b",
  state: "#06b6d4",
  sensory: "#ec4899",
}

function KpiCard({
  label,
  value,
  icon: Icon,
  loading,
}: {
  label: string
  value: string | number
  icon: React.ElementType
  loading: boolean
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-4 p-5">
        <div className="flex size-10 items-center justify-center rounded-lg bg-primary/10">
          <Icon className="size-5 text-primary" aria-hidden="true" />
        </div>
        <div>
          <p className="text-xs text-muted-foreground">{label}</p>
          {loading ? (
            <Skeleton className="mt-1 h-6 w-16" />
          ) : (
            <p className="font-mono text-xl font-bold tracking-tight">
              {typeof value === "number" ? value.toLocaleString() : value}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default function TimelinePage() {
  const [days, setDays] = useState(30)
  const { data: dailyStats, isLoading: statsLoading } = useDailyStats(days)
  const { data: timeline, isLoading: timelineLoading } = useTimeline(20, 0)
  const { t } = useTranslation()

  const kpis = useMemo(() => {
    if (!dailyStats || dailyStats.length === 0) {
      return { today: 0, total: 0, avgPerDay: 0, activeDays: 0 }
    }
    const todayStr = new Date().toISOString().slice(0, 10)
    const todayEntry = dailyStats.find((d) => d.date === todayStr)
    const today = todayEntry ? todayEntry.fibers_created : 0
    const total = dailyStats.reduce((sum, d) => sum + d.fibers_created, 0)
    const activeDays = dailyStats.filter((d) => d.fibers_created > 0).length
    const avgPerDay = activeDays > 0 ? Math.round(total / activeDays) : 0
    return { today, total, avgPerDay, activeDays }
  }, [dailyStats])

  const neuronDistribution = useMemo(() => {
    if (!dailyStats) return []
    const totals: Record<string, number> = {}
    for (const day of dailyStats) {
      for (const [type, count] of Object.entries(day.neuron_types)) {
        totals[type] = (totals[type] ?? 0) + count
      }
    }
    return Object.entries(totals)
      .map(([type, count]) => ({ type, count }))
      .sort((a, b) => b.count - a.count)
  }, [dailyStats])

  const chartData = useMemo(() => {
    if (!dailyStats) return []
    return dailyStats.map((d) => ({
      date: d.date.slice(5), // "MM-DD"
      neurons: d.neurons_created,
      fibers: d.fibers_created,
      synapses: d.synapses_created,
    }))
  }, [dailyStats])

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-bold">{t("timeline.title")}</h1>

        {/* Range Picker */}
        <div className="flex gap-1 rounded-lg border border-border p-1">
          {RANGE_OPTIONS.map((opt) => (
            <button
              key={opt.days}
              onClick={() => setDays(opt.days)}
              className={`cursor-pointer rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                days === opt.days
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent/50"
              }`}
            >
              {t(`timeline.${opt.key}`)}
            </button>
          ))}
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard
          label={t("timeline.todayMemories")}
          value={kpis.today}
          icon={Brain}
          loading={statsLoading}
        />
        <KpiCard
          label={t("timeline.totalPeriod")}
          value={kpis.total}
          icon={Stack}
          loading={statsLoading}
        />
        <KpiCard
          label={t("timeline.avgPerDay")}
          value={kpis.avgPerDay}
          icon={Lightning}
          loading={statsLoading}
        />
        <KpiCard
          label={t("timeline.activeDays")}
          value={kpis.activeDays}
          icon={CalendarBlank}
          loading={statsLoading}
        />
      </div>

      {/* Activity Trend Chart */}
      <Card>
        <CardHeader>
          <CardTitle>{t("timeline.activityTrend")}</CardTitle>
        </CardHeader>
        <CardContent>
          {statsLoading ? (
            <Skeleton className="h-72 w-full" />
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorNeurons" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-chart-1)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--color-chart-1)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorFibers" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-chart-2)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--color-chart-2)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorSynapses" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-chart-4)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--color-chart-4)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.3} />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
                  interval="preserveStartEnd"
                />
                <YAxis
                  tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
                  width={40}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--color-card)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="neurons"
                  name={t("timeline.neurons")}
                  stroke="var(--color-chart-1)"
                  fill="url(#colorNeurons)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="fibers"
                  name={t("timeline.fibers")}
                  stroke="var(--color-chart-2)"
                  fill="url(#colorFibers)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="synapses"
                  name={t("timeline.synapses")}
                  stroke="var(--color-chart-4)"
                  fill="url(#colorSynapses)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Neuron Type Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>{t("timeline.neuronDistribution")}</CardTitle>
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : neuronDistribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={neuronDistribution} layout="vertical">
                  <XAxis
                    type="number"
                    tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
                  />
                  <YAxis
                    dataKey="type"
                    type="category"
                    tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
                    width={70}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--color-card)",
                      border: "1px solid var(--color-border)",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {neuronDistribution.map((entry) => (
                      <Cell
                        key={entry.type}
                        fill={NEURON_TYPE_COLORS[entry.type] ?? "var(--color-chart-1)"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-muted-foreground">{t("timeline.noEntries")}</p>
            )}
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>
              {t("timeline.recentActivity")}
              {timeline && (
                <span className="ml-2 text-sm font-normal text-muted-foreground">
                  {t("timeline.entries", { total: timeline.total.toLocaleString() })}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {timelineLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-14 w-full" />
                ))}
              </div>
            ) : timeline?.entries && timeline.entries.length > 0 ? (
              <div className="max-h-96 space-y-2 overflow-y-auto">
                {timeline.entries.map((entry) => (
                  <div
                    key={entry.id}
                    className="flex items-start gap-3 rounded-lg border border-border/50 p-3 transition-colors hover:bg-accent/30"
                  >
                    <Badge variant="outline" className="mt-0.5 shrink-0">
                      {entry.neuron_type}
                    </Badge>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm leading-relaxed line-clamp-2">
                        {entry.content}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {entry.created_at
                          ? new Date(entry.created_at).toLocaleString()
                          : "-"}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">{t("timeline.noEntries")}</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
