import { useMemo } from "react"
import { useTranslation } from "react-i18next"
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
  PieChart,
  Pie,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  Legend,
} from "recharts"
import { ProGate } from "@/components/common/ProGate"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  useDailyStats,
  useHealth,
  useToolStats,
  useEvolution,
  useStats,
} from "@/api/hooks/useDashboard"

/* ------------------------------------------------------------------ */
/*  Shared chart styles                                                */
/* ------------------------------------------------------------------ */

const TOOLTIP_STYLE = {
  backgroundColor: "var(--color-card)",
  border: "1px solid var(--color-border)",
  borderRadius: "8px",
  fontSize: "12px",
} as const

const AXIS_TICK = { fill: "var(--color-muted-foreground)", fontSize: 11 }

const TYPE_COLORS: Record<string, string> = {
  fact: "#3b82f6",
  decision: "#8b5cf6",
  error: "#ef4444",
  insight: "#f59e0b",
  preference: "#10b981",
  workflow: "#06b6d4",
  instruction: "#ec4899",
  concept: "#6366f1",
  entity: "#14b8a6",
  pattern: "#f97316",
}

const HEALTH_KEYS = [
  { key: "connectivity", label: "Connectivity" },
  { key: "diversity", label: "Diversity" },
  { key: "freshness", label: "Freshness" },
  { key: "consolidation_ratio", label: "Consolidation" },
  { key: "activation_efficiency", label: "Activation" },
  { key: "recall_confidence", label: "Recall" },
] as const

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

function ChartSkeleton() {
  return (
    <div className="flex h-[200px] items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
    </div>
  )
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
      {message}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Activity chart — neurons created per day (30d)                     */
/* ------------------------------------------------------------------ */

function ActivityChart() {
  const { data, isLoading } = useDailyStats(30)

  const chartData = useMemo(() => {
    if (!data) return []
    return data.map((d) => ({
      date: d.date.slice(5), // MM-DD
      neurons: d.neurons_created,
      fibers: d.fibers_created,
      synapses: d.synapses_created,
    }))
  }, [data])

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Memory Activity (30 days)</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <ChartSkeleton />
        ) : chartData.length === 0 ? (
          <EmptyState message="No activity data yet" />
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="gNeurons" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gFibers" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.3} />
              <XAxis dataKey="date" tick={AXIS_TICK} interval="preserveStartEnd" />
              <YAxis tick={AXIS_TICK} width={35} allowDecimals={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend iconSize={10} />
              <Area
                type="monotone"
                dataKey="neurons"
                name="Neurons"
                stroke="#3b82f6"
                fill="url(#gNeurons)"
                strokeWidth={2}
              />
              <Area
                type="monotone"
                dataKey="fibers"
                name="Fibers"
                stroke="#10b981"
                fill="url(#gFibers)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/*  Memory type distribution — donut chart                             */
/* ------------------------------------------------------------------ */

function TypeDistributionChart() {
  const { data, isLoading } = useDailyStats(30)

  const pieData = useMemo(() => {
    if (!data) return []
    const totals: Record<string, number> = {}
    for (const day of data) {
      for (const [type, count] of Object.entries(day.neuron_types)) {
        totals[type] = (totals[type] ?? 0) + count
      }
    }
    return Object.entries(totals)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10)
  }, [data])

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Memory Types</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <ChartSkeleton />
        ) : pieData.length === 0 ? (
          <EmptyState message="No type data yet" />
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={85}
                paddingAngle={2}
                label={({ name, percent }: { name?: string; percent?: number }) =>
                  `${name ?? ""} ${((percent ?? 0) * 100).toFixed(0)}%`
                }
                labelLine={false}
              >
                {pieData.map((entry) => (
                  <Cell
                    key={entry.name}
                    fill={TYPE_COLORS[entry.name] ?? "#64748b"}
                  />
                ))}
              </Pie>
              <Tooltip contentStyle={TOOLTIP_STYLE} />
            </PieChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/*  Brain health radar                                                 */
/* ------------------------------------------------------------------ */

function HealthRadar() {
  const { data, isLoading } = useHealth()

  const radarData = useMemo(() => {
    if (!data) return []
    return HEALTH_KEYS.map(({ key, label }) => ({
      metric: label,
      value: Math.round(((data[key] as number) ?? 0) * 100),
    }))
  }, [data])

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Brain Health</CardTitle>
          {data && (
            <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-bold">
              {data.grade} ({data.purity_score}%)
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <ChartSkeleton />
        ) : radarData.length === 0 ? (
          <EmptyState message="No health data" />
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
              <PolarGrid stroke="var(--color-border)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} />
              <Radar
                dataKey="value"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.25}
                strokeWidth={2}
              />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v?: number) => [`${v ?? 0}%`, "Score"]} />
            </RadarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/*  Top tools bar chart                                                */
/* ------------------------------------------------------------------ */

function TopToolsChart() {
  const { data, isLoading } = useToolStats(30)

  const chartData = useMemo(() => {
    if (!data?.summary.top_tools) return []
    return data.summary.top_tools
      .slice(0, 8)
      .map((t) => ({
        name: t.tool_name.replace(/^nmem_/, ""),
        calls: t.count,
        rate: Math.round(t.success_rate * 100),
      }))
  }, [data])

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Top Tools (30 days)</CardTitle>
          {data && (
            <span className="text-xs text-muted-foreground">
              {data.summary.total_events} total calls
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <ChartSkeleton />
        ) : chartData.length === 0 ? (
          <EmptyState message="No tool usage data" />
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.3} />
              <XAxis type="number" tick={AXIS_TICK} allowDecimals={false} />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                width={80}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(v?: number, name?: string) => [
                  name === "calls" ? (v ?? 0) : `${v ?? 0}%`,
                  name === "calls" ? "Calls" : "Success",
                ]}
              />
              <Bar dataKey="calls" name="Calls" fill="#06b6d4" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/*  Brain evolution metrics                                            */
/* ------------------------------------------------------------------ */

function EvolutionMetrics() {
  const { data, isLoading } = useEvolution()

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Brain Evolution</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartSkeleton />
        </CardContent>
      </Card>
    )
  }

  if (!data) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Brain Evolution</CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState message="No evolution data" />
        </CardContent>
      </Card>
    )
  }

  const metrics = [
    { label: "Proficiency", value: data.proficiency_index, color: "#3b82f6" },
    { label: "Plasticity", value: data.plasticity, color: "#10b981" },
    { label: "Density", value: data.knowledge_density, color: "#f59e0b" },
    { label: "Coherence", value: data.topology_coherence, color: "#8b5cf6" },
    { label: "Semantic", value: data.semantic_ratio, color: "#ec4899" },
  ]

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Brain Evolution</CardTitle>
          <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-bold capitalize">
            {data.proficiency_level}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {metrics.map((m) => (
            <div key={m.label} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{m.label}</span>
                <span className="font-mono font-bold">{(m.value * 100).toFixed(0)}%</span>
              </div>
              <div className="h-2 rounded-full bg-muted">
                <div
                  className="h-2 rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(m.value * 100, 100)}%`,
                    backgroundColor: m.color,
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Stage distribution */}
        {data.stage_distribution && (
          <div className="mt-4 grid grid-cols-4 gap-2 text-center">
            {(["short_term", "working", "episodic", "semantic"] as const).map((stage) => (
              <div key={stage} className="rounded-md bg-muted/50 p-2">
                <div className="text-lg font-bold font-mono">
                  {data.stage_distribution?.[stage] ?? 0}
                </div>
                <div className="text-[10px] text-muted-foreground capitalize">
                  {stage.replace("_", " ")}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/*  Summary KPI cards                                                  */
/* ------------------------------------------------------------------ */

function KpiRow() {
  const { data } = useStats()
  const { data: health } = useHealth()
  const { data: tools } = useToolStats(30)

  const kpis = [
    {
      label: "Neurons",
      value: data?.total_neurons ?? 0,
      icon: "N",
      color: "#3b82f6",
    },
    {
      label: "Synapses",
      value: data?.total_synapses ?? 0,
      icon: "S",
      color: "#10b981",
    },
    {
      label: "Fibers",
      value: data?.total_fibers ?? 0,
      icon: "F",
      color: "#f59e0b",
    },
    {
      label: "Health",
      value: health?.purity_score ?? 0,
      suffix: "%",
      icon: health?.grade ?? "-",
      color: "#8b5cf6",
    },
    {
      label: "Tool Calls",
      value: tools?.summary.total_events ?? 0,
      icon: "T",
      color: "#06b6d4",
    },
    {
      label: "Success Rate",
      value: tools ? Math.round(tools.summary.success_rate * 100) : 0,
      suffix: "%",
      icon: "R",
      color: "#10b981",
    },
  ]

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      {kpis.map((k) => (
        <div
          key={k.label}
          className="rounded-lg border border-border bg-card p-3 text-center"
        >
          <div
            className="mx-auto mb-1 flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold text-white"
            style={{ backgroundColor: k.color }}
          >
            {k.icon}
          </div>
          <div className="text-lg font-bold font-mono">
            {k.value.toLocaleString()}
            {k.suffix ?? ""}
          </div>
          <div className="text-[11px] text-muted-foreground">{k.label}</div>
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main Page                                                          */
/* ------------------------------------------------------------------ */

export default function VisualizePage() {
  const { t } = useTranslation()

  return (
    <ProGate label={t("license.pro_feature", "Pro Feature")}>
      <div className="space-y-6 p-6">
        <div>
          <h1 className="font-display text-2xl font-bold">
            {t("visualize.title", "Memory Insights")}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {t(
              "visualize.description",
              "Auto-generated analytics from your brain's memory graph.",
            )}
          </p>
        </div>

        {/* KPI summary row */}
        <KpiRow />

        {/* Charts grid — 2 columns on large, 1 on mobile */}
        <div className="grid gap-6 lg:grid-cols-2">
          <ActivityChart />
          <TypeDistributionChart />
          <HealthRadar />
          <TopToolsChart />
          <EvolutionMetrics />
        </div>
      </div>
    </ProGate>
  )
}
