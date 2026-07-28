import { useState, useMemo } from "react"
import { useToolStats } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import { useTranslation } from "react-i18next"

import { chartSeriesColor } from "@/lib/neuron-colors"

const RANGE_OPTIONS = [7, 14, 30, 90] as const

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatRate(rate: number): string {
  return `${Math.round(rate * 100)}%`
}

function getSuccessBadgeVariant(rate: number): "success" | "warning" | "destructive" {
  if (rate >= 0.95) return "success"
  if (rate >= 0.8) return "warning"
  return "destructive"
}

export default function ToolStatsPage() {
  const [days, setDays] = useState<number>(30)
  const { data, isLoading } = useToolStats(days)
  const { t } = useTranslation()

  const summary = data?.summary
  const daily = data?.daily ?? []

  // Aggregate daily data into per-date totals for the line chart
  const dailyTotals = useMemo(() => {
    const byDate = new Map<string, { date: string; count: number; successes: number }>()
    for (const entry of daily) {
      const existing = byDate.get(entry.date)
      if (existing) {
        existing.count += entry.count
        existing.successes += Math.round(entry.count * entry.success_rate)
      } else {
        byDate.set(entry.date, {
          date: entry.date,
          count: entry.count,
          successes: Math.round(entry.count * entry.success_rate),
        })
      }
    }
    return Array.from(byDate.values())
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((d) => ({
        date: d.date.slice(5), // MM-DD
        count: d.count,
        successRate: d.count > 0 ? Math.round((d.successes / d.count) * 100) : 0,
      }))
  }, [daily])

  // Top tools for horizontal bar chart
  const topTools = useMemo(() => {
    if (!summary?.top_tools) return []
    return summary.top_tools.slice(0, 10).map((tool) => ({
      ...tool,
      shortName: tool.tool_name.replace(/^nmem_/, ""),
    }))
  }, [summary])

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-bold">{t("toolStats.title")}</h1>
        <div className="flex gap-1">
          {RANGE_OPTIONS.map((d) => (
            <Button
              key={d}
              variant={days === d ? "default" : "outline"}
              size="sm"
              onClick={() => setDays(d)}
              className="cursor-pointer"
            >
              {d}d
            </Button>
          ))}
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {t("toolStats.totalEvents")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <p className="font-mono text-2xl font-bold">
                {(summary?.total_events ?? 0).toLocaleString()}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {t("toolStats.successRate")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <p className="font-mono text-2xl font-bold">
                {formatRate(summary?.success_rate ?? 0)}
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {t("toolStats.uniqueTools")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="font-mono text-2xl font-bold">
                {summary?.top_tools?.length ?? 0}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Top Tools Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t("toolStats.topTools")}</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-72 w-full" />
            ) : topTools.length === 0 ? (
              <p className="py-12 text-center text-sm text-muted-foreground">
                {t("toolStats.noData")}
              </p>
            ) : (
              <ResponsiveContainer width="100%" height={Math.max(topTools.length * 36, 200)}>
                <BarChart data={topTools} layout="vertical" margin={{ left: 80, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis type="number" />
                  <YAxis
                    type="category"
                    dataKey="shortName"
                    tick={{ fontSize: 12 }}
                    width={80}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: 8,
                    }}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                    {topTools.map((_, i) => (
                      <Cell key={i} fill={chartSeriesColor(i)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Usage Over Time Line Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t("toolStats.usageOverTime")}</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-72 w-full" />
            ) : dailyTotals.length === 0 ? (
              <p className="py-12 text-center text-sm text-muted-foreground">
                {t("toolStats.noData")}
              </p>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dailyTotals} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: 8,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="count"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={false}
                    name={t("toolStats.calls")}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Detailed Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t("toolStats.detailTable")}</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-48 w-full" />
          ) : !summary?.top_tools?.length ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {t("toolStats.noData")}
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="pb-3 font-medium">{t("toolStats.tool")}</th>
                    <th className="pb-3 font-medium text-right">{t("toolStats.calls")}</th>
                    <th className="pb-3 font-medium text-right">{t("toolStats.successRate")}</th>
                    <th className="pb-3 font-medium text-right">{t("toolStats.avgDuration")}</th>
                    <th className="pb-3 font-medium">{t("toolStats.server")}</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.top_tools.map((tool) => (
                    <tr key={tool.tool_name} className="border-b border-border/50">
                      <td className="py-2.5 font-mono text-xs">{tool.tool_name}</td>
                      <td className="py-2.5 text-right font-mono">{tool.count}</td>
                      <td className="py-2.5 text-right">
                        <Badge variant={getSuccessBadgeVariant(tool.success_rate)}>
                          {formatRate(tool.success_rate)}
                        </Badge>
                      </td>
                      <td className="py-2.5 text-right font-mono text-xs text-muted-foreground">
                        {formatDuration(tool.avg_duration_ms)}
                      </td>
                      <td className="py-2.5 text-xs text-muted-foreground">
                        {tool.server_name || "—"}
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
