import { useHealth } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts"

export default function HealthPage() {
  const { data: health, isLoading } = useHealth()

  const radarData = health
    ? [
        { metric: "Purity", value: health.purity_score * 100 },
        { metric: "Freshness", value: health.freshness * 100 },
        { metric: "Connectivity", value: health.connectivity * 100 },
        { metric: "Diversity", value: health.diversity * 100 },
        { metric: "Consolidation", value: health.consolidation_ratio * 100 },
        { metric: "Activation", value: health.activation_efficiency * 100 },
        { metric: "Recall", value: health.recall_confidence * 100 },
        { metric: "Orphan Rate", value: (1 - health.orphan_rate) * 100 },
      ]
    : []

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center gap-4">
        <h1 className="font-display text-2xl font-bold">Health</h1>
        {health && (
          <Badge
            variant={
              health.grade.startsWith("A")
                ? "success"
                : health.grade.startsWith("B")
                  ? "secondary"
                  : "warning"
            }
            className="text-lg px-3 py-1"
          >
            {health.grade}
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Radar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Brain Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-80 w-full" />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="var(--color-border)" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: "var(--color-muted-foreground)", fontSize: 12 }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 100]}
                    tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
                  />
                  <Radar
                    name="Health"
                    dataKey="value"
                    stroke="var(--color-primary)"
                    fill="var(--color-primary)"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Warnings */}
        <Card>
          <CardHeader>
            <CardTitle>Warnings & Recommendations</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                {health?.warnings && health.warnings.length > 0 ? (
                  <div className="space-y-2">
                    {health.warnings.map((w, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 rounded-lg border border-border p-3"
                      >
                        <Badge
                          variant={
                            w.severity === "critical"
                              ? "destructive"
                              : w.severity === "warning"
                                ? "warning"
                                : "secondary"
                          }
                          className="mt-0.5 shrink-0"
                        >
                          {w.severity}
                        </Badge>
                        <span className="text-sm">{w.message}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No warnings. Brain is healthy!
                  </p>
                )}

                {health?.recommendations && health.recommendations.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">
                      Recommendations
                    </h3>
                    <ul className="space-y-1 text-sm">
                      {health.recommendations.map((r, i) => (
                        <li key={i} className="flex gap-2">
                          <span className="text-primary">-</span>
                          <span>{r}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
