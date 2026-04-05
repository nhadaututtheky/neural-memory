import { useState } from "react"
import { useHealth } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import {
  CaretDown,
  CaretUp,
  Brain,
  Lightbulb,
  Lightning,
  BookOpen,
  Warning,
  TrendUp,
  ArrowRight,
} from "@phosphor-icons/react"
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts"
import { useTranslation } from "react-i18next"
import type { PenaltyFactor } from "@/api/types"

const ENRICHMENT_ICONS = [Brain, Lightbulb, Lightning, BookOpen] as const
const ENRICHMENT_COLORS = ["#6366f1", "#f59e0b", "#059669", "#06b6d4"] as const
const ENRICHMENT_KEYS = ["remember", "causal", "diverse", "train"] as const

const GRADE_CONFIG: Record<string, { color: string; bg: string; variant: "success" | "secondary" | "warning" | "destructive" }> = {
  A: { color: "#059669", bg: "#05966915", variant: "success" },
  B: { color: "#6366f1", bg: "#6366f115", variant: "secondary" },
  C: { color: "#f59e0b", bg: "#f59e0b15", variant: "warning" },
  D: { color: "#ef4444", bg: "#ef444415", variant: "destructive" },
  F: { color: "#ef4444", bg: "#ef444415", variant: "destructive" },
}

function getGradeConfig(grade: string) {
  const letter = grade.charAt(0).toUpperCase()
  return GRADE_CONFIG[letter] ?? GRADE_CONFIG.F
}

export default function HealthPage() {
  const { data: health, isLoading } = useHealth()
  const { t } = useTranslation()

  const gradeConfig = health ? getGradeConfig(health.grade) : GRADE_CONFIG.F

  const radarData = health
    ? [
        { metric: t("health.purity"), value: health.purity_score * 100 },
        { metric: t("health.freshness"), value: health.freshness * 100 },
        { metric: t("health.connectivity"), value: health.connectivity * 100 },
        { metric: t("health.diversity"), value: health.diversity * 100 },
        { metric: t("health.consolidation"), value: health.consolidation_ratio * 100 },
        { metric: t("health.activation"), value: health.activation_efficiency * 100 },
        { metric: t("health.recall"), value: health.recall_confidence * 100 },
        { metric: t("health.orphanRate"), value: (1 - health.orphan_rate) * 100 },
      ]
    : []

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center gap-4">
        <h1 className="font-display text-2xl font-bold">{t("health.title")}</h1>
        {health && (
          <Badge
            variant={gradeConfig.variant}
            className="text-lg px-3 py-1"
          >
            {health.grade}
          </Badge>
        )}
      </div>

      {/* Top Penalties — actionable cards */}
      {health && health.top_penalties && health.top_penalties.length > 0 && (
        <TopPenaltiesSection penalties={health.top_penalties} />
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Radar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>{t("health.brainMetrics")}</CardTitle>
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
                    name={t("health.radarName")}
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
            <CardTitle>{t("health.warnings")}</CardTitle>
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
                ) : health?.top_penalties && health.top_penalties.length > 0 ? (
                  <div className="space-y-2">
                    {health.top_penalties.map((p: PenaltyFactor, i: number) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 rounded-lg border border-border p-3"
                      >
                        <Badge variant="warning" className="mt-0.5 shrink-0">
                          {p.component.replace(/_/g, " ")}
                        </Badge>
                        <span className="text-sm">{p.action}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    {t("health.noWarnings")}
                  </p>
                )}

                {health?.recommendations && health.recommendations.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">
                      {t("health.recommendations")}
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
      {/* Memory Enrichment Guide */}
      <MemoryEnrichmentGuide />
    </div>
  )
}

function TopPenaltiesSection({ penalties }: { penalties: PenaltyFactor[] }) {
  const { t } = useTranslation()

  return (
    <Card className="border-amber-500/30 bg-amber-500/5">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Warning className="size-5 text-amber-500" />
          {t("health.topPenalties")}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {penalties.map((p, i) => (
            <PenaltyCard key={p.component} penalty={p} rank={i + 1} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function PenaltyCard({ penalty, rank }: { penalty: PenaltyFactor; rank: number }) {
  const { t } = useTranslation()
  const scorePct = Math.round(penalty.current_score * 100)
  const weightPct = Math.round(penalty.weight * 100)
  const penaltyPts = penalty.penalty_points.toFixed(1)
  const gainPts = penalty.estimated_gain.toFixed(1)

  const barColor =
    scorePct >= 60 ? "bg-amber-500" : scorePct >= 30 ? "bg-orange-500" : "bg-red-500"

  return (
    <div className="rounded-lg border border-border bg-card p-4 transition-shadow hover:shadow-md">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex size-6 items-center justify-center rounded-full bg-amber-500/15 text-xs font-bold text-amber-600">
            {rank}
          </span>
          <h4 className="text-sm font-semibold capitalize">{penalty.component}</h4>
        </div>
        <Badge variant="outline" className="text-xs text-red-500 border-red-500/30">
          {t("health.penaltyPoints", { points: penaltyPts })}
        </Badge>
      </div>

      {/* Score bar */}
      <div className="mb-3">
        <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
          <span>{t("health.currentScore", { score: scorePct })}</span>
          <span>{t("health.weight", { weight: weightPct })}</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className={`h-full rounded-full transition-all ${barColor}`}
            style={{ width: `${scorePct}%` }}
          />
        </div>
      </div>

      {/* Estimated gain */}
      <div className="mb-3 flex items-center gap-1.5 rounded-md bg-emerald-500/10 px-2.5 py-1.5">
        <TrendUp className="size-3.5 text-emerald-500" />
        <span className="text-xs font-medium text-emerald-600">
          {t("health.estimatedGain", { gain: gainPts })}
        </span>
      </div>

      {/* Action */}
      <div className="flex items-start gap-1.5 text-xs text-muted-foreground">
        <ArrowRight className="mt-0.5 size-3 shrink-0 text-primary" />
        <span>{penalty.action}</span>
      </div>
    </div>
  )
}

function MemoryEnrichmentGuide() {
  const [expanded, setExpanded] = useState(false)
  const { t } = useTranslation()

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Brain className="size-5 text-primary" />
            {t("health.enrichTitle")}
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded((v) => !v)}
            aria-label={expanded ? t("health.collapseTips") : t("health.expandTips")}
          >
            {expanded ? <CaretUp className="size-4" /> : <CaretDown className="size-4" />}
            <span className="ml-1 text-xs">{expanded ? t("health.less") : t("health.more")}</span>
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          {t("health.enrichDesc")}
        </p>
      </CardHeader>
      <CardContent>
        <div className={`grid grid-cols-1 gap-4 ${expanded ? "md:grid-cols-2" : "md:grid-cols-4"}`}>
          {ENRICHMENT_KEYS.map((key, idx) => {
            const Icon = ENRICHMENT_ICONS[idx]
            const color = ENRICHMENT_COLORS[idx]
            const title = t(`enrichment.${key}Title`)
            const tips = t(`enrichment.${key}Tips`, { returnObjects: true }) as string[]

            return (
              <div
                key={key}
                className="rounded-lg border border-border p-4 transition-shadow hover:shadow-sm"
              >
                <div className="mb-3 flex items-center gap-2">
                  <div
                    className="flex size-8 items-center justify-center rounded-lg"
                    style={{ backgroundColor: `${color}15` }}
                  >
                    <Icon className="size-4" style={{ color }} />
                  </div>
                  <h3 className="text-sm font-semibold">{title}</h3>
                </div>
                <ul className="space-y-2">
                  {(expanded ? tips : tips.slice(0, 2)).map((tip, i) => (
                    <li key={i} className="flex gap-2 text-xs leading-relaxed">
                      <span className="mt-0.5 shrink-0 text-muted-foreground">-</span>
                      <span className={tip.startsWith("BAD:") || tip.startsWith("TỆ:") ? "text-destructive" : tip.startsWith("GOOD:") || tip.startsWith("TỐT:") ? "text-primary" : ""}>
                        {tip}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
