import { useEffect, useRef, useState } from "react"
import { useVisualize } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useTranslation } from "react-i18next"
import type { VisualizeResponse } from "@/api/types"

const QUERY_SUGGESTIONS = [
  { label: "Memory types", query: "memory types breakdown", icon: "◐" },
  { label: "Timeline", query: "memories over time", icon: "◷" },
  { label: "Top tags", query: "most used tags", icon: "◈" },
  { label: "Priority", query: "priority distribution", icon: "▥" },
  { label: "Activity", query: "activity by day", icon: "▦" },
  { label: "Connections", query: "most connected neurons", icon: "◎" },
] as const

interface MemoryChartProps {
  /** Initial query to visualize */
  query?: string
  /** Chart type override (auto-detect if omitted) */
  chartType?: string
  /** Output format: vega_lite | markdown_table | ascii */
  format?: string
  /** Max data points */
  limit?: number
  /** Compact mode — no card wrapper */
  compact?: boolean
}

export default function MemoryChart({
  query: initialQuery = "",
  chartType,
  format = "vega_lite",
  limit = 20,
  compact = false,
}: MemoryChartProps) {
  const { t } = useTranslation()
  const vegaRef = useRef<HTMLDivElement>(null)
  const visualize = useVisualize()
  const [query, setQuery] = useState(initialQuery)
  const [result, setResult] = useState<VisualizeResponse | null>(null)

  const handleVisualize = (q?: string) => {
    const finalQuery = q ?? query
    if (!finalQuery.trim()) return
    if (q) setQuery(q)
    visualize.mutate(
      { query: finalQuery.trim(), chart_type: chartType, format, limit },
      { onSuccess: (data) => setResult(data) },
    )
  }

  // Render Vega-Lite spec when available — patch labels for readability
  useEffect(() => {
    if (!result?.vega_lite || !vegaRef.current) return

    let cancelled = false

    // Patch Vega-Lite spec for readable labels
    const spec = { ...(result.vega_lite as Record<string, unknown>) }

    // Override config for readable axis labels
    const patchConfig = {
      ...(spec.config as Record<string, unknown> | undefined),
      axis: {
        labelFontSize: 11,
        labelAngle: 0,
        labelLimit: 120,
        labelOverlap: "greedy",
        titleFontSize: 12,
        titlePadding: 8,
      },
      view: { stroke: "transparent" },
      legend: { labelFontSize: 11, titleFontSize: 12 },
    }
    spec.config = patchConfig

    // Make chart responsive
    if (!spec.width) spec.width = "container"
    if (!spec.height) spec.height = 300

    // Patch encoding label angle if present
    const encoding = spec.encoding as Record<string, Record<string, unknown>> | undefined
    if (encoding?.x?.axis) {
      encoding.x.axis = { ...encoding.x.axis, labelAngle: -30, labelLimit: 100 }
    }

    import("vega-embed").then((vegaEmbed) => {
      if (cancelled || !vegaRef.current) return
      vegaEmbed
        .default(vegaRef.current, spec, {
          actions: { export: true, source: false, compiled: false, editor: false },
          theme: document.documentElement.classList.contains("dark") ? "dark" : undefined,
          renderer: "svg",
          width: vegaRef.current.clientWidth - 32,
        })
        .catch((err: unknown) => console.error("Vega render error:", err))
    })

    return () => {
      cancelled = true
    }
  }, [result?.vega_lite])

  const content = (
    <div className="space-y-3">
      {/* Query input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleVisualize()}
          placeholder="e.g. memory types breakdown, activity by day..."
          className="flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          aria-label="Chart query"
        />
        <Button
          size="sm"
          onClick={() => handleVisualize()}
          disabled={visualize.isPending || !query.trim()}
          className="cursor-pointer"
        >
          {visualize.isPending ? t("common.loading") : "Visualize"}
        </Button>
      </div>

      {/* Query suggestion chips */}
      {!result && (
        <div className="flex flex-wrap gap-2">
          {QUERY_SUGGESTIONS.map((s) => (
            <button
              key={s.query}
              type="button"
              onClick={() => handleVisualize(s.query)}
              disabled={visualize.isPending}
              className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/50 px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground cursor-pointer disabled:opacity-50"
            >
              <span>{s.icon}</span>
              {s.label}
            </button>
          ))}
        </div>
      )}

      {/* Chart output */}
      {result && (
        <div className="space-y-2">
          {/* Vega-Lite chart */}
          {result.vega_lite && (
            <div
              ref={vegaRef}
              className="w-full min-h-[300px] rounded-md border border-border bg-background p-4 [&_svg]:max-w-full"
            />
          )}

          {/* Markdown table fallback */}
          {result.markdown && !result.vega_lite && (
            <pre className="whitespace-pre-wrap rounded-md border border-border bg-muted p-3 text-xs font-mono overflow-x-auto">
              {result.markdown}
            </pre>
          )}

          {/* ASCII chart fallback */}
          {result.ascii && !result.vega_lite && (
            <pre className="whitespace-pre rounded-md border border-border bg-muted p-3 text-xs font-mono overflow-x-auto">
              {result.ascii}
            </pre>
          )}

          {/* No data message */}
          {result.message && !result.vega_lite && !result.markdown && !result.ascii && (
            <p className="text-sm text-muted-foreground">{result.message}</p>
          )}

          {/* Memory list fallback */}
          {result.memories && result.memories.length > 0 && (
            <div className="space-y-1">
              {result.memories.map((m) => (
                <div key={m.id} className="text-xs text-muted-foreground truncate">
                  <span className="font-mono text-[10px] opacity-50">[{m.type}]</span>{" "}
                  {m.content}
                </div>
              ))}
            </div>
          )}

          {/* Chart meta + actions */}
          <div className="flex items-center justify-between">
            {result.data_points_count != null && result.data_points_count > 0 && (
              <p className="text-[11px] text-muted-foreground">
                {result.data_points_count} data points · {result.chart_type}
              </p>
            )}
            <button
              type="button"
              onClick={() => {
                setResult(null)
                setQuery("")
              }}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              ← Try another query
            </button>
          </div>

          {/* Quick re-query chips */}
          <div className="flex flex-wrap gap-1.5 pt-1">
            {QUERY_SUGGESTIONS.filter((s) => s.query !== query).slice(0, 4).map((s) => (
              <button
                key={s.query}
                type="button"
                onClick={() => handleVisualize(s.query)}
                disabled={visualize.isPending}
                className="inline-flex items-center gap-1 rounded-full border border-border/50 px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground cursor-pointer disabled:opacity-50"
              >
                <span>{s.icon}</span>
                {s.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )

  if (compact) return content

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Memory Visualizer</CardTitle>
      </CardHeader>
      <CardContent>{content}</CardContent>
    </Card>
  )
}
