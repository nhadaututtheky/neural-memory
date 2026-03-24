import { useEffect, useRef, useState } from "react"
import { useVisualize } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { useTranslation } from "react-i18next"
import type { VisualizeResponse } from "@/api/types"

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

  const handleVisualize = () => {
    if (!query.trim()) return
    visualize.mutate(
      { query: query.trim(), chart_type: chartType, format, limit },
      { onSuccess: (data) => setResult(data) },
    )
  }

  // Render Vega-Lite spec when available
  useEffect(() => {
    if (!result?.vega_lite || !vegaRef.current) return

    let cancelled = false
    import("vega-embed").then((vegaEmbed) => {
      if (cancelled || !vegaRef.current) return
      vegaEmbed
        .default(vegaRef.current, result.vega_lite as Record<string, unknown>, {
          actions: { export: true, source: false, compiled: false, editor: false },
          theme: document.documentElement.classList.contains("dark") ? "dark" : undefined,
          renderer: "svg",
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
          placeholder={t("commandPalette.placeholder")}
          className="flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          aria-label="Chart query"
        />
        <Button
          size="sm"
          onClick={handleVisualize}
          disabled={visualize.isPending || !query.trim()}
          className="cursor-pointer"
        >
          {visualize.isPending ? t("common.loading") : "Visualize"}
        </Button>
      </div>

      {/* Chart output */}
      {result && (
        <div className="space-y-2">
          {/* Vega-Lite chart */}
          {result.vega_lite && (
            <div
              ref={vegaRef}
              className="w-full min-h-[200px] rounded-md border border-border bg-background p-2"
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

          {/* Data points count */}
          {result.data_points_count != null && result.data_points_count > 0 && (
            <p className="text-[11px] text-muted-foreground">
              {result.data_points_count} data points · {result.chart_type}
            </p>
          )}
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
