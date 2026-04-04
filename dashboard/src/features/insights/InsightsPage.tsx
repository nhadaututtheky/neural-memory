import { lazy, Suspense } from "react"
import { useSearchParams } from "react-router-dom"
import { useTranslation } from "react-i18next"
import {
  Heartbeat,
  Clock,
  TrendUp,
  ChartBar,
} from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { PageSkeleton } from "@/components/common/PageSkeleton"
import type { ComponentType, ElementType } from "react"

/* ------------------------------------------------------------------ */
/*  Lazy-loaded tab content — reuses existing full page components     */
/* ------------------------------------------------------------------ */

const HealthPage = lazy(() => import("@/features/health/HealthPage"))
const TimelinePage = lazy(() => import("@/features/timeline/TimelinePage"))
const EvolutionPage = lazy(() => import("@/features/evolution/EvolutionPage"))
const ToolStatsPage = lazy(() => import("@/features/tool-stats/ToolStatsPage"))

/* ------------------------------------------------------------------ */
/*  Tab definitions                                                    */
/* ------------------------------------------------------------------ */

interface TabDef {
  id: string
  labelKey: string
  icon: ElementType
  component: ComponentType
}

const TABS: TabDef[] = [
  { id: "health", labelKey: "nav.health", icon: Heartbeat, component: HealthPage },
  { id: "timeline", labelKey: "nav.timeline", icon: Clock, component: TimelinePage },
  { id: "evolution", labelKey: "nav.evolution", icon: TrendUp, component: EvolutionPage },
  { id: "tools", labelKey: "nav.toolStats", icon: ChartBar, component: ToolStatsPage },
]

const DEFAULT_TAB = "health"

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export default function InsightsPage() {
  const { t } = useTranslation()
  const [searchParams, setSearchParams] = useSearchParams()
  const activeTab = searchParams.get("tab") ?? DEFAULT_TAB
  const current = TABS.find((tab) => tab.id === activeTab) ?? TABS[0]
  const TabContent = current.component

  return (
    <div className="flex flex-col">
      {/* Tab bar */}
      <div className="sticky top-0 z-10 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex gap-1 px-4 pt-2">
          {TABS.map((tab) => {
            const Icon = tab.icon
            const isActive = tab.id === activeTab
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setSearchParams({ tab: tab.id })}
                className={cn(
                  "inline-flex items-center gap-2 rounded-t-lg px-4 py-2.5 text-sm font-medium transition-colors cursor-pointer",
                  isActive
                    ? "border-b-2 border-primary bg-card text-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
                aria-selected={isActive}
                role="tab"
              >
                <Icon className="size-4" aria-hidden="true" />
                {t(tab.labelKey)}
              </button>
            )
          })}
        </div>
      </div>

      {/* Tab content — renders the full original page component */}
      <Suspense fallback={<PageSkeleton />}>
        <TabContent />
      </Suspense>
    </div>
  )
}
