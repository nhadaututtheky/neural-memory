import { lazy, Suspense } from "react"
import { Routes, Route, Navigate } from "react-router-dom"
import { AppShell } from "@/components/layout/AppShell"
import { PageSkeleton } from "@/components/common/PageSkeleton"

const OverviewPage = lazy(() => import("@/features/overview/OverviewPage"))
const InsightsPage = lazy(() => import("@/features/insights/InsightsPage"))
const GraphPage = lazy(() => import("@/features/graph/GraphPage"))
const DiagramsPage = lazy(() => import("@/features/diagrams/DiagramsPage"))
const SettingsPage = lazy(() => import("@/features/settings/SettingsPage"))
const SyncPage = lazy(() => import("@/features/sync/SyncPage"))
const OraclePage = lazy(() => import("@/features/oracle/OraclePage"))
const VisualizePage = lazy(() => import("@/features/visualize/VisualizePage"))
const StoragePage = lazy(() => import("@/features/storage/StoragePage"))
const StorePage = lazy(() => import("@/features/store/StorePage"))
const TierAnalyticsPage = lazy(() => import("@/features/tier-analytics/TierAnalyticsPage"))
const LivingBrainPage = lazy(() => import("@/features/living-brain/LivingBrainPage"))

function SuspensePage({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<PageSkeleton />}>{children}</Suspense>
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<SuspensePage><OverviewPage /></SuspensePage>} />
        <Route path="insights" element={<SuspensePage><InsightsPage /></SuspensePage>} />
        <Route path="graph" element={<SuspensePage><GraphPage /></SuspensePage>} />
        <Route path="living-brain" element={<SuspensePage><LivingBrainPage /></SuspensePage>} />
        <Route path="diagrams" element={<SuspensePage><DiagramsPage /></SuspensePage>} />
        <Route path="visualize" element={<SuspensePage><VisualizePage /></SuspensePage>} />
        <Route path="oracle" element={<SuspensePage><OraclePage /></SuspensePage>} />
        <Route path="sync" element={<SuspensePage><SyncPage /></SuspensePage>} />
        <Route path="store" element={<SuspensePage><StorePage /></SuspensePage>} />
        <Route path="storage" element={<SuspensePage><StoragePage /></SuspensePage>} />
        <Route path="tier-analytics" element={<SuspensePage><TierAnalyticsPage /></SuspensePage>} />
        <Route path="settings" element={<SuspensePage><SettingsPage /></SuspensePage>} />

        {/* Legacy redirects — preserve bookmarks */}
        <Route path="health" element={<Navigate to="/insights?tab=health" replace />} />
        <Route path="timeline" element={<Navigate to="/insights?tab=timeline" replace />} />
        <Route path="evolution" element={<Navigate to="/insights?tab=evolution" replace />} />
        <Route path="tool-stats" element={<Navigate to="/insights?tab=tools" replace />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
