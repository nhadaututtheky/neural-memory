import { lazy, Suspense } from "react"
import { Routes, Route, Navigate } from "react-router-dom"
import { AppShell } from "@/components/layout/AppShell"
import { PageSkeleton } from "@/components/common/PageSkeleton"

const OverviewPage = lazy(() => import("@/features/overview/OverviewPage"))
const HealthPage = lazy(() => import("@/features/health/HealthPage"))
const GraphPage = lazy(() => import("@/features/graph/GraphPage"))
const TimelinePage = lazy(() => import("@/features/timeline/TimelinePage"))
const EvolutionPage = lazy(() => import("@/features/evolution/EvolutionPage"))
const DiagramsPage = lazy(() => import("@/features/diagrams/DiagramsPage"))
const SettingsPage = lazy(() => import("@/features/settings/SettingsPage"))

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route
          index
          element={
            <Suspense fallback={<PageSkeleton />}>
              <OverviewPage />
            </Suspense>
          }
        />
        <Route
          path="health"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <HealthPage />
            </Suspense>
          }
        />
        <Route
          path="graph"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <GraphPage />
            </Suspense>
          }
        />
        <Route
          path="timeline"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <TimelinePage />
            </Suspense>
          }
        />
        <Route
          path="evolution"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <EvolutionPage />
            </Suspense>
          }
        />
        <Route
          path="diagrams"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <DiagramsPage />
            </Suspense>
          }
        />
        <Route
          path="settings"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <SettingsPage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
