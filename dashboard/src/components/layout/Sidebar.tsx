import { NavLink } from "react-router-dom"
import {
  LayoutDashboard,
  HeartPulse,
  Network,
  Clock,
  TrendingUp,
  Share2,
  Cloud,
  HardDrive,
  Settings,
  Brain,
  Sparkles,
  BarChart3,
  LineChart,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useLayoutStore } from "@/stores/useLayoutStore"
import { useTranslation } from "react-i18next"

const navItems = [
  { to: "/", icon: LayoutDashboard, labelKey: "nav.overview" },
  { to: "/health", icon: HeartPulse, labelKey: "nav.health" },
  { to: "/graph", icon: Network, labelKey: "nav.graph" },
  { to: "/timeline", icon: Clock, labelKey: "nav.timeline" },
  { to: "/evolution", icon: TrendingUp, labelKey: "nav.evolution" },
  { to: "/diagrams", icon: Share2, labelKey: "nav.mindmap" },
  { to: "/sync", icon: Cloud, labelKey: "nav.sync" },
  { to: "/oracle", icon: Sparkles, labelKey: "nav.oracle" },
  { to: "/tool-stats", icon: BarChart3, labelKey: "nav.toolStats" },
  { to: "/visualize", icon: LineChart, labelKey: "nav.visualize" },
  { to: "/storage", icon: HardDrive, labelKey: "nav.storage" },
  { to: "/settings", icon: Settings, labelKey: "nav.settings" },
] as const

export function Sidebar() {
  const sidebarOpen = useLayoutStore((s) => s.sidebarOpen)
  const { t } = useTranslation()

  return (
    <aside
      className={cn(
        "fixed inset-y-0 left-0 z-30 flex flex-col border-r border-sidebar-border bg-sidebar transition-all duration-[var(--transition-normal)]",
        sidebarOpen ? "w-56" : "w-16",
      )}
    >
      {/* Logo */}
      <div className="flex h-14 items-center gap-3 border-b border-sidebar-border px-4">
        <Brain className="size-6 shrink-0 text-sidebar-primary" />
        {sidebarOpen && (
          <span className="font-display text-base font-bold text-sidebar-foreground truncate">
            Neural Memory
          </span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-2" aria-label={t("common.mainNavigation")}>
        {navItems.map(({ to, icon: Icon, labelKey }) => {
          const label = t(labelKey)
          return (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors cursor-pointer",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-primary"
                    : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground",
                  !sidebarOpen && "justify-center px-0",
                )
              }
              title={label}
            >
              <Icon className="size-5 shrink-0" aria-hidden="true" />
              {sidebarOpen && <span>{label}</span>}
            </NavLink>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-sidebar-border p-3">
        {sidebarOpen && (
          <p className="text-xs text-sidebar-foreground/50 text-center">
            Neural Memory
          </p>
        )}
      </div>
    </aside>
  )
}
