import { NavLink } from "react-router-dom"
import {
  SquaresFour,
  Lightbulb,
  Graph,
  ShareNetwork,
  Cloud,
  HardDrive,
  Gear,
  Brain,
  Sparkle,
  ChartLine,
  Gauge,
} from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { useLayoutStore } from "@/stores/useLayoutStore"
import { useTranslation } from "react-i18next"
import type { ElementType } from "react"

interface NavItem {
  to: string
  icon: ElementType
  labelKey: string
  separator?: boolean
}

const navItems: NavItem[] = [
  // ── Core (Free) ──
  { to: "/", icon: SquaresFour, labelKey: "nav.overview" },
  { to: "/insights", icon: Lightbulb, labelKey: "nav.insights" },
  { to: "/graph", icon: Graph, labelKey: "nav.graph" },
  { to: "/diagrams", icon: ShareNetwork, labelKey: "nav.mindmap" },
  // ── Pro ──
  { to: "/visualize", icon: ChartLine, labelKey: "nav.visualize", separator: true },
  { to: "/oracle", icon: Sparkle, labelKey: "nav.oracle" },
  { to: "/sync", icon: Cloud, labelKey: "nav.sync" },
  { to: "/storage", icon: HardDrive, labelKey: "nav.storage" },
  { to: "/tier-analytics", icon: Gauge, labelKey: "nav.tierAnalytics" },
  // ── Settings ──
  { to: "/settings", icon: Gear, labelKey: "nav.settings", separator: true },
]

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
        {navItems.map(({ to, icon: Icon, labelKey, separator }) => {
          const label = t(labelKey)
          return (
            <div key={to}>
              {separator && (
                <div className="my-2 border-t border-sidebar-border" />
              )}
              <NavLink
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
            </div>
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
