import { NavLink } from "react-router-dom"
import { Brain } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { useLayoutStore } from "@/stores/useLayoutStore"
import { useIsPro } from "@/api/hooks/useDashboard"
import { useTranslation } from "react-i18next"
import {
  NAV_GROUP_LABEL_KEYS,
  NAV_GROUP_ORDER,
  navItemsByGroup,
  type NavGroup,
} from "./nav-items"

export function Sidebar() {
  const sidebarOpen = useLayoutStore((s) => s.sidebarOpen)
  const isPro = useIsPro()
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
      <nav
        className="flex-1 overflow-y-auto p-2"
        aria-label={t("common.mainNavigation")}
      >
        {NAV_GROUP_ORDER.map((group: NavGroup) => {
          const items = navItemsByGroup(group)
          if (items.length === 0) return null
          const groupLabel = t(NAV_GROUP_LABEL_KEYS[group])

          return (
            <div key={group} className="mb-3 last:mb-0">
              {sidebarOpen ? (
                // Presentational: the accessible grouping comes from the
                // aria-label on the list below, so this heading is not
                // announced twice and never receives focus.
                <p
                  aria-hidden="true"
                  className="px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-sidebar-foreground/45"
                >
                  {groupLabel}
                </p>
              ) : (
                // Collapsed: a rule stands in for the heading so the grouping
                // is still legible at 64px.
                <div className="mx-2 my-2 border-t border-sidebar-border" />
              )}

              <ul className="space-y-1" aria-label={groupLabel}>
                {items.map(({ to, icon: Icon, labelKey, pro }) => {
                  const label = t(labelKey)
                  const showProBadge = pro && !isPro
                  return (
                    <li key={to}>
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
                        title={showProBadge ? `${label} (Pro)` : label}
                      >
                        <Icon className="size-5 shrink-0" aria-hidden="true" />
                        {sidebarOpen && (
                          <>
                            <span className="truncate">{label}</span>
                            {showProBadge && (
                              <span className="ml-auto rounded bg-sidebar-primary/15 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-sidebar-primary">
                                {t("license.pro", "Pro")}
                              </span>
                            )}
                          </>
                        )}
                      </NavLink>
                    </li>
                  )
                })}
              </ul>
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
