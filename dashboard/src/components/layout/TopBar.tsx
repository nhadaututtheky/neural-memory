import { SidebarSimple, Sun, Moon, Monitor, Globe, Question } from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { useLayoutStore } from "@/stores/useLayoutStore"
import { useStats, useHealthCheck } from "@/api/hooks/useDashboard"
import { Badge } from "@/components/ui/badge"
import { useTranslation } from "react-i18next"
import { CommandPalette } from "@/components/common/CommandPalette"

const themeIcons = {
  light: Sun,
  dark: Moon,
  system: Monitor,
} as const

const themeKeys = {
  light: "common.lightMode",
  dark: "common.darkMode",
  system: "common.systemTheme",
} as const

export function TopBar() {
  const { sidebarOpen, toggleSidebar, theme, cycleTheme } = useLayoutStore()
  const { data: stats } = useStats()
  const { data: healthCheck } = useHealthCheck()
  const { t, i18n } = useTranslation()

  const ThemeIcon = themeIcons[theme]
  const currentLang = i18n.language?.startsWith("vi") ? "vi" : "en"

  const toggleLanguage = () => {
    const next = currentLang === "vi" ? "en" : "vi"
    i18n.changeLanguage(next)
  }

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-4 border-b border-border bg-background/80 px-4 backdrop-blur-sm">
      {/* Sidebar toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={toggleSidebar}
        aria-label={sidebarOpen ? t("common.collapseSidebar") : t("common.expandSidebar")}
      >
        {sidebarOpen ? (
          <SidebarSimple className="size-5" weight="bold" />
        ) : (
          <SidebarSimple className="size-5" />
        )}
      </Button>

      {/* Active brain indicator */}
      {stats?.active_brain && (
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">{t("common.brain")}:</span>
          <Badge variant="secondary" className="font-mono text-xs">
            {stats.active_brain}
          </Badge>
        </div>
      )}

      {/* Command palette trigger */}
      <CommandPalette />

      {/* Spacer */}
      <div className="flex-1" />

      {/* Help / Guide link */}
      <Button
        variant="ghost"
        size="icon"
        asChild
        aria-label="Quickstart Guide"
        title="Quickstart Guide"
      >
        <a
          href="https://nhadaututtheky.github.io/neural-memory/guides/quickstart-guide/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Question className="size-4" />
        </a>
      </Button>

      {/* Language toggle */}
      <Button
        variant="ghost"
        size="sm"
        onClick={toggleLanguage}
        className="gap-1.5 px-2 text-xs font-medium"
        aria-label="Switch language"
      >
        <Globe className="size-3.5" />
        {currentLang === "vi" ? "VI" : "EN"}
      </Button>

      {/* Theme toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={cycleTheme}
        aria-label={t(themeKeys[theme])}
        title={t(themeKeys[theme])}
        data-testid="theme-toggle"
      >
        <ThemeIcon className="size-4" />
      </Button>

      {/* Version */}
      {healthCheck?.version && (
        <span className="text-xs text-muted-foreground font-mono">
          v{healthCheck.version}
        </span>
      )}
    </header>
  )
}
