import { useCallback, useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Command } from "cmdk"
import { useNavigate } from "react-router-dom"
import { useFibers } from "@/api/hooks/useDashboard"
import { api } from "@/api/client"
import { useTranslation } from "react-i18next"
import {
  SquaresFour,
  Lightbulb,
  Graph,
  ShareNetwork,
  Cloud,
  Gear,
  Sparkle,
  ChartLine,
  Gauge,
  HardDrive,
  MagnifyingGlass,
  Stack,
  Brain,
  Lock,
  Command as CommandIcon,
} from "@phosphor-icons/react"
import { openUpgradeModal } from "@/components/common/UpgradeModal"

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface NeuronResult {
  id: string
  content: string
  type: string
}

/* ------------------------------------------------------------------ */
/*  Navigation pages                                                   */
/* ------------------------------------------------------------------ */

const pages = [
  { path: "/", icon: SquaresFour, labelKey: "nav.overview" },
  { path: "/insights?tab=health", icon: Lightbulb, labelKey: "nav.health" },
  { path: "/insights?tab=timeline", icon: Lightbulb, labelKey: "nav.timeline" },
  { path: "/insights?tab=evolution", icon: Lightbulb, labelKey: "nav.evolution" },
  { path: "/insights?tab=tools", icon: Lightbulb, labelKey: "nav.toolStats" },
  { path: "/graph", icon: Graph, labelKey: "nav.graph" },
  { path: "/diagrams", icon: ShareNetwork, labelKey: "nav.mindmap" },
  { path: "/visualize", icon: ChartLine, labelKey: "nav.visualize" },
  { path: "/oracle", icon: Sparkle, labelKey: "nav.oracle" },
  { path: "/sync", icon: Cloud, labelKey: "nav.sync" },
  { path: "/storage", icon: HardDrive, labelKey: "nav.storage" },
  { path: "/tier-analytics", icon: Gauge, labelKey: "nav.tierAnalytics" },
  { path: "/settings", icon: Gear, labelKey: "nav.settings" },
] as const

/* ------------------------------------------------------------------ */
/*  Neuron type badge colors                                           */
/* ------------------------------------------------------------------ */

const TYPE_COLORS: Record<string, string> = {
  fact: "#06b6d4",
  decision: "#f59e0b",
  error: "#ef4444",
  insight: "#8b5cf6",
  preference: "#ec4899",
  workflow: "#059669",
  instruction: "#6366f1",
  pattern: "#14b8a6",
  concept: "#6366f1",
  entity: "#06b6d4",
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [neurons, setNeurons] = useState<NeuronResult[]>([])
  const [searching, setSearching] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null)
  const navigate = useNavigate()
  const { data: fiberList } = useFibers()
  const { t } = useTranslation()

  // Ctrl+K / Cmd+K toggle
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [])

  // Reset on close
  useEffect(() => {
    if (!open) {
      setQuery("")
      setNeurons([])
    }
  }, [open])

  // Debounced neuron search
  const searchNeurons = useCallback((q: string) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (q.length < 2) {
      setNeurons([])
      setSearching(false)
      return
    }
    setSearching(true)
    debounceRef.current = setTimeout(async () => {
      try {
        const data = await api.get<{ neurons: NeuronResult[] }>(
          `/neurons?content_contains=${encodeURIComponent(q)}&limit=8`,
        )
        setNeurons(data.neurons ?? [])
      } catch {
        setNeurons([])
      } finally {
        setSearching(false)
      }
    }, 300)
  }, [])

  const handleValueChange = (value: string) => {
    setQuery(value)
    searchNeurons(value)
  }

  const goTo = (path: string) => {
    navigate(path)
    setOpen(false)
  }

  const fibers = fiberList?.fibers ?? []
  const filteredFibers = query.length > 0
    ? fibers.filter((f) =>
        f.summary.toLowerCase().includes(query.toLowerCase()),
      )
    : fibers.slice(0, 5)

  return (
    <>
      {/* Trigger button in TopBar */}
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground cursor-pointer"
        aria-label={t("commandPalette.placeholder")}
      >
        <MagnifyingGlass className="size-3.5" aria-hidden="true" />
        <span className="hidden sm:inline">{t("commandPalette.search")}</span>
        <kbd className="pointer-events-none hidden select-none rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px] font-medium sm:inline-flex">
          <CommandIcon className="mr-0.5 inline size-2.5" aria-hidden="true" />K
        </kbd>
      </button>

      {/* Dialog overlay — portal to body to escape stacking contexts */}
      {open && createPortal(
        <div
          className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] bg-black/50 backdrop-blur-sm"
          onClick={(e) => {
            if (e.target === e.currentTarget) setOpen(false)
          }}
        >
          <Command
            className="w-full max-w-lg rounded-xl border border-border bg-card shadow-lg overflow-hidden"
            shouldFilter={false}
          >
            <div className="flex items-center border-b border-border px-3">
              <MagnifyingGlass className="mr-2 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
              <Command.Input
                value={query}
                onValueChange={handleValueChange}
                placeholder={t("commandPalette.placeholder")}
                className="flex h-11 w-full bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground"
              />
              {searching && (
                <div className="size-4 shrink-0 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
              )}
            </div>
            <Command.List className="max-h-80 overflow-y-auto px-2 py-2">
              <Command.Empty className="py-6 text-center text-sm text-muted-foreground">
                {t("commandPalette.noResults")}
              </Command.Empty>

              {/* Pages */}
              <Command.Group
                heading={
                  <span className="px-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {t("commandPalette.pages")}
                  </span>
                }
              >
                {pages.map(({ path, icon: Icon, labelKey }) => (
                  <Command.Item
                    key={path}
                    value={`page-${t(labelKey)}`}
                    onSelect={() => goTo(path)}
                    className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm cursor-pointer aria-selected:bg-accent"
                  >
                    <Icon className="size-4 text-muted-foreground" aria-hidden="true" />
                    <span>{t(labelKey)}</span>
                  </Command.Item>
                ))}
              </Command.Group>

              {/* Fibers */}
              {filteredFibers.length > 0 && (
                <Command.Group
                  heading={
                    <span className="px-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      {t("commandPalette.fibers")}
                    </span>
                  }
                >
                  {filteredFibers.slice(0, 6).map((fiber) => (
                    <Command.Item
                      key={fiber.id}
                      value={`fiber-${fiber.summary}`}
                      onSelect={() => {
                        navigate(`/diagrams`)
                        setOpen(false)
                      }}
                      className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm cursor-pointer aria-selected:bg-accent"
                    >
                      <Stack className="size-4 text-muted-foreground" aria-hidden="true" />
                      <span className="flex-1 truncate">{fiber.summary}</span>
                      <span className="text-[10px] text-muted-foreground font-mono">
                        {fiber.neuron_count}
                      </span>
                    </Command.Item>
                  ))}
                </Command.Group>
              )}

              {/* Neurons (exact match) */}
              {neurons.length > 0 && (
                <Command.Group
                  heading={
                    <span className="px-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      {t("commandPalette.neurons")}
                    </span>
                  }
                >
                  {neurons.map((neuron) => (
                    <Command.Item
                      key={neuron.id}
                      value={`neuron-${neuron.id}`}
                      onSelect={() => {
                        navigate(`/graph`)
                        setOpen(false)
                      }}
                      className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm cursor-pointer aria-selected:bg-accent"
                    >
                      <Brain className="size-4 text-muted-foreground" aria-hidden="true" />
                      <span className="flex-1 truncate text-xs">{neuron.content}</span>
                      <span
                        className="shrink-0 rounded-full px-1.5 py-0.5 text-[9px] font-semibold"
                        style={{
                          backgroundColor: `${TYPE_COLORS[neuron.type] ?? "#94a3b8"}20`,
                          color: TYPE_COLORS[neuron.type] ?? "#94a3b8",
                        }}
                      >
                        {neuron.type}
                      </span>
                    </Command.Item>
                  ))}
                </Command.Group>
              )}

              {/* Pro upsell hints — click opens upgrade modal */}
              <Command.Group
                heading={
                  <span className="px-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Pro
                  </span>
                }
              >
                <Command.Item
                  value="pro-semantic-search"
                  onSelect={() => { setOpen(false); openUpgradeModal() }}
                  className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm opacity-60 cursor-pointer aria-selected:bg-accent aria-selected:opacity-100"
                >
                  <Lock className="size-4 text-muted-foreground" aria-hidden="true" />
                  <span className="flex-1">{t("commandPalette.semanticSearch")}</span>
                  <span className="rounded-md bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
                    PRO
                  </span>
                </Command.Item>
                <Command.Item
                  value="pro-cross-brain-search"
                  onSelect={() => { setOpen(false); openUpgradeModal() }}
                  className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm opacity-60 cursor-pointer aria-selected:bg-accent aria-selected:opacity-100"
                >
                  <Lock className="size-4 text-muted-foreground" aria-hidden="true" />
                  <span className="flex-1">{t("commandPalette.crossBrainSearch")}</span>
                  <span className="rounded-md bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
                    PRO
                  </span>
                </Command.Item>
              </Command.Group>
            </Command.List>

            {/* Footer hint */}
            <div className="flex items-center justify-between border-t border-border px-4 py-2">
              <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                <span>
                  <kbd className="rounded border border-border bg-muted px-1 py-0.5 font-mono">↑↓</kbd>{" "}
                  {t("commandPalette.navigate")}
                </span>
                <span>
                  <kbd className="rounded border border-border bg-muted px-1 py-0.5 font-mono">↵</kbd>{" "}
                  {t("commandPalette.select")}
                </span>
                <span>
                  <kbd className="rounded border border-border bg-muted px-1 py-0.5 font-mono">esc</kbd>{" "}
                  {t("commandPalette.close")}
                </span>
              </div>
            </div>
          </Command>
        </div>,
        document.body,
      )}
    </>
  )
}
