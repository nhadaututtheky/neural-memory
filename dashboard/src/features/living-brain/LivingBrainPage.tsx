import { useCallback, useEffect, useMemo, useRef } from "react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { LockSimple } from "@phosphor-icons/react"
import { useIsPro } from "@/api/hooks/useDashboard"
import { openUpgradeModal } from "@/components/common/UpgradeModal"
import { useLivingBrain } from "./hooks/useLivingBrain"
import { useReducedMotion } from "./hooks/useReducedMotion"
import { useKeyboardNav } from "./hooks/useKeyboardNav"
import { useFocusDeepLink } from "./hooks/useFocusDeepLink"
import { useActivationStream } from "./hooks/useActivationStream"
import { BrainCanvas } from "./components/BrainCanvas"
import { CanvasErrorBoundary } from "./components/CanvasErrorBoundary"
import { ModeToggle, type GraphMode } from "./components/ModeToggle"
import { NodeDetailPanel } from "./components/NodeDetailPanel"
import { StatsBar } from "./components/StatsBar"
import { SettingsDrawer } from "./components/SettingsDrawer"
import { ShareBrain } from "./components/ShareBrain"
import { useLivingBrainStore } from "./stores/livingBrainStore"
import { useBrainSettings } from "./stores/brainSettingsStore"
import type { BrainLayout, PositionedNeuron } from "./engine/types"

const LIVING_BRAIN_NODE_LIMIT = 500
const MODE_STORAGE_KEY = "nm.graph.mode"

const EMPTY_LAYOUT: BrainLayout = {
  neurons: [],
  edges: [],
  neighbors: new Map(),
  indexById: new Map(),
}

export default function LivingBrainPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const isPro = useIsPro()

  const handleModeChange = useCallback(
    (next: GraphMode) => {
      if (typeof window !== "undefined") {
        window.localStorage.setItem(MODE_STORAGE_KEY, next)
      }
      if (next === "2d") navigate("/graph")
    },
    [navigate],
  )

  // Pro gate lives at the top: free users short-circuit BEFORE any data hooks
  // run, so we don't hit /api/graph, never start ambient pulse timers, and
  // never leak neuron content into the free tier.
  if (!isPro) {
    return (
      <div className="flex h-[calc(100vh-3.5rem)] flex-col gap-4 p-4">
        <PageHeader
          t={t}
          isPro={false}
          layout={null}
          onModeChange={handleModeChange}
          onRequestRender={null}
          getCanvasEl={null}
        />
        <ProUpsellCard />
      </div>
    )
  }

  return <LivingBrainStage onModeChange={handleModeChange} />
}

function LivingBrainStage({
  onModeChange,
}: {
  onModeChange: (m: GraphMode) => void
}) {
  const { t } = useTranslation()
  const { layout, isLoading } = useLivingBrain(LIVING_BRAIN_NODE_LIMIT)
  const reducedMotion = useReducedMotion()

  const selectedId = useLivingBrainStore((s) => s.selectedId)
  const hoveredId = useLivingBrainStore((s) => s.hoveredId)
  const setSelected = useLivingBrainStore((s) => s.setSelected)
  const setHovered = useLivingBrainStore((s) => s.setHovered)
  const clear = useLivingBrainStore((s) => s.clear)

  const effects = useBrainSettings((s) => s.effects)
  const brainShell = useBrainSettings((s) => s.brainShell)
  const activationPulse = useBrainSettings((s) => s.activationPulse)

  const canvasWrapperRef = useRef<HTMLDivElement>(null)
  const getCanvasEl = useCallback(
    () => canvasWrapperRef.current?.querySelector("canvas") ?? null,
    [],
  )

  // P4 review fix (H3): the Canvas runs frameloop="demand", so the WebGL
  // drawing buffer may not reflect the latest state when the user hits
  // "Share". We plumb an imperative `gl.render(scene, camera)` callback up
  // from inside the Canvas and call it right before `canvas.toBlob`.
  const rendererRef = useRef<(() => void) | null>(null)
  const registerRenderer = useCallback((fn: (() => void) | null) => {
    rendererRef.current = fn
  }, [])
  const requestRender = useCallback(() => {
    rendererRef.current?.()
  }, [])

  const pulseEnabled = activationPulse && !reducedMotion
  const { getPulses, kickTick, pulseDurationMs, activeCount, totalPulses } =
    useActivationStream(layout, { enabled: pulseEnabled })

  const selected = useMemo<PositionedNeuron | null>(() => {
    if (!layout || !selectedId) return null
    const idx = layout.indexById.get(selectedId)
    return idx === undefined ? null : layout.neurons[idx]
  }, [layout, selectedId])

  const knownIds = useMemo(
    () => new Set(layout?.neurons.map((n) => n.id) ?? []),
    [layout],
  )

  // P3 review fix (HIGH): when layout swaps and the stored selection no longer
  // maps to any known neuron, clear it so the URL doesn't keep a dead ?focus=.
  useEffect(() => {
    if (!layout) return
    if (selectedId && !knownIds.has(selectedId)) {
      clear()
    }
  }, [layout, selectedId, knownIds, clear])

  const applyFocus = useCallback(
    (id: string) => {
      setSelected(id)
    },
    [setSelected],
  )

  useFocusDeepLink({ knownIds, selectedId, onApply: applyFocus })
  useKeyboardNav({
    layout: layout ?? EMPTY_LAYOUT,
    selectedId,
    hoveredId,
    onSelect: setSelected,
  })

  const handleCanvasClick = useCallback(
    (n: PositionedNeuron) => setSelected(n.id),
    [setSelected],
  )

  const handleClose = useCallback(() => clear(), [clear])

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col gap-4 p-4">
      <PageHeader
        t={t}
        isPro
        layout={layout}
        onModeChange={onModeChange}
        onRequestRender={requestRender}
        getCanvasEl={getCanvasEl}
      />

      <Card className="relative flex-1 flex flex-col min-h-0 overflow-hidden">
        <CardContent className="flex-1 p-0 min-h-0">
          {isLoading ? (
            <Skeleton className="h-full w-full" />
          ) : layout ? (
            <CanvasErrorBoundary
              fallback={
                <div className="flex h-full items-center justify-center">
                  <p className="text-sm text-muted-foreground">
                    {t("livingBrain.webglUnavailable")}
                  </p>
                </div>
              }
            >
              {/*
                P3 review fix (MEDIUM): onPointerLeave on the canvas wrapper
                clears hover state as a belt-and-braces over r3f's
                onPointerOut — which sometimes misses when the pointer exits
                to DOM chrome quickly.
              */}
              <div
                ref={canvasWrapperRef}
                className="h-full w-full"
                onPointerLeave={() => setHovered(null)}
              >
                <BrainCanvas
                  layout={layout}
                  onNodeClick={handleCanvasClick}
                  reducedMotion={reducedMotion}
                  showBrainShell={brainShell}
                  autoRotate={effects}
                  getPulses={getPulses}
                  pulseDurationMs={pulseDurationMs}
                  pulseKickTick={kickTick}
                  onRegisterRenderer={registerRenderer}
                />
              </div>
              <div className="pointer-events-none absolute inset-0">
                <NodeDetailPanel
                  layout={layout}
                  selected={selected}
                  onClose={handleClose}
                  onSelectNeighbor={setSelected}
                />
                <StatsBar
                  neurons={layout.neurons.length}
                  synapses={layout.edges.length}
                  activeNeurons={activeCount}
                  totalPulses={totalPulses}
                />
              </div>
            </CanvasErrorBoundary>
          ) : (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-muted-foreground">
                {t("livingBrain.empty")}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function PageHeader({
  t,
  isPro,
  layout,
  onModeChange,
  onRequestRender,
  getCanvasEl,
}: {
  t: (k: string, opts?: Record<string, unknown>) => string
  isPro: boolean
  layout: BrainLayout | null
  onModeChange: (m: GraphMode) => void
  onRequestRender: (() => void) | null
  getCanvasEl: (() => HTMLCanvasElement | null) | null
}) {
  return (
    <div className="flex items-center justify-between shrink-0">
      <div>
        <h1 className="font-display text-2xl font-bold">
          {t("livingBrain.title")}
        </h1>
        <p className="text-sm text-muted-foreground">
          {t("livingBrain.subtitle")}
        </p>
      </div>
      <div className="flex items-center gap-3">
        {layout && (
          <span className="text-xs text-muted-foreground">
            {t("graph.nodesCount", {
              nodes: layout.neurons.length.toLocaleString(),
              edges: layout.edges.length.toLocaleString(),
            })}
          </span>
        )}
        {isPro && getCanvasEl && (
          <>
            <ShareBrain
              getCanvasEl={getCanvasEl}
              onRequestRender={onRequestRender}
            />
            <SettingsDrawer />
          </>
        )}
        <ModeToggle mode="3d" onChange={onModeChange} />
      </div>
    </div>
  )
}

function ProUpsellCard() {
  const { t } = useTranslation()
  return (
    <Card className="flex flex-1 items-center justify-center">
      <CardContent className="flex max-w-xl flex-col items-center gap-4 p-8 text-center">
        <Badge variant="default" className="px-3 py-1 text-xs uppercase">
          <LockSimple className="mr-1 h-3 w-3" /> {t("livingBrain.upsell.badge")}
        </Badge>
        <h2 className="font-display text-2xl font-bold">
          {t("livingBrain.upsell.title")}
        </h2>
        <p className="text-sm text-muted-foreground">
          {t("livingBrain.upsell.body")}
        </p>
        <Button onClick={openUpgradeModal} size="lg">
          {t("livingBrain.upsell.cta")}
        </Button>
      </CardContent>
    </Card>
  )
}
