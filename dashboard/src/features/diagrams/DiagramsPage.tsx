import { useState } from "react"
import { useFibers, useFiberDiagram } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { FiberMindmap } from "./FiberMindmap"
import type { FiberSummary } from "@/api/types"
import { useTranslation } from "react-i18next"

interface SelectedNeuron {
  id: string
  content: string
  type: string
}

export default function DiagramsPage() {
  const { data: fiberList, isLoading: fibersLoading } = useFibers()
  const [selectedFiber, setSelectedFiber] = useState("")
  const { data: diagram, isLoading: diagramLoading } =
    useFiberDiagram(selectedFiber)
  const [selectedNeuron, setSelectedNeuron] = useState<SelectedNeuron | null>(null)
  const selectedFiberSummary = fiberList?.fibers.find((f: FiberSummary) => f.id === selectedFiber)
  const { t } = useTranslation()

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col gap-4 p-4">
      <h1 className="font-display text-2xl font-bold shrink-0">{t("diagrams.title")}</h1>

      <div className="flex flex-1 gap-4 min-h-0">
        {/* Fiber selector — narrow sidebar */}
        <Card className="w-56 shrink-0 flex flex-col">
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm">{t("diagrams.fibers")}</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto px-2 pb-2">
            {fibersLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : fiberList?.fibers && fiberList.fibers.length > 0 ? (
              <div className="space-y-0.5">
                {fiberList.fibers.map((fiber) => (
                  <button
                    key={fiber.id}
                    onClick={() => {
                      setSelectedFiber(fiber.id)
                      setSelectedNeuron(null)
                    }}
                    className={`w-full cursor-pointer rounded-md px-2.5 py-1.5 text-left text-xs transition-colors ${
                      selectedFiber === fiber.id
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-accent"
                    }`}
                  >
                    <p className="font-medium truncate">{fiber.summary}</p>
                    <p className="text-[10px] text-muted-foreground">
                      {fiber.neuron_count} {t("common.neurons")}
                    </p>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">{t("diagrams.noFibers")}</p>
            )}
          </CardContent>
        </Card>

        {/* Mindmap — takes remaining space */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          <Card className="flex-1 flex flex-col min-h-0">
            <CardHeader className="py-3 px-4 shrink-0 flex flex-row items-center justify-between">
              <CardTitle className="text-sm">
                {diagram
                  ? selectedFiberSummary?.summary ?? t("diagrams.fiberLabel", { id: diagram.fiber_id.slice(0, 16) })
                  : t("diagrams.selectFiber")}
              </CardTitle>
              {diagram && (
                <div className="flex gap-4 text-xs text-muted-foreground">
                  <span>
                    {t("diagrams.neuronsCount")} <strong className="font-mono text-foreground">{diagram.neurons.length}</strong>
                  </span>
                  <span>
                    {t("diagrams.connectionsCount")} <strong className="font-mono text-foreground">{diagram.synapses.length}</strong>
                  </span>
                </div>
              )}
            </CardHeader>
            <CardContent className="flex-1 p-2 min-h-0">
              {!selectedFiber ? (
                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                  {t("diagrams.selectFiberPrompt")}
                </div>
              ) : diagramLoading ? (
                <Skeleton className="h-full w-full" />
              ) : diagram ? (
                <FiberMindmap
                  diagram={diagram}
                  fiberName={selectedFiberSummary?.summary}
                  onSelectNeuron={(id, content, type) =>
                    setSelectedNeuron({ id, content, type })
                  }
                />
              ) : null}
            </CardContent>
          </Card>

          {/* Neuron detail — bottom bar when selected */}
          {selectedNeuron && (
            <Card className="shrink-0">
              <CardContent className="flex items-start gap-4 p-3">
                <Badge variant="secondary" className="shrink-0 mt-0.5">
                  {selectedNeuron.type}
                </Badge>
                <p className="flex-1 text-sm leading-relaxed">
                  {selectedNeuron.content}
                </p>
                <p className="shrink-0 font-mono text-[10px] text-muted-foreground mt-0.5">
                  {selectedNeuron.id.slice(0, 12)}...
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
