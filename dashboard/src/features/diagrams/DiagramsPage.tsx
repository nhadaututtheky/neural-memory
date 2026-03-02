import { useState } from "react"
import { useFibers, useFiberDiagram } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

export default function DiagramsPage() {
  const { data: fiberList, isLoading: fibersLoading } = useFibers()
  const [selectedFiber, setSelectedFiber] = useState("")
  const { data: diagram, isLoading: diagramLoading } =
    useFiberDiagram(selectedFiber)

  return (
    <div className="space-y-6 p-6">
      <h1 className="font-display text-2xl font-bold">Diagrams</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Fiber selector */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Fibers</CardTitle>
          </CardHeader>
          <CardContent>
            {fibersLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : fiberList?.fibers && fiberList.fibers.length > 0 ? (
              <div className="space-y-1 max-h-[500px] overflow-y-auto">
                {fiberList.fibers.map((fiber) => (
                  <button
                    key={fiber.id}
                    onClick={() => setSelectedFiber(fiber.id)}
                    className={`w-full cursor-pointer rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                      selectedFiber === fiber.id
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-accent"
                    }`}
                  >
                    <p className="font-medium truncate">{fiber.summary}</p>
                    <p className="text-xs text-muted-foreground">
                      {fiber.neuron_count} neurons
                    </p>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No fibers found.</p>
            )}
          </CardContent>
        </Card>

        {/* Diagram view */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>
              {diagram ? `Fiber: ${diagram.fiber_id.slice(0, 12)}...` : "Select a Fiber"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!selectedFiber ? (
              <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
                Select a fiber to view its structure
              </div>
            ) : diagramLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : diagram ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Neurons</p>
                    <p className="font-mono text-lg font-bold">
                      {diagram.neurons.length}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Connections</p>
                    <p className="font-mono text-lg font-bold">
                      {diagram.synapses.length}
                    </p>
                  </div>
                </div>
                <div className="flex h-48 items-center justify-center rounded-lg border border-border bg-muted/30">
                  <p className="text-sm text-muted-foreground">
                    Sigma.js subgraph visualization — coming in Phase 2
                  </p>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
