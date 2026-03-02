import { useGraph } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

export default function GraphPage() {
  const { data: graph, isLoading } = useGraph(500)

  return (
    <div className="space-y-6 p-6">
      <h1 className="font-display text-2xl font-bold">Neural Graph</h1>

      <Card className="min-h-[600px]">
        <CardHeader>
          <CardTitle>
            Network Visualization
            {graph && (
              <span className="ml-2 text-sm font-normal text-muted-foreground">
                {graph.total_neurons.toLocaleString()} neurons,{" "}
                {graph.total_synapses.toLocaleString()} synapses
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-[500px] w-full" />
          ) : (
            <div className="flex h-[500px] items-center justify-center rounded-lg border border-border bg-muted/30">
              <p className="text-sm text-muted-foreground">
                Sigma.js graph visualization — coming in Phase 2
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
