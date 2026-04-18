import { useEffect, useMemo, useRef } from "react"
import Graph from "graphology"
import Sigma from "sigma"
import forceAtlas2 from "graphology-layout-forceatlas2"
import type { GraphResponse } from "@/api/types"
import { NEURON_TYPE_COLORS as TYPE_COLORS } from "@/lib/neuron-colors"

interface NetworkGraphProps {
  data: GraphResponse
  height?: string
  onNodeClick?: (nodeId: string, content: string, type: string) => void
}

export function NetworkGraph({ data, height = "500px", onNodeClick }: NetworkGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sigmaRef = useRef<Sigma | null>(null)

  const graph = useMemo(() => {
    const g = new Graph()

    // Count connections per node for sizing
    const connectionCount = new Map<string, number>()
    for (const synapse of data.synapses) {
      connectionCount.set(synapse.source_id, (connectionCount.get(synapse.source_id) ?? 0) + 1)
      connectionCount.set(synapse.target_id, (connectionCount.get(synapse.target_id) ?? 0) + 1)
    }

    // Add nodes
    for (const neuron of data.neurons) {
      const connections = connectionCount.get(neuron.id) ?? 0
      const size = Math.min(Math.max(3, connections * 1.5), 15)
      g.addNode(neuron.id, {
        label: neuron.content.slice(0, 40) + (neuron.content.length > 40 ? "..." : ""),
        size,
        color: TYPE_COLORS[neuron.type] ?? TYPE_COLORS.other,
        x: Math.random() * 100,
        y: Math.random() * 100,
        // Store full data for click handler
        fullContent: neuron.content,
        nodeType: neuron.type,
      })
    }

    // Add edges (skip if source/target missing)
    const nodeSet = new Set(data.neurons.map((n) => n.id))
    for (const synapse of data.synapses) {
      if (nodeSet.has(synapse.source_id) && nodeSet.has(synapse.target_id)) {
        try {
          g.addEdge(synapse.source_id, synapse.target_id, {
            size: Math.max(0.5, synapse.weight * 2),
            color: "rgba(128, 128, 128, 0.3)",
          })
        } catch {
          // Skip duplicate edges
        }
      }
    }

    // Run ForceAtlas2 layout (synchronous, limited iterations)
    if (g.order > 0) {
      forceAtlas2.assign(g, {
        iterations: 100,
        settings: {
          gravity: 1,
          scalingRatio: 10,
          barnesHutOptimize: g.order > 200,
          strongGravityMode: true,
        },
      })
    }

    return g
  }, [data])

  useEffect(() => {
    if (!containerRef.current || graph.order === 0) return

    // Clean up previous instance
    if (sigmaRef.current) {
      sigmaRef.current.kill()
    }

    const sigma = new Sigma(graph, containerRef.current, {
      renderEdgeLabels: false,
      defaultEdgeType: "line",
      labelFont: "Inter, sans-serif",
      labelSize: 12,
      labelColor: { color: "#78716c" },
      defaultNodeColor: "#a8a29e",
      defaultEdgeColor: "rgba(128, 128, 128, 0.2)",
    })

    sigma.on("clickNode", ({ node }) => {
      const attrs = graph.getNodeAttributes(node)
      onNodeClick?.(node, attrs.fullContent ?? "", attrs.nodeType ?? "")
    })

    sigmaRef.current = sigma

    return () => {
      sigma.kill()
      sigmaRef.current = null
    }
  }, [graph, onNodeClick])

  return (
    <div
      ref={containerRef}
      style={{ height, width: "100%" }}
      className="rounded-lg border border-border bg-muted/20"
    />
  )
}
