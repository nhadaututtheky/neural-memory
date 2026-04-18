import { useMemo } from "react"
import { useGraph } from "@/api/hooks/useDashboard"
import { runForceLayout } from "../engine/force-3d"
import {
  type BrainLayout,
  type PositionedNeuron,
  type PositionedEdge,
  computeConnectionCount,
  radiusForConnections,
  neuronColor,
} from "../engine/types"

interface UseLivingBrainResult {
  layout: BrainLayout | null
  isLoading: boolean
  error: unknown
}

const layoutCache = new WeakMap<object, BrainLayout>()

export function useLivingBrain(limit = 500): UseLivingBrainResult {
  const { data, isLoading, error } = useGraph(limit)

  const layout = useMemo<BrainLayout | null>(() => {
    if (!data || data.neurons.length === 0) return null
    const cached = layoutCache.get(data)
    if (cached) return cached

    const connectionCount = computeConnectionCount(data.synapses)

    const layoutResult = runForceLayout(
      data.neurons.map((n) => ({ id: n.id, type: n.type })),
      data.synapses.map((s) => ({
        source: s.source_id,
        target: s.target_id,
        weight: s.weight,
      })),
      { iterations: 200 },
    )

    const positionById = new Map(layoutResult.nodes.map((n) => [n.id, n]))
    const neurons: PositionedNeuron[] = data.neurons.map((n) => {
      const pos = positionById.get(n.id)
      const c = connectionCount.get(n.id) ?? 0
      return {
        id: n.id,
        content: n.content,
        type: n.type,
        x: pos?.x ?? 0,
        y: pos?.y ?? 0,
        z: pos?.z ?? 0,
        radius: radiusForConnections(c),
        color: neuronColor(n.type),
      }
    })

    const indexById = new Map(neurons.map((n, i) => [n.id, i]))
    const edges: PositionedEdge[] = []
    const neighborWeights = new Map<string, Map<string, number>>()
    for (const s of data.synapses) {
      const si = indexById.get(s.source_id)
      const ti = indexById.get(s.target_id)
      if (si === undefined || ti === undefined) continue
      edges.push({ sourceIdx: si, targetIdx: ti, weight: s.weight })
      const srcMap = neighborWeights.get(s.source_id) ?? new Map<string, number>()
      srcMap.set(s.target_id, Math.max(srcMap.get(s.target_id) ?? 0, s.weight))
      neighborWeights.set(s.source_id, srcMap)
      const tgtMap = neighborWeights.get(s.target_id) ?? new Map<string, number>()
      tgtMap.set(s.source_id, Math.max(tgtMap.get(s.source_id) ?? 0, s.weight))
      neighborWeights.set(s.target_id, tgtMap)
    }

    const neighbors = new Map<string, ReadonlyArray<string>>()
    for (const [id, weights] of neighborWeights) {
      const sorted = [...weights.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([nid]) => nid)
      neighbors.set(id, sorted)
    }

    const result: BrainLayout = { neurons, edges, neighbors, indexById }
    layoutCache.set(data, result)
    return result
  }, [data])

  return { layout, isLoading, error }
}
