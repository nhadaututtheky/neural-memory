import type { GraphNeuron, GraphSynapse } from "@/api/types"
import {
  NEURON_TYPE_COLORS,
  NEURON_DEFAULT_COLOR,
  colorForNeuronType,
} from "@/lib/neuron-colors"

export const TYPE_COLORS = NEURON_TYPE_COLORS
export const DEFAULT_COLOR = NEURON_DEFAULT_COLOR

export interface PositionedNeuron {
  id: string
  content: string
  type: string
  x: number
  y: number
  z: number
  radius: number
  color: string
}

export interface PositionedEdge {
  sourceIdx: number
  targetIdx: number
  weight: number
}

export interface BrainLayout {
  neurons: ReadonlyArray<Readonly<PositionedNeuron>>
  edges: ReadonlyArray<Readonly<PositionedEdge>>
  /** id → array of direct neighbor ids, sorted by synapse weight desc. */
  neighbors: ReadonlyMap<string, ReadonlyArray<string>>
  /** id → index in `neurons` array for O(1) instance lookup. */
  indexById: ReadonlyMap<string, number>
}

export function computeConnectionCount(
  synapses: ReadonlyArray<GraphSynapse>,
): Map<string, number> {
  const counts = new Map<string, number>()
  for (const s of synapses) {
    counts.set(s.source_id, (counts.get(s.source_id) ?? 0) + 1)
    counts.set(s.target_id, (counts.get(s.target_id) ?? 0) + 1)
  }
  return counts
}

export function radiusForConnections(n: number): number {
  return Math.min(Math.max(0.4, 0.15 + n * 0.08), 1.6)
}

export const neuronColor = colorForNeuronType

export function neuronKey(n: Pick<GraphNeuron, "id">): string {
  return n.id
}
