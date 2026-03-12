import type { GraphNeuron, GraphSynapse } from "@/api/types"
import { CARD_SUITS, DEFAULT_SUIT, type OracleCard, type CardSuit, type CardSuitKey } from "./types"

function getSuit(neuronType: string): { suit: CardSuit; key: CardSuitKey | "unknown" } {
  if (neuronType in CARD_SUITS) {
    return { suit: CARD_SUITS[neuronType as CardSuitKey], key: neuronType as CardSuitKey }
  }
  return { suit: DEFAULT_SUIT, key: "unknown" as const }
}

function formatAge(createdAt: string): string {
  const created = new Date(createdAt)
  const now = new Date()
  const diffMs = now.getTime() - created.getTime()
  const days = Math.floor(diffMs / (1000 * 60 * 60 * 24))
  if (days === 0) return "today"
  if (days === 1) return "1d"
  if (days < 30) return `${days}d`
  if (days < 365) return `${Math.floor(days / 30)}mo`
  return `${Math.floor(days / 365)}y`
}

function countConnections(neuronId: string, synapses: GraphSynapse[]): number {
  return synapses.filter(
    (s) => s.source_id === neuronId || s.target_id === neuronId,
  ).length
}

function truncateContent(content: string, maxLen = 120): string {
  if (content.length <= maxLen) return content
  return content.slice(0, maxLen).trimEnd() + "..."
}

export function neuronsToCards(
  neurons: GraphNeuron[],
  synapses: GraphSynapse[],
): OracleCard[] {
  return neurons.map((neuron) => {
    const { suit, key } = getSuit(neuron.type)
    const meta = neuron.metadata ?? {}
    const activation = typeof meta.activation_level === "number" ? meta.activation_level : 0.5
    const priority = typeof meta.priority === "number" ? meta.priority : 5
    const createdAt = typeof meta.created_at === "string" ? meta.created_at : new Date().toISOString()

    return {
      id: neuron.id,
      title: suit.name,
      content: truncateContent(neuron.content),
      suit,
      suitKey: key,
      activation: Math.round(activation * 100) / 10,
      connectionCount: countConnections(neuron.id, synapses),
      age: formatAge(createdAt),
      priority,
      createdAt,
    }
  })
}
