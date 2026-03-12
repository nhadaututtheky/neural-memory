import { useMemo } from "react"
import { useGraph } from "@/api/hooks/useDashboard"
import { neuronsToCards } from "../engine/card-generator"
import type { OracleCard } from "../engine/types"

export function useOracleData(): {
  cards: OracleCard[]
  isLoading: boolean
  error: Error | null
} {
  const { data, isLoading, error } = useGraph(500)

  const cards = useMemo(() => {
    if (!data) return []
    return neuronsToCards(data.neurons, data.synapses)
  }, [data])

  return { cards, isLoading, error: error as Error | null }
}
