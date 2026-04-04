import { useQuery } from "@tanstack/react-query"
import { api } from "@/api/client"

interface VelocityMetrics {
  promoted: number
  demoted: number
  archived: number
}

interface TierAnalyticsResponse {
  breakdown_by_type: Record<string, Record<string, number>>
  velocity_7d: VelocityMetrics
  velocity_30d: VelocityMetrics
  total_memories: number
}

export interface TierChangeEvent {
  fiber_id: string
  memory_type: string
  from_tier: string
  to_tier: string
  reason: string
  at: string
}

interface TierHistoryResponse {
  events: TierChangeEvent[]
  total: number
  limit: number
  offset: number
}

const keys = {
  analytics: ["tier", "analytics"] as const,
  history: (limit: number) => ["tier", "history", limit] as const,
}

export function useTierAnalytics() {
  return useQuery({
    queryKey: keys.analytics,
    queryFn: () =>
      api.get<TierAnalyticsResponse>("/api/dashboard/tier-analytics"),
  })
}

export function useTierHistory(limit = 20) {
  return useQuery({
    queryKey: keys.history(limit),
    queryFn: () =>
      api.get<TierHistoryResponse>(
        `/api/dashboard/tier-history?limit=${limit}`,
      ),
  })
}
