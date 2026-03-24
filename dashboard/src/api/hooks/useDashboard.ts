import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/api/client"
import type {
  DashboardStats,
  BrainSummary,
  BrainSwitchResponse,
  HealthReport,
  HealthCheckResponse,
  TimelineResponse,
  DailyStats,
  EvolutionResponse,
  FiberListResponse,
  FiberDiagramResponse,
  GraphResponse,
  BrainFilesResponse,
  ToolStatsResponse,
  ConfigStatusResponse,
  EmbeddingConfigResponse,
  ConfigUpdateRequest,
  ConfigUpdateResponse,
  EmbeddingTestResponse,
  WatcherStatusResponse,
  VisualizeRequest,
  VisualizeResponse,
} from "@/api/types"

// Keys
const keys = {
  stats: ["dashboard", "stats"] as const,
  brains: ["dashboard", "brains"] as const,
  health: ["dashboard", "health"] as const,
  healthCheck: ["health"] as const,
  toolStats: (days: number) => ["dashboard", "tool-stats", days] as const,
  timeline: (limit: number, offset: number) =>
    ["dashboard", "timeline", limit, offset] as const,
  dailyStats: (days: number) => ["dashboard", "timeline", "daily-stats", days] as const,
  evolution: ["dashboard", "evolution"] as const,
  fibers: ["dashboard", "fibers"] as const,
  fiberDiagram: (id: string) => ["dashboard", "fiber", id, "diagram"] as const,
  graph: (limit: number) => ["graph", limit] as const,
  brainFiles: ["dashboard", "brain-files"] as const,
  configStatus: ["dashboard", "config-status"] as const,
  embeddingConfig: ["config", "embedding"] as const,
  watcherStatus: ["dashboard", "watcher-status"] as const,
}

// Stats
export function useStats() {
  return useQuery({
    queryKey: keys.stats,
    queryFn: () => api.get<DashboardStats>("/api/dashboard/stats"),
  })
}

// Health check (for version)
export function useHealthCheck() {
  return useQuery({
    queryKey: keys.healthCheck,
    queryFn: () => api.get<HealthCheckResponse>("/health"),
    staleTime: 300_000,
  })
}

// Brains list
export function useBrains() {
  return useQuery({
    queryKey: keys.brains,
    queryFn: () => api.get<BrainSummary[]>("/api/dashboard/brains"),
  })
}

// Switch brain
export function useSwitchBrain() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (brainName: string) =>
      api.post<BrainSwitchResponse>("/api/dashboard/brains/switch", {
        brain_name: brainName,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries()
    },
  })
}

// Health
export function useHealth() {
  return useQuery({
    queryKey: keys.health,
    queryFn: () => api.get<HealthReport>("/api/dashboard/health"),
  })
}

// Timeline
export function useTimeline(limit = 100, offset = 0) {
  return useQuery({
    queryKey: keys.timeline(limit, offset),
    queryFn: () =>
      api.get<TimelineResponse>(
        `/api/dashboard/timeline?limit=${limit}&offset=${offset}`,
      ),
  })
}

// Daily Stats (timeline charts)
export function useDailyStats(days = 30) {
  return useQuery({
    queryKey: keys.dailyStats(days),
    queryFn: () =>
      api.get<DailyStats[]>(
        `/api/dashboard/timeline/daily-stats?days=${days}`,
      ),
  })
}

// Evolution
export function useEvolution() {
  return useQuery({
    queryKey: keys.evolution,
    queryFn: () => api.get<EvolutionResponse>("/api/dashboard/evolution"),
  })
}

// Fibers list
export function useFibers() {
  return useQuery({
    queryKey: keys.fibers,
    queryFn: () => api.get<FiberListResponse>("/api/dashboard/fibers"),
  })
}

// Fiber diagram
export function useFiberDiagram(fiberId: string) {
  return useQuery({
    queryKey: keys.fiberDiagram(fiberId),
    queryFn: () =>
      api.get<FiberDiagramResponse>(
        `/api/dashboard/fiber/${fiberId}/diagram`,
      ),
    enabled: !!fiberId,
  })
}

// Graph
export function useGraph(limit = 500) {
  return useQuery({
    queryKey: keys.graph(limit),
    queryFn: () => api.get<GraphResponse>(`/api/graph?limit=${limit}`),
  })
}

// Delete brain
export function useDeleteBrain() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (brainId: string) =>
      api.delete<{ status: string; brain_id: string }>(`/brain/${brainId}`),
    onSuccess: () => {
      queryClient.invalidateQueries()
    },
  })
}

// Tool stats
export function useToolStats(days = 30) {
  return useQuery({
    queryKey: keys.toolStats(days),
    queryFn: () =>
      api.get<ToolStatsResponse>(
        `/api/dashboard/tool-stats?days=${days}`,
      ),
  })
}

// Brain files
export function useBrainFiles() {
  return useQuery({
    queryKey: keys.brainFiles,
    queryFn: () => api.get<BrainFilesResponse>("/api/dashboard/brain-files"),
  })
}

// Config status
export function useConfigStatus() {
  return useQuery({
    queryKey: keys.configStatus,
    queryFn: () => api.get<ConfigStatusResponse>("/api/dashboard/config-status"),
    staleTime: 60_000,
  })
}

// Embedding config
export function useEmbeddingConfig() {
  return useQuery({
    queryKey: keys.embeddingConfig,
    queryFn: () => api.get<EmbeddingConfigResponse>("/api/dashboard/config/embedding"),
    staleTime: 60_000,
  })
}

// Update config
export function useUpdateConfig() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: ConfigUpdateRequest) =>
      api.put<ConfigUpdateResponse>("/api/dashboard/config", body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.configStatus })
      queryClient.invalidateQueries({ queryKey: keys.embeddingConfig })
    },
  })
}

// Watcher status
export function useWatcherStatus() {
  return useQuery({
    queryKey: keys.watcherStatus,
    queryFn: () => api.get<WatcherStatusResponse>("/api/dashboard/watcher/status"),
    staleTime: 30_000,
  })
}

// Test embedding
export function useTestEmbedding() {
  return useMutation({
    mutationFn: () =>
      api.post<EmbeddingTestResponse>("/api/dashboard/config/embedding/test", {}),
  })
}

// Visualize memory data as chart
export function useVisualize() {
  return useMutation({
    mutationFn: (req: VisualizeRequest) =>
      api.post<VisualizeResponse>("/api/dashboard/visualize", req),
  })
}
