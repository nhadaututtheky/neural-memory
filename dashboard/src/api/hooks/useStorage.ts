import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/api/client"
import type {
  StorageStatusResponse,
  MigrationJobStatus,
  SetBackendResponse,
  TierDistribution,
} from "@/api/types"

const keys = {
  status: ["storage", "status"] as const,
  job: (id: string) => ["storage", "job", id] as const,
  tierStats: ["storage", "tier-stats"] as const,
}

export function useStorageStatus() {
  return useQuery({
    queryKey: keys.status,
    queryFn: () => api.get<StorageStatusResponse>("/api/dashboard/storage/status"),
  })
}

export function useMigrationJob(jobId: string | null) {
  return useQuery({
    queryKey: keys.job(jobId ?? ""),
    queryFn: () =>
      api.get<MigrationJobStatus>(`/api/dashboard/storage/migrate/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const state = query.state.data?.state
      return state === "running" ? 1500 : false
    },
  })
}

export function useStartMigration() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (direction: "to_infinitydb" | "to_sqlite") =>
      api.post<{ job_id: string; brain: string; message: string; disk_warning?: string }>(
        "/api/dashboard/storage/migrate",
        { direction },
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.status })
    },
  })
}

export function useTierStats() {
  return useQuery({
    queryKey: keys.tierStats,
    queryFn: () => api.get<TierDistribution>("/api/dashboard/tier-stats"),
  })
}

export function useSetBackend() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (backend: "sqlite" | "infinitydb") =>
      api.post<SetBackendResponse>("/api/dashboard/storage/backend", {
        backend,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.status })
      queryClient.invalidateQueries({ queryKey: ["dashboard", "stats"] })
    },
  })
}
