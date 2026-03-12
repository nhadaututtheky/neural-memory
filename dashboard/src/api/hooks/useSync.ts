import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/api/client"
import type { SyncStatusResponse, SyncConfigUpdateResponse } from "@/api/types"

const keys = {
  status: ["sync", "status"] as const,
}

export function useSyncStatus() {
  return useQuery({
    queryKey: keys.status,
    queryFn: () => api.get<SyncStatusResponse>("/api/dashboard/sync-status"),
    refetchInterval: 30_000,
  })
}

export function useUpdateSyncConfig() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api.post<SyncConfigUpdateResponse>("/api/dashboard/sync-config", body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.status })
    },
  })
}
