import { useQuery, useMutation } from "@tanstack/react-query"
import { api } from "@/api/client"
import type {
  TelegramStatus,
  TelegramTestResponse,
  TelegramBackupResponse,
} from "@/api/types"

const keys = {
  status: ["telegram", "status"] as const,
}

export function useTelegramStatus() {
  return useQuery({
    queryKey: keys.status,
    queryFn: () => api.get<TelegramStatus>("/api/dashboard/telegram/status"),
    retry: false,
  })
}

export function useTelegramTest() {
  return useMutation({
    mutationFn: () =>
      api.post<TelegramTestResponse>("/api/dashboard/telegram/test", {}),
  })
}

export function useTelegramBackup() {
  return useMutation({
    mutationFn: (brain?: string) =>
      api.post<TelegramBackupResponse>("/api/dashboard/telegram/backup", {
        brain,
      }),
  })
}
