import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/api/client"
import type {
  StoreRegistryResponse,
  BrainPreview,
  StoreImportResponse,
  StoreExportResponse,
  StoreRatingResponse,
} from "@/api/types"

const keys = {
  registry: (category?: string, search?: string, tag?: string, sortBy?: string) =>
    ["store", "registry", category, search, tag, sortBy] as const,
  preview: (name: string) => ["store", "preview", name] as const,
  ratings: (id: string) => ["store", "ratings", id] as const,
}

export function useRegistry(params?: {
  category?: string
  search?: string
  tag?: string
  sort_by?: string
}) {
  const qs = new URLSearchParams()
  if (params?.category) qs.set("category", params.category)
  if (params?.search) qs.set("search", params.search)
  if (params?.tag) qs.set("tag", params.tag)
  if (params?.sort_by) qs.set("sort_by", params.sort_by)
  const query = qs.toString()

  return useQuery({
    queryKey: keys.registry(params?.category, params?.search, params?.tag, params?.sort_by),
    queryFn: () =>
      api.get<StoreRegistryResponse>(
        `/api/dashboard/store/registry${query ? `?${query}` : ""}`,
      ),
  })
}

export function useBrainPreview(name: string | null) {
  return useQuery({
    queryKey: keys.preview(name ?? ""),
    queryFn: () => api.get<BrainPreview>(`/api/dashboard/store/registry/preview/${name}`),
    enabled: !!name,
  })
}

export function useImportRemoteBrain() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (sourceUrl: string) =>
      api.post<StoreImportResponse>("/api/dashboard/store/import-remote", {
        source_url: sourceUrl,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dashboard", "stats"] })
      qc.invalidateQueries({ queryKey: ["dashboard", "brains"] })
    },
  })
}

export function useExportBrain() {
  return useMutation({
    mutationFn: (data: {
      display_name: string
      description: string
      author: string
      category: string
      tags: string[]
    }) => api.post<StoreExportResponse>("/api/dashboard/store/export", data),
  })
}

export function useRateBrain() {
  return useMutation({
    mutationFn: (data: { brain_package_id: string; rating: number; comment?: string }) =>
      api.post<StoreRatingResponse>("/api/dashboard/store/rate", data),
  })
}
