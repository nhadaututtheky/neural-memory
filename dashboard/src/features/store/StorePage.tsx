import { useState, useDeferredValue } from "react"
import { Storefront, Package } from "@phosphor-icons/react"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { useRegistry } from "@/api/hooks/useStore"
import { useTranslation } from "react-i18next"
import { StoreFilters } from "./StoreFilters"
import { BrainCard } from "./BrainCard"
import { BrainPreviewDialog } from "./BrainPreviewDialog"

export default function StorePage() {
  const { t } = useTranslation()
  const [search, setSearch] = useState("")
  const [category, setCategory] = useState("")
  const [sortBy, setSortBy] = useState("created_at")
  const [previewName, setPreviewName] = useState<string | null>(null)

  const deferredSearch = useDeferredValue(search)

  const { data, isLoading, error, refetch } = useRegistry({
    category: category || undefined,
    search: deferredSearch || undefined,
    sort_by: sortBy,
  })

  const brains = data?.brains ?? []

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="font-display text-2xl font-bold text-foreground inline-flex items-center gap-2.5">
          <Storefront className="size-7" aria-hidden="true" />
          {t("store.title")}
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {t("store.subtitle")}
        </p>
      </div>

      {/* Filters */}
      <StoreFilters
        search={search}
        onSearchChange={setSearch}
        category={category}
        onCategoryChange={setCategory}
        sortBy={sortBy}
        onSortChange={setSortBy}
      />

      {/* Content */}
      {isLoading && <StoreGridSkeleton />}

      {error && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <p className="text-sm text-destructive">{t("store.fetchError")}</p>
          <Button variant="outline" size="sm" onClick={() => refetch()} className="mt-4">
            {t("store.retry")}
          </Button>
        </div>
      )}

      {!isLoading && !error && brains.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Package className="size-12 text-muted-foreground/40" aria-hidden="true" />
          <p className="mt-3 text-sm text-muted-foreground">{t("store.empty")}</p>
          {(search || category) && (
            <Button
              variant="outline"
              size="sm"
              className="mt-3"
              onClick={() => {
                setSearch("")
                setCategory("")
              }}
            >
              {t("store.clearFilters")}
            </Button>
          )}
        </div>
      )}

      {!isLoading && !error && brains.length > 0 && (
        <>
          <p className="text-xs text-muted-foreground">
            {t("store.showing", { count: brains.length })}
            {data?.cached && (
              <span className="ml-1.5 text-muted-foreground/50">({t("store.cached")})</span>
            )}
          </p>
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {brains.map((manifest) => (
              <BrainCard
                key={manifest.id ?? manifest.name}
                manifest={manifest}
                onClick={() => setPreviewName(manifest.name)}
              />
            ))}
          </div>
        </>
      )}

      {/* Preview Dialog */}
      <BrainPreviewDialog
        brainName={previewName}
        onClose={() => setPreviewName(null)}
      />
    </div>
  )
}

function StoreGridSkeleton() {
  return (
    <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="rounded-xl border border-border bg-card p-4 space-y-3">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-2/3" />
          <div className="flex gap-2 pt-2">
            <Skeleton className="h-5 w-12" />
            <Skeleton className="h-5 w-12" />
            <Skeleton className="h-5 w-12" />
          </div>
        </div>
      ))}
    </div>
  )
}
