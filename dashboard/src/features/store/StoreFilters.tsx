import { MagnifyingGlass, SortAscending } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { useTranslation } from "react-i18next"

const CATEGORIES = [
  { value: "", labelKey: "store.categoryAll" },
  { value: "programming", labelKey: "store.categoryProgramming" },
  { value: "devops", labelKey: "store.categoryDevops" },
  { value: "writing", labelKey: "store.categoryWriting" },
  { value: "science", labelKey: "store.categoryScience" },
  { value: "security", labelKey: "store.categorySecurity" },
  { value: "data", labelKey: "store.categoryData" },
  { value: "personal", labelKey: "store.categoryPersonal" },
] as const

const SORT_OPTIONS = [
  { value: "created_at", labelKey: "store.sortNewest" },
  { value: "rating_avg", labelKey: "store.sortTopRated" },
  { value: "download_count", labelKey: "store.sortPopular" },
] as const

interface StoreFiltersProps {
  search: string
  onSearchChange: (value: string) => void
  category: string
  onCategoryChange: (value: string) => void
  sortBy: string
  onSortChange: (value: string) => void
}

export function StoreFilters({
  search,
  onSearchChange,
  category,
  onCategoryChange,
  sortBy,
  onSortChange,
}: StoreFiltersProps) {
  const { t } = useTranslation()

  return (
    <div className="space-y-4">
      {/* Search + Sort Row */}
      <div className="flex gap-3">
        <div className="relative flex-1">
          <MagnifyingGlass
            className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground"
            aria-hidden="true"
          />
          <input
            type="text"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder={t("store.searchPlaceholder")}
            className="h-9 w-full rounded-lg border border-border bg-background pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-2 focus:outline-offset-1 focus:outline-ring"
            aria-label={t("store.searchPlaceholder")}
          />
        </div>
        <div className="relative">
          <SortAscending
            className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground pointer-events-none"
            aria-hidden="true"
          />
          <select
            value={sortBy}
            onChange={(e) => onSortChange(e.target.value)}
            className="h-9 appearance-none rounded-lg border border-border bg-background pl-9 pr-8 text-sm text-foreground cursor-pointer focus:outline-2 focus:outline-offset-1 focus:outline-ring"
            aria-label={t("store.sortLabel")}
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {t(opt.labelKey)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Category Tabs */}
      <div className="flex gap-1.5 overflow-x-auto pb-1 scrollbar-none" role="tablist">
        {CATEGORIES.map((cat) => (
          <button
            key={cat.value}
            role="tab"
            aria-selected={category === cat.value}
            onClick={() => onCategoryChange(cat.value)}
            className={cn(
              "shrink-0 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors cursor-pointer",
              category === cat.value
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-foreground",
            )}
          >
            {t(cat.labelKey)}
          </button>
        ))}
      </div>
    </div>
  )
}
