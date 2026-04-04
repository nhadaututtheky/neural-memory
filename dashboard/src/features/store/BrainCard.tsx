import { Atom, TreeStructure, GitBranch, User, Star } from "@phosphor-icons/react"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import type { BrainManifest } from "@/api/types"
import { useTranslation } from "react-i18next"

const SIZE_TIER_STYLES = {
  micro: "success",
  small: "secondary",
  medium: "warning",
} as const

interface BrainCardProps {
  manifest: BrainManifest
  onClick: () => void
}

export function BrainCard({ manifest, onClick }: BrainCardProps) {
  const { t } = useTranslation()
  const tierVariant = SIZE_TIER_STYLES[manifest.size_tier as keyof typeof SIZE_TIER_STYLES] ?? "outline"

  return (
    <Card
      className="group flex flex-col p-4 transition-transform duration-200 hover:scale-[1.02] cursor-pointer"
      onClick={onClick}
      role="button"
      tabIndex={0}
      aria-label={`${manifest.display_name} by ${manifest.author}`}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onClick()
        }
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <h3 className="font-display text-sm font-bold text-card-foreground line-clamp-1">
          {manifest.display_name}
        </h3>
        <Badge variant={tierVariant as "success" | "secondary" | "warning" | "outline"} className="shrink-0 text-[10px]">
          {manifest.size_tier}
        </Badge>
      </div>

      {/* Description */}
      <p className="mt-1.5 text-xs text-muted-foreground line-clamp-2 flex-1">
        {manifest.description}
      </p>

      {/* Stats Row */}
      <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1" title={t("store.neurons")}>
          <Atom className="size-3.5" aria-hidden="true" />
          {manifest.stats.neuron_count}
        </span>
        <span className="inline-flex items-center gap-1" title={t("store.synapses")}>
          <TreeStructure className="size-3.5" aria-hidden="true" />
          {manifest.stats.synapse_count}
        </span>
        <span className="inline-flex items-center gap-1" title={t("store.fibers")}>
          <GitBranch className="size-3.5" aria-hidden="true" />
          {manifest.stats.fiber_count}
        </span>
      </div>

      {/* Tags */}
      <div className="mt-2.5 flex flex-wrap gap-1">
        {manifest.tags.slice(0, 3).map((tag) => (
          <Badge key={tag} variant="outline" className="text-[10px] px-1.5 py-0">
            {tag}
          </Badge>
        ))}
        {manifest.tags.length > 3 && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
            +{manifest.tags.length - 3}
          </Badge>
        )}
      </div>

      {/* Footer: Author + Rating */}
      <div className="mt-3 flex items-center justify-between border-t border-border pt-2.5 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <User className="size-3" aria-hidden="true" />
          {manifest.author}
        </span>
        {manifest.rating_count > 0 && (
          <span className="inline-flex items-center gap-1">
            <Star className="size-3 text-amber-500" weight="fill" aria-hidden="true" />
            {manifest.rating_avg.toFixed(1)}
            <span className="text-muted-foreground/60">({manifest.rating_count})</span>
          </span>
        )}
      </div>
    </Card>
  )
}
