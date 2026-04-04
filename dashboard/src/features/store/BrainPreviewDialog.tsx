import { useState } from "react"
import {
  Atom,
  TreeStructure,
  GitBranch,
  User,
  Shield,
  ShieldWarning,
  Star,
  DownloadSimple,
  Warning,
  Tag,
  Certificate,
  Hash,
} from "@phosphor-icons/react"
import { Dialog, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { useBrainPreview, useImportRemoteBrain, useRateBrain } from "@/api/hooks/useStore"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"

interface BrainPreviewDialogProps {
  brainName: string | null
  registryRepo?: string
  onClose: () => void
}

export function BrainPreviewDialog({
  brainName,
  onClose,
}: BrainPreviewDialogProps) {
  const { t } = useTranslation()
  const { data: preview, isLoading, error } = useBrainPreview(brainName)
  const importMutation = useImportRemoteBrain()
  const rateMutation = useRateBrain()
  const [userRating, setUserRating] = useState(0)
  const [hoverRating, setHoverRating] = useState(0)

  const handleImport = () => {
    if (!brainName) return
    const url = `https://raw.githubusercontent.com/neural-memory/brain-store/main/brains/${brainName}/brain.json`
    importMutation.mutate(url, {
      onSuccess: (data) => {
        toast.success(
          t("store.importSuccess", {
            name: data.brain_name,
            neurons: data.neurons_imported,
          }),
        )
        onClose()
      },
      onError: () => {
        toast.error(t("store.importError"))
      },
    })
  }

  const handleRate = (rating: number) => {
    if (!preview?.manifest.id) return
    setUserRating(rating)
    rateMutation.mutate({
      brain_package_id: preview.manifest.id,
      rating,
    })
  }

  return (
    <Dialog open={!!brainName} onClose={onClose}>
      {isLoading && (
        <div className="space-y-4">
          <Skeleton className="h-6 w-2/3" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      )}

      {error && (
        <div className="text-center py-8">
          <p className="text-sm text-destructive">{t("store.previewError")}</p>
          <Button variant="outline" size="sm" onClick={onClose} className="mt-4">
            {t("common.cancel")}
          </Button>
        </div>
      )}

      {preview && (
        <>
          <DialogHeader>
            <div className="flex items-start justify-between gap-3">
              <div>
                <DialogTitle>{preview.manifest.display_name}</DialogTitle>
                <p className="mt-1 text-xs text-muted-foreground inline-flex items-center gap-1.5">
                  <User className="size-3" aria-hidden="true" />
                  {preview.manifest.author}
                  <span className="mx-1">·</span>
                  v{preview.manifest.version}
                </p>
              </div>
              {preview.scan_result.safe ? (
                <Badge variant="success" className="shrink-0 inline-flex items-center gap-1">
                  <Shield className="size-3" aria-hidden="true" />
                  {t("store.safe")}
                </Badge>
              ) : (
                <Badge variant="warning" className="shrink-0 inline-flex items-center gap-1">
                  <ShieldWarning className="size-3" aria-hidden="true" />
                  {preview.scan_result.risk_level}
                </Badge>
              )}
            </div>
          </DialogHeader>

          {/* Description */}
          <p className="text-sm text-muted-foreground">{preview.manifest.description}</p>

          {/* Scan Warnings */}
          {preview.scan_result.findings.length > 0 && (
            <div className="mt-3 rounded-lg border border-health-warn/30 bg-health-warn/5 p-3">
              <p className="text-xs font-semibold text-health-warn inline-flex items-center gap-1.5 mb-2">
                <Warning className="size-3.5" aria-hidden="true" />
                {t("store.scanWarnings", { count: preview.scan_result.finding_count })}
              </p>
              <ul className="space-y-1">
                {preview.scan_result.findings.slice(0, 5).map((f, i) => (
                  <li key={i} className="text-xs text-muted-foreground">
                    <Badge
                      variant={f.severity === "high" ? "destructive" : "outline"}
                      className="text-[10px] mr-1.5"
                    >
                      {f.severity}
                    </Badge>
                    {f.description}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Stats */}
          <div className="mt-4 grid grid-cols-3 gap-3">
            <StatBox
              icon={<Atom className="size-4" aria-hidden="true" />}
              label={t("store.neurons")}
              value={preview.manifest.stats.neuron_count}
            />
            <StatBox
              icon={<TreeStructure className="size-4" aria-hidden="true" />}
              label={t("store.synapses")}
              value={preview.manifest.stats.synapse_count}
            />
            <StatBox
              icon={<GitBranch className="size-4" aria-hidden="true" />}
              label={t("store.fibers")}
              value={preview.manifest.stats.fiber_count}
            />
          </div>

          {/* Neuron Type Breakdown */}
          {Object.keys(preview.neuron_type_breakdown).length > 0 && (
            <div className="mt-4">
              <h4 className="text-xs font-semibold text-muted-foreground mb-2">
                {t("store.neuronTypes")}
              </h4>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(preview.neuron_type_breakdown).map(([type, count]) => (
                  <Badge key={type} variant="secondary" className="text-[10px]">
                    {type}: {count}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Sample Neurons */}
          {preview.sample_neurons.length > 0 && (
            <div className="mt-4">
              <h4 className="text-xs font-semibold text-muted-foreground mb-2">
                {t("store.sampleNeurons")}
              </h4>
              <div className="max-h-48 space-y-1.5 overflow-y-auto rounded-lg border border-border p-2">
                {preview.sample_neurons.map((n) => (
                  <div
                    key={n.id}
                    className="rounded-md bg-muted/50 px-2.5 py-1.5 text-xs text-card-foreground"
                  >
                    <Badge variant="outline" className="text-[9px] mr-1.5 align-middle">
                      {n.type}
                    </Badge>
                    {n.content}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tags */}
          {preview.top_tags.length > 0 && (
            <div className="mt-4 flex flex-wrap items-center gap-1.5">
              <Tag className="size-3.5 text-muted-foreground" aria-hidden="true" />
              {preview.top_tags.map((tag) => (
                <Badge key={tag} variant="outline" className="text-[10px]">
                  {tag}
                </Badge>
              ))}
            </div>
          )}

          {/* Metadata Row */}
          <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <Certificate className="size-3" aria-hidden="true" />
              {preview.manifest.license}
            </span>
            <span className="inline-flex items-center gap-1">
              <Hash className="size-3" aria-hidden="true" />
              {preview.manifest.content_hash.slice(0, 15)}...
            </span>
            <span>
              {(preview.manifest.size_bytes / 1024).toFixed(1)} KB
            </span>
          </div>

          {/* Rating */}
          <div className="mt-4 flex items-center gap-2">
            <span className="text-xs text-muted-foreground">{t("store.rateThis")}:</span>
            <div className="flex gap-0.5">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => handleRate(star)}
                  onMouseEnter={() => setHoverRating(star)}
                  onMouseLeave={() => setHoverRating(0)}
                  className="cursor-pointer p-0.5 transition-transform hover:scale-110"
                  aria-label={`${star} stars`}
                >
                  <Star
                    className="size-4"
                    weight={(hoverRating || userRating) >= star ? "fill" : "regular"}
                    color={(hoverRating || userRating) >= star ? "#f59e0b" : undefined}
                  />
                </button>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="mt-6 flex justify-end gap-3 border-t border-border pt-4">
            <Button variant="outline" size="sm" onClick={onClose}>
              {t("common.cancel")}
            </Button>
            <Button
              size="sm"
              onClick={handleImport}
              disabled={importMutation.isPending || !preview.scan_result.safe}
            >
              {importMutation.isPending ? (
                <span className="inline-flex items-center gap-1.5">
                  <span className="size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  {t("store.importing")}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5">
                  <DownloadSimple className="size-4" aria-hidden="true" />
                  {t("store.importBrain")}
                </span>
              )}
            </Button>
          </div>
        </>
      )}
    </Dialog>
  )
}

function StatBox({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: number
}) {
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-2.5 text-center">
      <div className="flex items-center justify-center gap-1.5 text-muted-foreground">
        {icon}
        <span className="text-[10px] uppercase tracking-wider">{label}</span>
      </div>
      <p className="mt-1 font-mono text-lg font-bold text-card-foreground">
        {value.toLocaleString()}
      </p>
    </div>
  )
}
