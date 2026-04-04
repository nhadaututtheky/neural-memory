import { useState } from "react"
import { useStats, useBrains, useSwitchBrain, useDeleteBrain } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { Brain, Lightning, Link, Stack, Trash, Export } from "@phosphor-icons/react"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"
import QuickActionsCard from "./QuickActionsCard"
import GuideCard from "./GuideCard"
import { ExportDialog } from "@/features/store/ExportDialog"

function KpiCard({
  label,
  value,
  icon: Icon,
  loading,
}: {
  label: string
  value: string | number
  icon: React.ElementType
  loading: boolean
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-4 p-6">
        <div className="flex size-12 items-center justify-center rounded-lg bg-primary/10">
          <Icon className="size-6 text-primary" aria-hidden="true" />
        </div>
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          {loading ? (
            <Skeleton className="mt-1 h-7 w-20" />
          ) : (
            <p className="font-mono text-2xl font-bold tracking-tight">
              {typeof value === "number" ? value.toLocaleString() : value}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default function OverviewPage() {
  const { data: stats, isLoading: statsLoading } = useStats()
  const { data: brains, isLoading: brainsLoading } = useBrains()
  const switchBrain = useSwitchBrain()
  const deleteBrain = useDeleteBrain()
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null)
  const [exportTarget, setExportTarget] = useState<string | null>(null)
  const { t } = useTranslation()

  const handleSwitchBrain = (brainName: string) => {
    switchBrain.mutate(brainName, {
      onSuccess: () => toast.success(t("overview.switchedTo", { name: brainName })),
      onError: () => toast.error(t("overview.switchFailed")),
    })
  }

  const handleDeleteBrain = () => {
    if (!deleteTarget) return
    deleteBrain.mutate(deleteTarget.id, {
      onSuccess: () => {
        toast.success(t("overview.deleted", { name: deleteTarget.name }))
        setDeleteTarget(null)
      },
      onError: () => {
        toast.error(t("overview.deleteFailed"))
        setDeleteTarget(null)
      },
    })
  }

  return (
    <div className="space-y-6 p-6">
      <h1 className="font-display text-2xl font-bold">{t("overview.title")}</h1>

      {/* Guide card for new users */}
      <GuideCard />

      {/* KPI Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          label={t("overview.neurons")}
          value={stats?.total_neurons ?? 0}
          icon={Brain}
          loading={statsLoading}
        />
        <KpiCard
          label={t("overview.synapses")}
          value={stats?.total_synapses ?? 0}
          icon={Link}
          loading={statsLoading}
        />
        <KpiCard
          label={t("overview.fibers")}
          value={stats?.total_fibers ?? 0}
          icon={Stack}
          loading={statsLoading}
        />
        <KpiCard
          label={t("overview.brains")}
          value={stats?.total_brains ?? 0}
          icon={Lightning}
          loading={statsLoading}
        />
      </div>

      {/* Quick Actions */}
      <QuickActionsCard />

      {/* Brain List */}
      <Card>
        <CardHeader>
          <CardTitle>{t("overview.brainList")}</CardTitle>
        </CardHeader>
        <CardContent>
          {brainsLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : brains && brains.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-muted-foreground">
                    <th className="pb-2 font-medium">{t("overview.name")}</th>
                    <th className="pb-2 font-medium">{t("overview.neurons")}</th>
                    <th className="pb-2 font-medium">{t("overview.synapses")}</th>
                    <th className="pb-2 font-medium">{t("overview.fibers")}</th>
                    <th className="pb-2 font-medium">{t("overview.grade")}</th>
                    <th className="pb-2 font-medium">{t("overview.status")}</th>
                    <th className="pb-2 font-medium">{t("overview.actions")}</th>
                  </tr>
                </thead>
                <tbody>
                  {brains.map((brain) => (
                    <tr
                      key={brain.id}
                      className={`border-b border-border/50 last:border-0 transition-colors ${
                        !brain.is_active
                          ? "cursor-pointer hover:bg-accent/50"
                          : ""
                      }`}
                      onClick={() => {
                        if (!brain.is_active) handleSwitchBrain(brain.name)
                      }}
                      title={
                        brain.is_active
                          ? t("overview.currentBrain")
                          : t("overview.switchTo", { name: brain.name })
                      }
                    >
                      <td className="py-3 font-mono font-medium">
                        {brain.name}
                      </td>
                      <td className="py-3 font-mono">
                        {brain.neuron_count.toLocaleString()}
                      </td>
                      <td className="py-3 font-mono">
                        {brain.synapse_count.toLocaleString()}
                      </td>
                      <td className="py-3 font-mono">
                        {brain.fiber_count.toLocaleString()}
                      </td>
                      <td className="py-3">
                        <Badge
                          variant={
                            brain.grade === "A" || brain.grade === "A+"
                              ? "success"
                              : brain.grade === "B" || brain.grade === "B+"
                                ? "secondary"
                                : "warning"
                          }
                        >
                          {brain.grade}
                        </Badge>
                      </td>
                      <td className="py-3">
                        {brain.is_active ? (
                          <Badge variant="default">{t("common.active")}</Badge>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>
                      <td className="py-3">
                        <div className="flex items-center gap-1">
                          {brain.is_active && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="size-8 text-muted-foreground hover:text-primary"
                              onClick={(e) => {
                                e.stopPropagation()
                                setExportTarget(brain.name)
                              }}
                              aria-label={t("store.exportBrain")}
                            >
                              <Export className="size-4" />
                            </Button>
                          )}
                          {!brain.is_active && (
                            <Button
                              variant="ghost"
                              size="icon"
                              className="size-8 text-muted-foreground hover:text-destructive"
                              onClick={(e) => {
                                e.stopPropagation()
                                setDeleteTarget({ id: brain.id, name: brain.name })
                              }}
                              aria-label={t("overview.deleteBrain", { name: brain.name })}
                            >
                              <Trash className="size-4" />
                            </Button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">{t("overview.noBrains")}</p>
          )}
        </CardContent>
      </Card>

      {/* Export dialog */}
      <ExportDialog
        open={!!exportTarget}
        onClose={() => setExportTarget(null)}
        brainName={exportTarget ?? ""}
      />

      {/* Delete confirmation dialog */}
      <ConfirmDialog
        open={!!deleteTarget}
        title={t("overview.deleteBrainTitle")}
        description={t("overview.deleteBrainDesc", { name: deleteTarget?.name })}
        confirmLabel={t("common.delete")}
        variant="destructive"
        onConfirm={handleDeleteBrain}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}
