import { useStorageStatus, useTierStats } from "@/api/hooks/useStorage"
import { useStats } from "@/api/hooks/useDashboard"
import { StorageStatusCard } from "./StorageStatusCard"
import { MigrationCard } from "./MigrationCard"
import { TierDistributionCard } from "./TierDistributionCard"
import { Skeleton } from "@/components/ui/skeleton"
import { useTranslation } from "react-i18next"

export default function StoragePage() {
  const { data: status, isLoading } = useStorageStatus()
  const { data: stats } = useStats()
  const { data: tierStats } = useTierStats()
  const { t } = useTranslation()

  if (isLoading || !status) {
    return (
      <div className="space-y-6 p-6">
        <h1 className="text-2xl font-bold">{t("storage.title")}</h1>
        <div className="grid gap-6 md:grid-cols-2">
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </div>
    )
  }

  const brain = stats?.active_brain ?? "default"

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold">{t("storage.title")}</h1>
      <div className="grid gap-6 md:grid-cols-2">
        <StorageStatusCard status={status} brain={brain} />
        <MigrationCard status={status} />
      </div>
      {tierStats && tierStats.total > 0 && (
        <div className="grid gap-6 md:grid-cols-2">
          <TierDistributionCard data={tierStats} />
        </div>
      )}
    </div>
  )
}
