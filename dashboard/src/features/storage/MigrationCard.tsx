import { useEffect, useState } from "react"
import type { StorageStatusResponse } from "@/api/types"
import {
  useStartMigration,
  useMigrationJob,
  useSetBackend,
} from "@/api/hooks/useStorage"
import { ProGate } from "@/components/common/ProGate"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { MigrationProgress } from "./MigrationProgress"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"
import {
  ArrowsLeftRight,
  SpinnerGap,
  CheckCircle,
  Warning,
  Lightning,
} from "@phosphor-icons/react"

interface MigrationCardProps {
  status: StorageStatusResponse
}

export function MigrationCard({ status }: MigrationCardProps) {
  const { t } = useTranslation()
  const [localJobId, setLocalJobId] = useState<string | null>(null)

  // Sync from parent status — picks up running/done jobs on page load or refetch
  const activeJobId =
    localJobId ?? (status.migration_job ? status.migration_job.job_id : null)

  useEffect(() => {
    if (status.migration_job?.state === "running" && !localJobId) {
      setLocalJobId(status.migration_job.job_id)
    }
  }, [status.migration_job, localJobId])

  const startMigration = useStartMigration()
  const setBackend = useSetBackend()
  const { data: job } = useMigrationJob(activeJobId)

  const isRunning = job?.state === "running"
  const isDone = job?.state === "done"
  const isError = job?.state === "error"
  const isSqlite = status.current_backend === "sqlite"

  const handleMigrate = (direction: "to_infinitydb" | "to_sqlite") => {
    startMigration.mutate(direction, {
      onSuccess: (data) => {
        setLocalJobId(data.job_id)
        toast.success(t("storage.migrationStarted"))
        if (data.disk_warning) {
          toast.warning(data.disk_warning)
        }
      },
      onError: (err) => {
        toast.error(
          err instanceof Error ? err.message : t("storage.migrationFailed"),
        )
      },
    })
  }

  const handleSwitchBackend = (backend: "sqlite" | "infinitydb") => {
    setBackend.mutate(backend, {
      onSuccess: (data) => {
        if (data.status === "switched") {
          toast.success(t("storage.backendSwitched", { backend: data.backend }))
          setLocalJobId(null)
        }
      },
      onError: () => toast.error(t("storage.switchFailed")),
    })
  }

  const migrationContent = (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <ArrowsLeftRight className="size-5" aria-hidden="true" />
          {t("storage.migration")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Why upgrade bullets (when on SQLite) */}
        {isSqlite && !activeJobId && (
          <div className="space-y-2 rounded-lg border border-dashed p-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Lightning className="size-4 text-yellow-500" aria-hidden="true" />
              {t("storage.whyUpgrade")}
            </div>
            <ul className="space-y-1 text-xs text-muted-foreground list-disc pl-5">
              <li>{t("storage.whyBullet1")}</li>
              <li>{t("storage.whyBullet2")}</li>
              <li>{t("storage.whyBullet3")}</li>
            </ul>
          </div>
        )}

        {/* Migration progress */}
        {job && isRunning && <MigrationProgress job={job} />}

        {/* Done state — confirm switch */}
        {job && isDone && (
          <div className="space-y-3 rounded-lg border border-green-500/30 bg-green-500/5 p-3">
            <div className="flex items-center gap-2 text-sm font-medium text-green-600 dark:text-green-400">
              <CheckCircle className="size-4" aria-hidden="true" />
              {t("storage.migrationDone")}
            </div>
            <MigrationProgress job={job} />
            <Button
              size="sm"
              onClick={() =>
                handleSwitchBackend(
                  job.direction === "to_infinitydb" ? "infinitydb" : "sqlite",
                )
              }
              disabled={setBackend.isPending}
              className="cursor-pointer"
            >
              {setBackend.isPending && (
                <SpinnerGap className="mr-2 size-4 animate-spin" aria-hidden="true" />
              )}
              {t("storage.switchBackend", {
                backend:
                  job.direction === "to_infinitydb" ? "InfinityDB" : "SQLite",
              })}
            </Button>
          </div>
        )}

        {/* Error state */}
        {job && isError && (
          <div className="space-y-3 rounded-lg border border-destructive/30 bg-destructive/5 p-3">
            <div className="flex items-center gap-2 text-sm font-medium text-destructive">
              <Warning className="size-4" aria-hidden="true" />
              {t("storage.migrationError")}
            </div>
            {job.error && (
              <p className="text-xs text-muted-foreground">{job.error}</p>
            )}
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                setLocalJobId(null)
                startMigration.reset()
              }}
              className="cursor-pointer"
            >
              {t("storage.retry")}
            </Button>
          </div>
        )}

        {/* Action buttons */}
        {!activeJobId && (
          <div className="flex flex-wrap gap-2">
            {isSqlite && (
              <Button
                onClick={() => handleMigrate("to_infinitydb")}
                disabled={
                  startMigration.isPending ||
                  !status.pro_installed ||
                  !status.is_pro_license
                }
                className="cursor-pointer"
              >
                {startMigration.isPending && (
                  <SpinnerGap
                    className="mr-2 size-4 animate-spin"
                    aria-hidden="true"
                  />
                )}
                {t("storage.enableInfinitydb")}
              </Button>
            )}
            {!isSqlite && (
              <Button
                variant="outline"
                onClick={() => handleMigrate("to_sqlite")}
                disabled={startMigration.isPending}
                className="cursor-pointer"
              >
                {startMigration.isPending && (
                  <SpinnerGap
                    className="mr-2 size-4 animate-spin"
                    aria-hidden="true"
                  />
                )}
                {t("storage.rollbackSqlite")}
              </Button>
            )}
          </div>
        )}

        {/* Running indicator */}
        {isRunning && (
          <Badge variant="secondary" className="animate-pulse">
            <SpinnerGap className="mr-1.5 size-3 animate-spin" aria-hidden="true" />
            {t("storage.migrating")}
          </Badge>
        )}
      </CardContent>
    </Card>
  )

  // Rollback is always accessible (no Pro gate)
  if (!isSqlite) {
    return migrationContent
  }

  // Forward migration requires Pro
  return <ProGate>{migrationContent}</ProGate>
}
