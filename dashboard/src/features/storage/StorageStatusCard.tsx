import type { StorageStatusResponse } from "@/api/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { useTranslation } from "react-i18next"
import { Database, HardDrive, CheckCircle, XCircle } from "@phosphor-icons/react"

interface StorageStatusCardProps {
  status: StorageStatusResponse
  brain: string
}

function FileStatus({ exists, label }: { exists: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2 text-sm">
      {exists ? (
        <CheckCircle className="size-4 text-green-500" aria-hidden="true" />
      ) : (
        <XCircle className="size-4 text-muted-foreground" aria-hidden="true" />
      )}
      <span className={exists ? "text-foreground" : "text-muted-foreground"}>
        {label}
      </span>
    </div>
  )
}

const BACKEND_META = {
  infinitydb: { label: "InfinityDB", Icon: HardDrive, variant: "default" as const },
  postgres: { label: "PostgreSQL", Icon: Database, variant: "default" as const },
  sqlite: { label: "SQLite", Icon: Database, variant: "secondary" as const },
}

export function StorageStatusCard({ status, brain }: StorageStatusCardProps) {
  const { t } = useTranslation()
  const isPostgres = status.current_backend === "postgres"
  const meta = BACKEND_META[status.current_backend] ?? BACKEND_META.sqlite
  const BackendIcon = meta.Icon

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Database className="size-5" aria-hidden="true" />
          {t("storage.currentBackend")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-3">
          <Badge variant={meta.variant} className="text-sm px-3 py-1">
            <span className="flex items-center gap-1.5">
              <BackendIcon className="size-3.5" aria-hidden="true" />
              {meta.label}
            </span>
          </Badge>
          <span className="text-sm text-muted-foreground">
            {t("storage.brain")}: <span className="font-medium text-foreground">{brain}</span>
          </span>
        </div>

        <div className="space-y-2">
          {isPostgres ? (
            <FileStatus
              exists={status.postgres_configured ?? true}
              label={
                status.postgres_host
                  ? `${t("storage.postgresBackend")} (${status.postgres_host}/${status.postgres_database ?? ""})`
                  : t("storage.postgresBackend")
              }
            />
          ) : (
            <>
              <FileStatus
                exists={status.sqlite_exists}
                label={
                  status.sqlite_exists && status.sqlite_size_bytes > 0
                    ? `${t("storage.sqliteFile")} (${(status.sqlite_size_bytes / (1024 * 1024)).toFixed(1)} MB)`
                    : t("storage.sqliteFile")
                }
              />
              <FileStatus
                exists={status.infinitydb_exists}
                label={t("storage.infinitydbDir")}
              />
            </>
          )}
        </div>

        {status.is_pro_license && (
          <Badge variant="outline" className="text-xs">
            {t("license.pro")}
          </Badge>
        )}
      </CardContent>
    </Card>
  )
}
