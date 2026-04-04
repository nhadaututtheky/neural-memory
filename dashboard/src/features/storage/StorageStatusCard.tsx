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

export function StorageStatusCard({ status, brain }: StorageStatusCardProps) {
  const { t } = useTranslation()
  const isInfinity = status.current_backend === "infinitydb"

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
          <Badge
            variant={isInfinity ? "default" : "secondary"}
            className="text-sm px-3 py-1"
          >
            {isInfinity ? (
              <span className="flex items-center gap-1.5">
                <HardDrive className="size-3.5" aria-hidden="true" />
                InfinityDB
              </span>
            ) : (
              <span className="flex items-center gap-1.5">
                <Database className="size-3.5" aria-hidden="true" />
                SQLite
              </span>
            )}
          </Badge>
          <span className="text-sm text-muted-foreground">
            {t("storage.brain")}: <span className="font-medium text-foreground">{brain}</span>
          </span>
        </div>

        <div className="space-y-2">
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
