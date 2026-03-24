import { useWatcherStatus } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useTranslation } from "react-i18next"

export default function WatcherCard() {
  const { data, isLoading } = useWatcherStatus()
  const { t } = useTranslation()

  const statusBadge = () => {
    if (!data) return null
    if (data.running) {
      return <Badge variant="success">{t("settings.watcherRunning")}</Badge>
    }
    if (data.enabled) {
      return <Badge variant="warning">{t("settings.watcherStopped")}</Badge>
    }
    return <Badge variant="secondary">{t("settings.watcherDisabled")}</Badge>
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{t("settings.watcher")}</CardTitle>
          {isLoading ? (
            <Skeleton className="h-5 w-16" />
          ) : (
            statusBadge()
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        ) : data ? (
          <>
            {/* Watched paths */}
            {data.paths.length > 0 && (
              <div>
                <p className="mb-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  {t("settings.watcherPaths")}
                </p>
                <div className="space-y-1">
                  {data.paths.map((p) => (
                    <p key={p} className="font-mono text-xs truncate text-foreground" title={p}>
                      {p}
                    </p>
                  ))}
                </div>
              </div>
            )}

            {/* Recent activity */}
            <div>
              <p className="mb-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                {t("settings.watcherRecent")}
              </p>
              {data.recent.length > 0 ? (
                <div className="space-y-1">
                  {data.recent.slice(0, 5).map((r, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between rounded border border-border/50 px-2 py-1.5"
                    >
                      <span className="font-mono text-xs truncate max-w-[200px] text-foreground" title={r.path}>
                        {r.path.split(/[/\\]/).pop()}
                      </span>
                      <span className="shrink-0 ml-2 font-mono text-xs text-muted-foreground">
                        +{r.neurons_created}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">{t("settings.watcherNoActivity")}</p>
              )}
            </div>
          </>
        ) : null}
      </CardContent>
    </Card>
  )
}
