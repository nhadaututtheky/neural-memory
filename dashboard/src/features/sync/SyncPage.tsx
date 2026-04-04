import { useState } from "react"
import { useIsPro } from "@/api/hooks/useDashboard"
import { useSyncStatus, useUpdateSyncConfig } from "@/api/hooks/useSync"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"
import { useTranslation } from "react-i18next"
import {
  Cloud,
  CloudSlash,
  Monitor,
  ArrowsClockwise,
  Shield,
  WifiHigh,
} from "@phosphor-icons/react"

// No shared hub — users deploy their own Cloudflare Worker
const DEFAULT_HUB_URL = ""
const HUB_URL_PLACEHOLDER = "https://your-worker.your-subdomain.workers.dev"

const STRATEGY_LABELS: Record<string, string> = {
  prefer_recent: "Prefer Recent",
  prefer_local: "Prefer Local",
  prefer_remote: "Prefer Remote",
  prefer_stronger: "Prefer Stronger",
}

export default function SyncPage() {
  const { data: status, isLoading, refetch } = useSyncStatus()
  const updateConfig = useUpdateSyncConfig()
  const isPro = useIsPro()
  const { t } = useTranslation()

  const [hubUrl, setHubUrl] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [showSetup, setShowSetup] = useState(false)

  const handleConnect = () => {
    const url = hubUrl.trim() || DEFAULT_HUB_URL
    const key = apiKey.trim()

    if (!key) {
      toast.error(t("sync.keyRequired"))
      return
    }

    if (!key.startsWith("nmk_")) {
      toast.error(t("sync.keyInvalid"))
      return
    }

    updateConfig.mutate(
      { hub_url: url, api_key: key, enabled: true },
      {
        onSuccess: () => {
          toast.success(t("sync.connected"))
          setShowSetup(false)
          setApiKey("")
        },
        onError: () => toast.error(t("sync.connectFailed")),
      },
    )
  }

  const handleDisconnect = () => {
    updateConfig.mutate(
      { enabled: false, api_key: "", hub_url: "" },
      {
        onSuccess: () => toast.success(t("sync.disconnected")),
        onError: () => toast.error(t("sync.disconnectFailed")),
      },
    )
  }

  const handleStrategyChange = (strategy: string) => {
    updateConfig.mutate(
      { conflict_strategy: strategy },
      {
        onSuccess: () => toast.success(t("sync.strategyUpdated")),
        onError: () => toast.error(t("sync.updateFailed")),
      },
    )
  }

  const formatDate = (iso: string | null) => {
    if (!iso) return t("sync.never")
    return new Date(iso).toLocaleString()
  }

  if (isLoading) {
    return (
      <div className="space-y-6 p-6">
        <h1 className="font-display text-2xl font-bold">{t("sync.title")}</h1>
        <p className="text-muted-foreground">{t("common.loading")}</p>
      </div>
    )
  }

  const isConnected = status?.enabled && status.api_key !== "(not set)"

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-bold">{t("sync.title")}</h1>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          className="cursor-pointer"
        >
          <ArrowsClockwise className="mr-2 size-4" />
          {t("sync.refresh")}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Connection Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {isConnected ? (
                <Cloud className="size-5 text-emerald-500" />
              ) : (
                <CloudSlash className="size-5 text-muted-foreground" />
              )}
              {t("sync.connectionStatus")}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">{t("sync.status")}</span>
              <Badge variant={isConnected ? "success" : "warning"}>
                {isConnected ? t("sync.cloudConnected") : t("sync.notConnected")}
              </Badge>
            </div>

            {isConnected && (
              <>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t("sync.hubUrl")}</span>
                  <span className="max-w-[200px] truncate font-mono text-xs" title={status.hub_url}>
                    {status.hub_url}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t("sync.apiKey")}</span>
                  <span className="font-mono text-xs">{status.api_key}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t("sync.deviceId")}</span>
                  <span className="font-mono text-xs">{status.device_id?.slice(0, 12)}...</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t("sync.syncMode")}</span>
                  <Badge variant={isPro ? "success" : "secondary"}>
                    {isPro ? t("sync.merkleDelta") : t("sync.fullSync")}
                  </Badge>
                </div>
              </>
            )}

            <div className="flex gap-2 pt-2">
              {isConnected ? (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleDisconnect}
                  disabled={updateConfig.isPending}
                  className="cursor-pointer"
                >
                  {t("sync.disconnect")}
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={() => setShowSetup(true)}
                  className="cursor-pointer"
                >
                  {t("sync.setupCloud")}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Setup Form */}
        {(showSetup && !isConnected) && (
          <Card>
            <CardHeader>
              <CardTitle>{t("sync.setupTitle")}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <p className="text-muted-foreground">{t("sync.setupDesc")}</p>

              <div className="space-y-2">
                <label htmlFor="hub-url" className="text-xs font-medium text-muted-foreground">
                  {t("sync.hubUrl")}
                </label>
                <input
                  id="hub-url"
                  type="url"
                  value={hubUrl}
                  onChange={(e) => setHubUrl(e.target.value)}
                  placeholder={HUB_URL_PLACEHOLDER}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              <div className="space-y-2">
                <label htmlFor="api-key" className="text-xs font-medium text-muted-foreground">
                  {t("sync.apiKey")}
                </label>
                <input
                  id="api-key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="nmk_..."
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-ring"
                />
                <p className="text-xs text-muted-foreground">{t("sync.keyHint")}</p>
              </div>

              <div className="flex gap-2">
                <Button
                  size="sm"
                  onClick={handleConnect}
                  disabled={updateConfig.isPending}
                  className="cursor-pointer"
                >
                  {updateConfig.isPending ? t("sync.connecting") : t("sync.connect")}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowSetup(false)}
                  className="cursor-pointer"
                >
                  {t("common.cancel")}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Devices */}
        {isConnected && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Monitor className="size-5" />
                {t("sync.devices")} ({status.device_count})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {status.devices.length > 0 ? (
                status.devices.map((device) => (
                  <div
                    key={device.device_id}
                    className="flex items-center justify-between rounded-lg border border-border/50 px-3 py-2"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="font-mono text-xs font-medium">
                        {device.device_name || device.device_id.slice(0, 12)}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {t("sync.lastSync")}: {formatDate(device.last_sync_at)}
                      </p>
                    </div>
                    <Badge variant="outline" className="shrink-0">
                      seq {device.last_sync_sequence}
                    </Badge>
                  </div>
                ))
              ) : (
                <p className="text-muted-foreground">{t("sync.noDevices")}</p>
              )}
            </CardContent>
          </Card>
        )}

        {/* Change Log */}
        {isConnected && status.change_log && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <WifiHigh className="size-5" />
                {t("sync.changeLog")}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">{t("sync.totalChanges")}</span>
                <span className="font-mono">{status.change_log.total_changes}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">{t("sync.synced")}</span>
                <span className="font-mono text-emerald-500">
                  {status.change_log.synced_changes}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">{t("sync.pending")}</span>
                <span className="font-mono text-amber-500">
                  {status.change_log.unsynced_changes}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">{t("sync.latestSequence")}</span>
                <span className="font-mono">{status.change_log.latest_sequence}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Conflict Strategy */}
        {isConnected && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="size-5" />
                {t("sync.conflictStrategy")}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <p className="text-muted-foreground">{t("sync.conflictDesc")}</p>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(STRATEGY_LABELS).map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => handleStrategyChange(key)}
                    disabled={updateConfig.isPending}
                    className={`cursor-pointer rounded-lg border px-3 py-2 text-xs font-medium transition-colors ${
                      status.conflict_strategy === key
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border hover:bg-accent"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
