import { useStats, useHealthCheck, useBrainFiles, useLicense } from "@/api/hooks/useDashboard"
import { useTelegramStatus, useTelegramTest, useTelegramBackup } from "@/api/hooks/useTelegram"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"
import { ArrowSquareOut, Bug, ChatText, GithubLogo } from "@phosphor-icons/react"
import { useTranslation } from "react-i18next"
import ConfigStatusCards from "./ConfigStatusCards"
import EmbeddingConfig from "./EmbeddingConfig"
import WatcherCard from "./WatcherCard"
import { openUpgradeModal } from "@/components/common/UpgradeModal"

const FEEDBACK_ICONS = [Bug, ChatText, GithubLogo] as const
const FEEDBACK_COLORS = ["#ef4444", "#6366f1", "#a8a29e"] as const
const FEEDBACK_KEYS = ["reportBug", "featureRequest", "discussions"] as const
const FEEDBACK_URLS = [
  "https://github.com/nhadaututtheky/neural-memory/issues/new?template=bug_report.md",
  "https://github.com/nhadaututtheky/neural-memory/issues/new?template=feature_request.md",
  "https://github.com/nhadaututtheky/neural-memory/discussions",
] as const

export default function SettingsPage() {
  const { data: stats } = useStats()
  const { data: healthCheck } = useHealthCheck()
  const { data: brainFiles } = useBrainFiles()
  const { data: license } = useLicense()
  const { data: telegram, isLoading: telegramLoading } = useTelegramStatus()
  const testMutation = useTelegramTest()
  const backupMutation = useTelegramBackup()
  const { t } = useTranslation()

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 B"
    const k = 1024
    const sizes = ["B", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
  }

  const handleTelegramTest = () => {
    testMutation.mutate(undefined, {
      onSuccess: (data) => {
        if (data.status === "success") {
          toast.success(t("settings.testSuccess"))
        } else {
          toast.error(t("settings.testPartial"))
        }
      },
      onError: () => {
        toast.error(t("settings.testFailed"))
      },
    })
  }

  const handleTelegramBackup = () => {
    backupMutation.mutate(undefined, {
      onSuccess: (data) => {
        if (data.sent_to > 0) {
          toast.success(
            t("settings.backupSuccess", {
              brain: data.brain,
              size: data.size_mb,
              count: data.sent_to,
            })
          )
        } else {
          toast.error(t("settings.backupSendFailed"))
        }
      },
      onError: () => {
        toast.error(t("settings.backupFailed"))
      },
    })
  }

  return (
    <div className="space-y-6 p-6">
      <h1 className="font-display text-2xl font-bold">{t("settings.title")}</h1>

      {/* Config Status */}
      <ConfigStatusCards />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* General */}
        <Card>
          <CardHeader>
            <CardTitle>{t("settings.general")}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">{t("settings.version")}</span>
              <span className="font-mono">{healthCheck?.version ?? "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">{t("settings.activeBrain")}</span>
              <span className="font-mono">{stats?.active_brain ?? "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">{t("settings.totalBrains")}</span>
              <span className="font-mono">{stats?.total_brains ?? "-"}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">{t("license.tier", "License")}</span>
              <div className="flex items-center gap-2">
                <Badge variant={license?.is_pro ? "success" : "secondary"}>
                  {(license?.tier ?? "free").toUpperCase()}
                </Badge>
                {!license?.is_pro && (
                  <button
                    onClick={openUpgradeModal}
                    className="text-xs font-medium text-primary hover:text-primary/80 transition-colors cursor-pointer"
                  >
                    {t("upgrade.title", "Upgrade")}
                  </button>
                )}
              </div>
            </div>
            {license?.is_pro && license?.pro_deps_missing && (
              <div className="mt-3 rounded-md border border-yellow-500/30 bg-yellow-500/10 p-3 text-sm">
                <p className="font-medium text-yellow-400">Pro dependencies missing</p>
                <p className="text-muted-foreground mt-1">
                  Missing: {license.pro_deps_missing.join(", ")}
                </p>
                <code className="mt-2 block rounded bg-muted px-2 py-1 text-xs select-all">
                  pip install neural-memory[pro]
                </code>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Brain Files */}
        <Card>
          <CardHeader>
            <CardTitle>{t("settings.brainFiles")}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {brainFiles ? (
              <>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t("settings.brainsDirectory")}</span>
                  <span className="font-mono text-xs max-w-[200px] truncate" title={brainFiles.brains_dir}>
                    {brainFiles.brains_dir}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t("settings.totalDiskUsage")}</span>
                  <span className="font-mono">{formatBytes(brainFiles.total_size_bytes)}</span>
                </div>
                {brainFiles.brains.length > 0 && (
                  <div className="mt-3 space-y-2">
                    {brainFiles.brains.map((b) => (
                      <div key={b.name} className="flex items-center justify-between rounded-lg border border-border/50 px-3 py-2">
                        <span className="font-mono font-medium">{b.name}</span>
                        <span className="font-mono text-xs text-muted-foreground">
                          {formatBytes(b.size_bytes)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p className="text-muted-foreground">{t("common.loading")}</p>
            )}
          </CardContent>
        </Card>

        {/* Telegram Backup */}
        <Card>
          <CardHeader>
            <CardTitle>{t("settings.telegramBackup")}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            {telegramLoading ? (
              <p className="text-muted-foreground">{t("common.loading")}</p>
            ) : telegram?.configured ? (
              <>
                <div className="flex items-center gap-2">
                  <Badge variant="success">{t("settings.connected")}</Badge>
                  {telegram.bot_name && (
                    <span className="text-muted-foreground">
                      {telegram.bot_name}
                      {telegram.bot_username && (
                        <span className="font-mono text-xs"> @{telegram.bot_username}</span>
                      )}
                    </span>
                  )}
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t("settings.chatIds")}</span>
                  <span className="font-mono text-xs">
                    {telegram.chat_ids.length > 0
                      ? telegram.chat_ids.join(", ")
                      : t("common.none")}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">{t("settings.autoBackup")}</span>
                  <span className="font-mono">
                    {telegram.backup_on_consolidation ? t("common.yes") : t("common.no")}
                  </span>
                </div>

                <div className="flex gap-2 pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleTelegramTest}
                    disabled={testMutation.isPending}
                    className="cursor-pointer"
                  >
                    {testMutation.isPending ? t("settings.sending") : t("settings.sendTest")}
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleTelegramBackup}
                    disabled={backupMutation.isPending}
                    className="cursor-pointer"
                  >
                    {backupMutation.isPending ? t("settings.backingUp") : t("settings.backupNow")}
                  </Button>
                </div>
              </>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="warning">{t("settings.notConfigured")}</Badge>
                </div>
                {telegram?.error && (
                  <p className="text-xs text-destructive">{telegram.error}</p>
                )}
                <p
                  className="text-xs text-muted-foreground"
                  dangerouslySetInnerHTML={{ __html: t("settings.telegramSetup") }}
                />
              </div>
            )}
          </CardContent>
        </Card>

        {/* File Watcher */}
        <WatcherCard />

        {/* Embedding Provider (Pro) */}
        <EmbeddingConfig />

        {/* Feedback & Bug Report */}
        <Card>
          <CardHeader>
            <CardTitle>{t("settings.feedbackTitle")}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {FEEDBACK_KEYS.map((key, idx) => {
              const Icon = FEEDBACK_ICONS[idx]
              const color = FEEDBACK_COLORS[idx]
              const url = FEEDBACK_URLS[idx]

              return (
                <a
                  key={key}
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-3 rounded-lg border border-border/50 p-3 transition-colors hover:bg-accent cursor-pointer"
                >
                  <div
                    className="flex size-8 shrink-0 items-center justify-center rounded-lg"
                    style={{ backgroundColor: `${color}15` }}
                  >
                    <Icon className="size-4" style={{ color }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">{t(`settings.${key}`)}</p>
                    <p className="text-xs text-muted-foreground">{t(`settings.${key}Desc`)}</p>
                  </div>
                  <ArrowSquareOut className="size-3.5 shrink-0 text-muted-foreground mt-0.5" />
                </a>
              )
            })}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
