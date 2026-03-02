import { useStats, useHealthCheck, useBrainFiles } from "@/api/hooks/useDashboard"
import { useTelegramStatus, useTelegramTest, useTelegramBackup } from "@/api/hooks/useTelegram"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"

export default function SettingsPage() {
  const { data: stats } = useStats()
  const { data: healthCheck } = useHealthCheck()
  const { data: brainFiles } = useBrainFiles()
  const { data: telegram, isLoading: telegramLoading } = useTelegramStatus()
  const testMutation = useTelegramTest()
  const backupMutation = useTelegramBackup()

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
          toast.success("Test message sent successfully")
        } else {
          toast.error("Some messages failed to send")
        }
      },
      onError: () => {
        toast.error("Failed to send test message")
      },
    })
  }

  const handleTelegramBackup = () => {
    backupMutation.mutate(undefined, {
      onSuccess: (data) => {
        if (data.sent_to > 0) {
          toast.success(
            `Backup sent! ${data.brain} (${data.size_mb}MB) to ${data.sent_to} chat(s)`
          )
        } else {
          toast.error("Backup failed to send")
        }
      },
      onError: () => {
        toast.error("Backup failed")
      },
    })
  }

  return (
    <div className="space-y-6 p-6">
      <h1 className="font-display text-2xl font-bold">Settings</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* General */}
        <Card>
          <CardHeader>
            <CardTitle>General</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Version</span>
              <span className="font-mono">{healthCheck?.version ?? "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Active Brain</span>
              <span className="font-mono">{stats?.active_brain ?? "-"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Total Brains</span>
              <span className="font-mono">{stats?.total_brains ?? "-"}</span>
            </div>
          </CardContent>
        </Card>

        {/* Brain Files */}
        <Card>
          <CardHeader>
            <CardTitle>Brain Files</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {brainFiles ? (
              <>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Brains Directory</span>
                  <span className="font-mono text-xs max-w-[200px] truncate" title={brainFiles.brains_dir}>
                    {brainFiles.brains_dir}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Disk Usage</span>
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
              <p className="text-muted-foreground">Loading...</p>
            )}
          </CardContent>
        </Card>

        {/* Telegram Backup */}
        <Card>
          <CardHeader>
            <CardTitle>Telegram Backup</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            {telegramLoading ? (
              <p className="text-muted-foreground">Loading...</p>
            ) : telegram?.configured ? (
              <>
                <div className="flex items-center gap-2">
                  <Badge variant="success">Connected</Badge>
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
                  <span className="text-muted-foreground">Chat IDs</span>
                  <span className="font-mono text-xs">
                    {telegram.chat_ids.length > 0
                      ? telegram.chat_ids.join(", ")
                      : "(none)"}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">Auto-backup on consolidation</span>
                  <span className="font-mono">
                    {telegram.backup_on_consolidation ? "Yes" : "No"}
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
                    {testMutation.isPending ? "Sending..." : "Send Test"}
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleTelegramBackup}
                    disabled={backupMutation.isPending}
                    className="cursor-pointer"
                  >
                    {backupMutation.isPending ? "Backing up..." : "Backup Now"}
                  </Button>
                </div>
              </>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="warning">Not Configured</Badge>
                </div>
                {telegram?.error && (
                  <p className="text-xs text-destructive">{telegram.error}</p>
                )}
                <p className="text-xs text-muted-foreground">
                  Set <code className="font-mono rounded bg-muted px-1">NMEM_TELEGRAM_BOT_TOKEN</code> env
                  var and add <code className="font-mono rounded bg-muted px-1">[telegram] chat_ids</code> to
                  config.toml.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
