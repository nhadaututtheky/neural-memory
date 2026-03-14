import { toast } from "sonner"
import { CheckCircle2, AlertTriangle, XCircle, Info, Copy } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { useConfigStatus } from "@/api/hooks/useDashboard"
import type { ConfigStatusItem } from "@/api/types"

type StatusConfig = {
  icon: React.ElementType
  badgeVariant: "success" | "warning" | "destructive" | "secondary"
  label: string
}

const STATUS_MAP: Record<ConfigStatusItem["status"], StatusConfig> = {
  configured: {
    icon: CheckCircle2,
    badgeVariant: "success",
    label: "Configured",
  },
  warning: {
    icon: AlertTriangle,
    badgeVariant: "warning",
    label: "Warning",
  },
  not_configured: {
    icon: XCircle,
    badgeVariant: "destructive",
    label: "Not configured",
  },
  info: {
    icon: Info,
    badgeVariant: "secondary",
    label: "Info",
  },
}

function ConfigItemRow({ item }: { item: ConfigStatusItem }) {
  const { icon: StatusIcon, badgeVariant, label } = STATUS_MAP[item.status]

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(item.command)
      toast.success("Copied to clipboard")
    } catch {
      toast.error("Failed to copy")
    }
  }

  return (
    <div className="space-y-1.5 py-3 first:pt-0 last:pb-0">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <StatusIcon
            className={`size-4 shrink-0 ${
              item.status === "configured"
                ? "text-health-good"
                : item.status === "warning"
                  ? "text-health-warn"
                  : item.status === "not_configured"
                    ? "text-destructive"
                    : "text-muted-foreground"
            }`}
            aria-hidden="true"
          />
          <span className="text-sm font-medium">{item.label}</span>
        </div>
        <Badge variant={badgeVariant} className="shrink-0 text-xs">
          {label}
        </Badge>
      </div>

      {item.description && (
        <p className="pl-6 text-xs text-muted-foreground">{item.description}</p>
      )}

      {item.command && (
        <div className="flex items-center gap-2 pl-6">
          <code className="min-w-0 flex-1 truncate rounded-md border border-border bg-muted px-2.5 py-1.5 font-mono text-xs">
            {item.command}
          </code>
          <Button
            variant="ghost"
            size="icon"
            className="size-7 shrink-0 text-muted-foreground hover:text-foreground"
            onClick={handleCopy}
            aria-label={`Copy command: ${item.command}`}
          >
            <Copy className="size-3.5" aria-hidden="true" />
          </Button>
        </div>
      )}
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Skeleton className="size-4 rounded-full" />
              <Skeleton className="h-4 w-32" />
            </div>
            <Skeleton className="h-5 w-20 rounded-md" />
          </div>
          <Skeleton className="ml-6 h-3 w-48" />
          <Skeleton className="ml-6 h-7 w-full" />
        </div>
      ))}
    </div>
  )
}

export default function QuickActionsCard() {
  const { data, isLoading } = useConfigStatus()

  const items = data?.items ?? []

  if (!isLoading && items.length === 0) {
    return null
  }

  const allConfigured = items.length > 0 && items.every((item) => item.status === "configured")

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Quick Actions</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <LoadingSkeleton />
        ) : allConfigured ? (
          <div className="flex items-center gap-2 py-2 text-sm text-health-good">
            <CheckCircle2 className="size-4" aria-hidden="true" />
            <span>All features configured</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 divide-y divide-border sm:grid-cols-2 sm:divide-x sm:divide-y-0">
            {items.map((item, idx) => {
              const isRightColumn = idx % 2 === 1
              return (
                <div
                  key={item.key}
                  className={`${isRightColumn ? "sm:pl-4" : "sm:pr-4"}`}
                >
                  <ConfigItemRow item={item} />
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
