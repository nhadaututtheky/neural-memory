import { useConfigStatus } from "@/api/hooks/useDashboard"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useTranslation } from "react-i18next"
import type { ConfigStatusItem } from "@/api/types"

const STATUS_VARIANT: Record<
  ConfigStatusItem["status"],
  "success" | "warning" | "secondary" | "default"
> = {
  configured: "success",
  warning: "warning",
  not_configured: "secondary",
  info: "default",
}

export default function ConfigStatusCards() {
  const { data, isLoading } = useConfigStatus()
  const { t } = useTranslation()

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("settings.configStatus")}</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {data?.items.map((item) => (
              <div
                key={item.key}
                className="rounded-lg border border-border/50 px-3 py-2 space-y-1"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium">{item.label}</span>
                  <Badge variant={STATUS_VARIANT[item.status]}>{item.status}</Badge>
                </div>
                {item.value && (
                  <p className="font-mono text-xs text-foreground/80">{item.value}</p>
                )}
                {item.description && (
                  <p className="text-xs text-muted-foreground">{item.description}</p>
                )}
                {item.command && (
                  <p className="font-mono text-xs text-muted-foreground">{item.command}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
