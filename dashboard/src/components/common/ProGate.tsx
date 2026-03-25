import type { ReactNode } from "react"
import { useIsPro } from "@/api/hooks/useDashboard"
import { Badge } from "@/components/ui/badge"
import { useTranslation } from "react-i18next"
import { openUpgradeModal } from "@/components/common/UpgradeModal"

interface ProGateProps {
  children: ReactNode
  /** Optional label shown on the lock badge */
  label?: string
}

/**
 * Wraps children with a Pro-tier gate.
 * Free users see a dimmed overlay with a "PRO" badge — click opens upgrade modal.
 * Pro/Team users see children normally.
 */
export function ProGate({ children, label }: ProGateProps) {
  const isPro = useIsPro()
  const { t } = useTranslation()

  if (isPro) {
    return <>{children}</>
  }

  return (
    <div className="relative">
      <div className="pointer-events-none opacity-50 select-none">
        {children}
      </div>
      <button
        onClick={openUpgradeModal}
        className="absolute inset-0 flex items-center justify-center cursor-pointer group"
        aria-label={t("upgrade.title", "Get Neural Memory Pro")}
      >
        <Badge
          variant="default"
          className="bg-primary/90 text-primary-foreground px-3 py-1 text-sm shadow-lg group-hover:bg-primary transition-colors"
        >
          {label ?? t("license.pro_badge", "PRO")}
        </Badge>
      </button>
    </div>
  )
}
