import { useState } from "react"
import { Copy, DownloadSimple, Check } from "@phosphor-icons/react"
import { copyShareImage, downloadShareImage } from "../utils/share-image"
import type { DailyReading } from "../engine/types"
import { useTranslation } from "react-i18next"

interface ShareButtonProps {
  reading: DailyReading
}

export function ShareButton({ reading }: ShareButtonProps) {
  const { t } = useTranslation()
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    const ok = await copyShareImage(reading)
    if (ok) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleDownload = () => {
    downloadShareImage(reading)
  }

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleCopy}
        className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        aria-label={t("oracle.copyCard")}
      >
        {copied ? (
          <>
            <Check className="size-3.5 text-green-400" />
            {t("oracle.copied")}
          </>
        ) : (
          <>
            <Copy className="size-3.5" />
            {t("oracle.copyCard")}
          </>
        )}
      </button>
      <button
        onClick={handleDownload}
        className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        aria-label={t("oracle.downloadCard")}
      >
        <DownloadSimple className="size-3.5" />
        {t("oracle.downloadCard")}
      </button>
    </div>
  )
}
