import { useState } from "react"
import { useTranslation } from "react-i18next"
import { DownloadSimple, Check } from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"

interface ShareBrainProps {
  /**
   * Ref-like accessor to the Canvas' WebGL DOM element. We query it lazily
   * (not on mount) so Canvas lifecycle is untouched.
   */
  getCanvasEl: () => HTMLCanvasElement | null
  /**
   * Forces the r3f renderer to paint a fresh frame into the drawing buffer
   * before we capture it. Required because `frameloop="demand"` means the
   * buffer may be stale between pulses — `preserveDrawingBuffer: true`
   * preserves whatever was LAST drawn, not the current scene state.
   */
  onRequestRender?: (() => void) | null
  watermark?: string
}

const MAX_EDGE = 1920 // cap export resolution so 500-node scenes don't OOM low-end GPUs

export function ShareBrain({
  getCanvasEl,
  onRequestRender,
  watermark = "Neural Memory",
}: ShareBrainProps) {
  const { t } = useTranslation()
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState(false)

  const handleExport = async () => {
    const src = getCanvasEl()
    if (!src) return
    setBusy(true)
    setError(false)
    try {
      // P4 review fix (H3): force a fresh render so the drawing buffer
      // reflects the current scene. Two rAFs to make sure the draw has
      // committed before toBlob reads pixels.
      onRequestRender?.()
      await new Promise((r) => requestAnimationFrame(r))
      await new Promise((r) => requestAnimationFrame(r))

      const blob = await exportCanvasWithWatermark(src, watermark)
      if (!blob) {
        setError(true)
        setTimeout(() => setError(false), 2000)
        return
      }
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `neural-brain-${new Date().toISOString().slice(0, 10)}.png`
      document.body.appendChild(a)
      a.click()
      a.remove()
      // P4 review fix (H4): defer revoke so Firefox/Safari don't cancel
      // the download mid-flight. 1s is plenty for the browser to capture
      // the blob into its download manager.
      setTimeout(() => URL.revokeObjectURL(url), 1000)
      setDone(true)
      setTimeout(() => setDone(false), 1500)
    } catch {
      setError(true)
      setTimeout(() => setError(false), 2000)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Button
      variant="secondary"
      size="sm"
      onClick={handleExport}
      disabled={busy}
      aria-label={t("livingBrain.share.label")}
    >
      {done ? (
        <Check className="mr-1 h-3.5 w-3.5" />
      ) : (
        <DownloadSimple className="mr-1 h-3.5 w-3.5" />
      )}
      <span>
        {error
          ? t("livingBrain.share.error")
          : done
            ? t("livingBrain.share.done")
            : t("livingBrain.share.button")}
      </span>
    </Button>
  )
}

async function exportCanvasWithWatermark(
  src: HTMLCanvasElement,
  watermark: string,
): Promise<Blob | null> {
  const scale = Math.min(1, MAX_EDGE / Math.max(src.width, src.height))
  const w = Math.round(src.width * scale)
  const h = Math.round(src.height * scale)
  const out = document.createElement("canvas")
  out.width = w
  out.height = h
  const ctx = out.getContext("2d")
  if (!ctx) return null
  ctx.drawImage(src, 0, 0, w, h)

  const padding = Math.round(Math.min(w, h) * 0.025)
  const fontSize = Math.max(14, Math.round(Math.min(w, h) * 0.018))
  ctx.font = `600 ${fontSize}px "Inter", ui-sans-serif, system-ui, -apple-system, sans-serif`
  ctx.textBaseline = "bottom"
  ctx.textAlign = "right"
  ctx.fillStyle = "rgba(226, 208, 255, 0.85)"
  ctx.shadowColor = "rgba(0,0,0,0.6)"
  ctx.shadowBlur = 4
  ctx.fillText(watermark, w - padding, h - padding)

  return new Promise<Blob | null>((resolve) => {
    out.toBlob((b) => resolve(b), "image/png")
  })
}
