import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { GearSix, X } from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { useBrainSettings } from "../stores/brainSettingsStore"

export function SettingsDrawer() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  const effects = useBrainSettings((s) => s.effects)
  const brainShell = useBrainSettings((s) => s.brainShell)
  const activationPulse = useBrainSettings((s) => s.activationPulse)
  const labels = useBrainSettings((s) => s.labels)
  const setEffects = useBrainSettings((s) => s.setEffects)
  const setBrainShell = useBrainSettings((s) => s.setBrainShell)
  const setActivationPulse = useBrainSettings((s) => s.setActivationPulse)
  const setLabels = useBrainSettings((s) => s.setLabels)

  // Click-outside + Esc dismissal.
  useEffect(() => {
    if (!open) return
    const onPointer = (e: PointerEvent) => {
      const target = e.target as Node | null
      if (!target) return
      if (panelRef.current?.contains(target)) return
      if (buttonRef.current?.contains(target)) return
      setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false)
    }
    window.addEventListener("pointerdown", onPointer)
    window.addEventListener("keydown", onKey)
    return () => {
      window.removeEventListener("pointerdown", onPointer)
      window.removeEventListener("keydown", onKey)
    }
  }, [open])

  // P4 review fix (M3): on open → focus close button so keyboard users
  // land inside the panel. On close → return focus to the gear button so
  // they don't end up on <body>.
  useEffect(() => {
    if (open) {
      closeButtonRef.current?.focus()
    } else {
      buttonRef.current?.focus?.()
    }
  }, [open])

  return (
    <div className="relative">
      <Button
        ref={buttonRef}
        variant="secondary"
        size="sm"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={t("livingBrain.settings.label")}
      >
        <GearSix className="mr-1 h-3.5 w-3.5" />
        {t("livingBrain.settings.button")}
      </Button>
      {open && (
        <div
          ref={panelRef}
          role="dialog"
          aria-label={t("livingBrain.settings.label")}
          className="absolute right-0 top-full z-20 mt-2 w-64 rounded-md border border-border/60 bg-card/95 p-3 shadow-lg backdrop-blur-md"
        >
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              {t("livingBrain.settings.label")}
            </span>
            <button
              ref={closeButtonRef}
              type="button"
              aria-label={t("common.close")}
              onClick={() => setOpen(false)}
              className="rounded p-0.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="flex flex-col gap-1.5">
            <Toggle
              label={t("livingBrain.settings.effects")}
              checked={effects}
              onChange={setEffects}
            />
            <Toggle
              label={t("livingBrain.settings.brainShell")}
              checked={brainShell}
              onChange={setBrainShell}
            />
            <Toggle
              label={t("livingBrain.settings.activationPulse")}
              checked={activationPulse}
              onChange={setActivationPulse}
            />
            <Toggle
              label={t("livingBrain.settings.labels")}
              checked={labels}
              onChange={setLabels}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-2 rounded px-2 py-1.5 text-sm transition-colors hover:bg-accent focus-within:ring-2 focus-within:ring-primary focus-within:ring-offset-1 focus-within:ring-offset-card">
      <span>{label}</span>
      <span
        aria-hidden
        className={`relative inline-block h-4 w-7 shrink-0 rounded-full transition-colors ${
          checked ? "bg-primary" : "bg-muted"
        }`}
      >
        <span
          className={`absolute top-0.5 h-3 w-3 rounded-full bg-background shadow-sm transition-all ${
            checked ? "left-3.5" : "left-0.5"
          }`}
        />
      </span>
      <input
        type="checkbox"
        className="sr-only"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  )
}
