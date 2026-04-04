import { useCallback, useEffect, useState } from "react"
import { createPortal } from "react-dom"
import { useTranslation } from "react-i18next"
import { useIsPro } from "@/api/hooks/useDashboard"
import { api } from "@/api/client"
import { X, CaretRight, Key, Check, SpinnerGap } from "@phosphor-icons/react"

/* ------------------------------------------------------------------ */
/*  Pricing config                                                     */
/* ------------------------------------------------------------------ */

const PAY_HUB_URL = "https://pay.theio.vn"

const PRICING = {
  monthly: {
    product: "NM-PRO-MONTHLY",
    price: "$9",
    period: "/month",
    priceVnd: "219,000 VND/tháng",
    method: "Card / PayPal (Polar)",
    flag: "\uD83C\uDF10",
  },
  yearly: {
    product: "NM-PRO-YEARLY",
    price: "$89",
    period: "/year",
    badge: "Save $19",
    priceVnd: "2,190,000 VND/năm",
    method: "Card / PayPal (Polar)",
    flag: "\uD83C\uDF10",
  },
  vn: {
    product: "NM-PRO-MONTHLY",
    price: "219,000",
    period: " VND/tháng",
    method: "Bank Transfer (VietQR)",
    flag: "\uD83C\uDDFB\uD83C\uDDF3",
  },
} as const

/* ------------------------------------------------------------------ */
/*  Context: shared open state                                         */
/* ------------------------------------------------------------------ */

let _openModal: (() => void) | null = null

/** Imperatively open the upgrade modal from anywhere */
export function openUpgradeModal() {
  _openModal?.()
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function UpgradeModal() {
  const [open, setOpen] = useState(false)
  const [view, setView] = useState<"choose" | "email" | "activate">("choose")
  const [selectedPlan, setSelectedPlan] = useState<"monthly" | "yearly" | "vn">("monthly")
  const [email, setEmail] = useState("")
  const [licenseKey, setLicenseKey] = useState("")
  const [activating, setActivating] = useState(false)
  const [activated, setActivated] = useState(false)
  const [error, setError] = useState("")
  const isPro = useIsPro()
  const { t } = useTranslation()

  // Register global opener
  useEffect(() => {
    _openModal = () => setOpen(true)
    return () => { _openModal = null }
  }, [])

  // Reset on close
  useEffect(() => {
    if (!open) {
      setView("choose")
      setLicenseKey("")
      setEmail("")
      setError("")
      setActivated(false)
    }
  }, [open])

  // ESC to close
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false)
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [open])

  const handleSelectPlan = useCallback((plan: "monthly" | "yearly" | "vn") => {
    setSelectedPlan(plan)
    setError("")
    setView("email")
  }, [])

  const handleCheckout = useCallback(async () => {
    const trimmedEmail = email.trim()
    if (!trimmedEmail) return

    // Basic email validation
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmedEmail)) {
      setError(t("upgrade.invalidEmail", "Please enter a valid email address"))
      return
    }

    setActivating(true)
    setError("")

    try {
      const pricing = PRICING[selectedPlan]
      const endpoint = selectedPlan === "vn" ? "/order/sepay" : "/checkout/polar"
      const res = await fetch(`${PAY_HUB_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product: pricing.product,
          email: trimmedEmail,
        }),
      })
      const data = await res.json() as { url?: string; qr_url?: string }

      if (data.url) {
        window.open(data.url, "_blank")
      } else if (data.qr_url) {
        window.open(data.qr_url, "_blank")
      }
      setOpen(false)
    } catch {
      // Fallback: open Polar page directly
      if (selectedPlan !== "vn") {
        window.open("https://polar.sh/nhadaututtheky", "_blank")
      }
      setOpen(false)
    } finally {
      setActivating(false)
    }
  }, [email, selectedPlan, t])

  const handleActivate = useCallback(async () => {
    const key = licenseKey.trim()
    if (!key) return

    setActivating(true)
    setError("")

    try {
      const res = await api.post<{ success: boolean; tier?: string; error?: string }>(
        "/api/dashboard/license/activate",
        { license_key: key },
      )
      if (res.success) {
        setActivated(true)
        setTimeout(() => {
          setOpen(false)
          window.location.reload()
        }, 1500)
      } else {
        setError(res.error || t("upgrade.activationFailed", "Activation failed"))
      }
    } catch {
      setError(t("upgrade.activationFailed", "Activation failed. Check key format."))
    } finally {
      setActivating(false)
    }
  }, [licenseKey, t])

  if (!open || isPro) return null

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) setOpen(false) }}
    >
      <div className="w-full max-w-md rounded-2xl border border-border bg-card shadow-lg overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-6 pb-2">
          <div>
            <h2 className="font-display text-xl font-bold text-foreground">
              {t("upgrade.title", "Get Neural Memory Pro")}
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              {view === "choose" && t("upgrade.subtitle", "Choose your payment method")}
              {view === "email" && t("upgrade.enterEmailLabel", "Enter your email to receive the license key")}
              {view === "activate" && t("upgrade.enterKey", "Enter your license key")}
            </p>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="rounded-lg p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors cursor-pointer"
            aria-label={t("commandPalette.close", "Close")}
          >
            <X className="size-5" aria-hidden="true" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 pb-6 pt-4 space-y-3">
          {view === "choose" && (
            <>
              {/* Monthly — featured */}
              <button
                onClick={() => handleSelectPlan("monthly")}
                className="group relative flex w-full items-center gap-4 rounded-xl border-2 border-primary/50 bg-background px-5 py-4 text-left transition-all hover:border-primary hover:shadow-md cursor-pointer"
              >
                <span className="text-2xl" aria-hidden="true">{PRICING.monthly.flag}</span>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-foreground">
                    Pro Monthly
                  </p>
                  <p className="text-sm text-muted-foreground">{PRICING.monthly.method}</p>
                  <p className="text-sm mt-1">
                    <span className="font-mono font-bold text-foreground">{PRICING.monthly.price}</span>
                    <span className="text-muted-foreground">{PRICING.monthly.period}</span>
                  </p>
                </div>
                <CaretRight className="size-5 text-muted-foreground group-hover:text-foreground transition-colors" aria-hidden="true" />
              </button>

              {/* Yearly — save badge */}
              <button
                onClick={() => handleSelectPlan("yearly")}
                className="group relative flex w-full items-center gap-4 rounded-xl border border-border bg-background px-5 py-4 text-left transition-all hover:border-primary/40 hover:shadow-md cursor-pointer"
              >
                <span className="text-2xl" aria-hidden="true">{PRICING.yearly.flag}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="font-semibold text-foreground">Pro Yearly</p>
                    <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-bold text-primary">
                      {PRICING.yearly.badge}
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">{PRICING.yearly.method}</p>
                  <p className="text-sm mt-1">
                    <span className="font-mono font-bold text-foreground">{PRICING.yearly.price}</span>
                    <span className="text-muted-foreground">{PRICING.yearly.period}</span>
                  </p>
                </div>
                <CaretRight className="size-5 text-muted-foreground group-hover:text-foreground transition-colors" aria-hidden="true" />
              </button>

              {/* Vietnam QR */}
              <button
                onClick={() => handleSelectPlan("vn")}
                className="group flex w-full items-center gap-4 rounded-xl border border-border bg-background px-5 py-4 text-left transition-all hover:border-primary/40 hover:shadow-md cursor-pointer"
              >
                <span className="text-2xl" aria-hidden="true">{PRICING.vn.flag}</span>
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-foreground">Vietnam</p>
                  <p className="text-sm text-muted-foreground">{PRICING.vn.method}</p>
                  <p className="text-sm mt-1">
                    <span className="font-mono font-bold text-foreground">{PRICING.vn.price}</span>
                    <span className="text-muted-foreground">{PRICING.vn.period}</span>
                  </p>
                </div>
                <CaretRight className="size-5 text-muted-foreground group-hover:text-foreground transition-colors" aria-hidden="true" />
              </button>

              {/* Divider */}
              <div className="flex items-center gap-3 py-1">
                <div className="flex-1 border-t border-border" />
                <span className="text-xs text-muted-foreground">{t("upgrade.orActivate", "or activate a key")}</span>
                <div className="flex-1 border-t border-border" />
              </div>

              {/* Already have a key */}
              <button
                onClick={() => setView("activate")}
                className="group flex w-full items-center gap-4 rounded-xl border border-dashed border-border px-5 py-3 text-left transition-all hover:border-primary/40 cursor-pointer"
              >
                <Key className="size-5 text-muted-foreground" aria-hidden="true" />
                <span className="flex-1 text-sm font-medium text-muted-foreground group-hover:text-foreground">
                  {t("upgrade.haveKey", "I already have a license key")}
                </span>
                <CaretRight className="size-4 text-muted-foreground group-hover:text-foreground transition-colors" aria-hidden="true" />
              </button>
            </>
          )}

          {view === "email" && (
            <div className="space-y-4">
              <div>
                <label htmlFor="checkout-email" className="block text-sm font-medium text-foreground mb-2">
                  {t("upgrade.emailLabel", "Email (for license delivery)")}
                </label>
                <input
                  id="checkout-email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleCheckout() }}
                  placeholder="you@example.com"
                  className="w-full rounded-lg border border-input bg-background px-4 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  autoFocus
                />
                {error && (
                  <p className="mt-2 text-sm text-destructive">{error}</p>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                {t("upgrade.emailHint", "Your license key will be sent to this email after payment.")}
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => { setView("choose"); setError("") }}
                  className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-muted-foreground hover:bg-accent transition-colors cursor-pointer"
                >
                  {t("upgrade.back", "Back")}
                </button>
                <button
                  onClick={handleCheckout}
                  disabled={!email.trim() || activating}
                  className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
                >
                  {activating ? (
                    <SpinnerGap className="size-4 animate-spin" />
                  ) : (
                    t("upgrade.continue", "Continue to Payment")
                  )}
                </button>
              </div>
            </div>
          )}

          {view === "activate" && (
            <div className="space-y-4">
              {activated ? (
                <div className="flex flex-col items-center gap-3 py-6">
                  <div className="rounded-full bg-health-good/10 p-3">
                    <Check className="size-6 text-health-good" />
                  </div>
                  <p className="font-semibold text-foreground">
                    {t("upgrade.activated", "Pro activated!")}
                  </p>
                </div>
              ) : (
                <>
                  <div>
                    <label htmlFor="license-key" className="block text-sm font-medium text-foreground mb-2">
                      {t("upgrade.licenseKey", "License Key")}
                    </label>
                    <input
                      id="license-key"
                      type="text"
                      value={licenseKey}
                      onChange={(e) => setLicenseKey(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter") handleActivate() }}
                      placeholder="NM-PRO-XXXX-XXXX-XXXX"
                      className="w-full rounded-lg border border-input bg-background px-4 py-2.5 font-mono text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                      autoFocus
                    />
                    {error && (
                      <p className="mt-2 text-sm text-destructive">{error}</p>
                    )}
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={() => setView("choose")}
                      className="flex-1 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-muted-foreground hover:bg-accent transition-colors cursor-pointer"
                    >
                      {t("upgrade.back", "Back")}
                    </button>
                    <button
                      onClick={handleActivate}
                      disabled={!licenseKey.trim() || activating}
                      className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
                    >
                      {activating ? (
                        <SpinnerGap className="size-4 animate-spin" />
                      ) : (
                        t("upgrade.activate", "Activate")
                      )}
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
