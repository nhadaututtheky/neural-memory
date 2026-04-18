import { useEffect, useRef, useState } from "react"

/**
 * Animates a numeric value toward the target over `durationMs`.
 * Uses requestAnimationFrame with ease-out cubic. First render snaps to
 * the target (no `0 → N` flash on load) — subsequent changes animate.
 *
 * Interrupt-safe: if `target` changes mid-tween, the new animation starts
 * from the currently-displayed value (not the original `from`) so the
 * number never jumps backwards.
 */
export function useCountUp(target: number, durationMs = 600): number {
  const [display, setDisplay] = useState<number>(target)
  const displayRef = useRef<number>(target)
  const rafRef = useRef<number | null>(null)

  // Mirror display into a ref so the effect below can read the live value
  // without making `display` a dependency (which would restart the tween
  // on every animated frame).
  useEffect(() => {
    displayRef.current = display
  }, [display])

  useEffect(() => {
    const from = displayRef.current
    const to = target
    if (from === to) return
    const start = performance.now()

    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs)
      const eased = 1 - Math.pow(1 - t, 3)
      const value = Math.round(from + (to - from) * eased)
      setDisplay(value)
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [target, durationMs])

  return display
}
