import { useEffect, useMemo, useRef, useState } from "react"
import type { BrainLayout } from "../engine/types"

/**
 * Ambient activation stream — simulates "living brain" pulses client-side
 * because `/api/graph` doesn't return neuron activation levels and the
 * per-neuron state endpoint doesn't scale to polling 500 nodes.
 *
 * Two sources feed the pulse map:
 *   1. **Delta pulses** — any neuron whose id is new or whose `radius`
 *      changes across layout updates fires a `pulseDurationMs` pulse.
 *      Represents real graph change.
 *   2. **Ambient pulses** — every `ambientIntervalMs` ±25% jitter, pick a
 *      degree-weighted neuron and fire a pulse. Gives the "breathing" feel
 *      between data refreshes.
 *
 * Pulses live in a `Map<id, startTime>` stored behind a ref so r3f can read
 * them every frame via `useFrame` without triggering React re-renders.
 * Only lifetime stats (activeCount / totalPulses) are React state and only
 * update on the sweep interval — so StatsBar counters reflect reality
 * without thrashing the tree.
 */

export interface ActivationStreamOptions {
  enabled: boolean
  ambientIntervalMs?: number
  pulseDurationMs?: number
}

export interface ActivationStreamResult {
  /** Read-only accessor for the pulse map — used by r3f useFrame. */
  getPulses: () => ReadonlyMap<string, number>
  /**
   * Monotonic tick — bumps each time a new pulse starts. Used to wake the
   * r3f `frameloop="demand"` loop: a component inside Canvas watches this
   * and calls `invalidate()` so `useFrame` resumes. Without this, pulses
   * would sit silently in the ref between idle frames.
   */
  kickTick: number
  pulseDurationMs: number
  activeCount: number
  totalPulses: number
}

const DEFAULT_AMBIENT = 2500
const DEFAULT_DURATION = 900

export function useActivationStream(
  layout: BrainLayout | null,
  {
    enabled,
    ambientIntervalMs = DEFAULT_AMBIENT,
    pulseDurationMs = DEFAULT_DURATION,
  }: ActivationStreamOptions,
): ActivationStreamResult {
  const pulsesRef = useRef<Map<string, number>>(new Map())
  const prevRadiiRef = useRef<Map<string, number>>(new Map())
  const [activeCount, setActiveCount] = useState(0)
  const [totalPulses, setTotalPulses] = useState(0)
  const [kickTick, setKickTick] = useState(0)

  const weightedPool = useMemo<ReadonlyArray<{ id: string; weight: number }>>(() => {
    if (!layout) return []
    return layout.neurons.map((n) => {
      const degree = layout.neighbors.get(n.id)?.length ?? 0
      return { id: n.id, weight: 1 + Math.min(degree, 20) }
    })
  }, [layout])

  // Delta detection on layout identity change.
  useEffect(() => {
    if (!layout) return
    const nextRadii = new Map<string, number>()
    const deltas: string[] = []
    const seed = prevRadiiRef.current.size === 0
    for (const n of layout.neurons) {
      nextRadii.set(n.id, n.radius)
      const prevRadius = prevRadiiRef.current.get(n.id)
      if (seed) continue
      if (prevRadius === undefined) {
        deltas.push(n.id)
      } else if (Math.abs(prevRadius - n.radius) > 0.05) {
        deltas.push(n.id)
      }
    }
    prevRadiiRef.current = nextRadii

    if (!enabled || deltas.length === 0) return
    const now = performance.now()
    for (const id of deltas) pulsesRef.current.set(id, now)
    // Defer the React state bump to a microtask so the setState lands after
    // the current effect finishes. The sibling `pulsesRef` update is still
    // synchronous — r3f picks it up on the next frame via getPulses().
    queueMicrotask(() => {
      setTotalPulses((n) => n + deltas.length)
      setKickTick((n) => n + 1)
    })
  }, [enabled, layout])

  // Ambient pulse ticker.
  useEffect(() => {
    if (!enabled || weightedPool.length === 0) return
    let timer: ReturnType<typeof setTimeout> | null = null

    const schedule = () => {
      const jitter = ambientIntervalMs * (0.75 + Math.random() * 0.5)
      timer = setTimeout(() => {
        const id = pickWeighted(weightedPool)
        if (id) {
          pulsesRef.current.set(id, performance.now())
          setTotalPulses((n) => n + 1)
          setKickTick((n) => n + 1)
        }
        schedule()
      }, jitter)
    }
    schedule()
    return () => {
      if (timer) clearTimeout(timer)
    }
  }, [enabled, weightedPool, ambientIntervalMs])

  // Sweep expired pulses + mirror activeCount into React for HUD readouts.
  useEffect(() => {
    const interval = setInterval(() => {
      const cutoff = performance.now() - pulseDurationMs
      const map = pulsesRef.current
      for (const [id, ts] of map) {
        if (ts < cutoff) map.delete(id)
      }
      setActiveCount(map.size)
    }, Math.max(200, pulseDurationMs / 2))
    return () => clearInterval(interval)
  }, [pulseDurationMs])

  // When disabled, clear pending pulses so visuals stop immediately.
  // P4 review fix (H2): also bump kickTick so the demand frameloop wakes
  // once and NeuronNodes.useFrame can paint the cleared state — otherwise
  // the scene keeps showing the LAST pulsed frame until some unrelated
  // invalidation comes along.
  useEffect(() => {
    if (enabled) return
    pulsesRef.current.clear()
    queueMicrotask(() => {
      setActiveCount(0)
      setKickTick((n) => n + 1)
    })
  }, [enabled])

  const getPulses = useMemo(
    () => (): ReadonlyMap<string, number> => pulsesRef.current,
    [],
  )

  return { getPulses, kickTick, pulseDurationMs, activeCount, totalPulses }
}

function pickWeighted(pool: ReadonlyArray<{ id: string; weight: number }>): string | null {
  if (pool.length === 0) return null
  let total = 0
  for (const p of pool) total += p.weight
  let roll = Math.random() * total
  for (const p of pool) {
    roll -= p.weight
    if (roll <= 0) return p.id
  }
  return pool[pool.length - 1].id
}
