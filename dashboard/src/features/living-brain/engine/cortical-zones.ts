import type { ForceNode } from "./force-3d"

/**
 * Anatomical anchor points, expressed as fractions of brain radii (-1..1).
 * Multiplied by (rx, ry, rz) at apply-time.
 *
 *   +Z = frontal   |  -Z = occipital
 *   +Y = parietal  |  -Y = cerebellar
 *   ±X = temporal
 */
export const ZONE_ANCHORS: Readonly<Record<string, readonly [number, number, number]>> = {
  decision: [0, 0.25, 0.7],
  concept: [0.7, 0.45, 0.0],
  entity: [0, 0.7, 0.15],
  action: [0, 0.0, 0.3],
  state: [0, 0.1, -0.65],
  time: [0, -0.1, 0.0],
  relation: [-0.65, 0.35, 0.1],
  preference: [0.0, 0.55, -0.35],
  attribute: [-0.5, -0.15, 0.3],
  other: [0, 0, 0],
}

export interface CorticalZoneOptions {
  radii: readonly [number, number, number]
  typeById: ReadonlyMap<string, string>
  strength?: number
}

/**
 * Soft attraction toward each neuron-type's anatomical anchor.
 * Called per simulation tick; scales nudge by current alpha so motion
 * cools as the layout settles. Capped at 0.03 to prevent early-tick
 * overshoot when alpha is near 1.0.
 */
export function applyCorticalZones(
  nodes: ForceNode[],
  alpha: number,
  opts: CorticalZoneOptions,
): void {
  const [rx, ry, rz] = opts.radii
  const k = Math.min(0.03, (opts.strength ?? 0.05) * alpha)

  for (const n of nodes) {
    const type = opts.typeById.get(n.id)
    if (!type) continue
    const anchor = ZONE_ANCHORS[type] ?? ZONE_ANCHORS.other
    const tx = anchor[0] * rx
    const ty = anchor[1] * ry
    const tz = anchor[2] * rz
    const x = n.x ?? 0
    const y = n.y ?? 0
    const z = n.z ?? 0
    n.vx = (n.vx ?? 0) + (tx - x) * k
    n.vy = (n.vy ?? 0) + (ty - y) * k
    n.vz = (n.vz ?? 0) + (tz - z) * k
  }
}
