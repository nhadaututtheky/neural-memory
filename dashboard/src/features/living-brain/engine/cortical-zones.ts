import type { ForceNode } from "./force-3d"

/**
 * Cortical region anchors, expressed as fractions of brain radii (-1..1).
 * Multiplied by (rx, ry, rz) at apply-time.
 *
 *   +Z = frontal   |  -Z = occipital
 *   +Y = superior  |  -Y = inferior
 *   ±X = right / left hemisphere
 *
 * These are real regions of the shell, and — unlike the map this replaced —
 * they are handed out to detected COMMUNITIES rather than to neuron types.
 * Anchoring by type meant position duplicated the color channel and pulled
 * every same-type node into one clump; anchoring by community means physical
 * proximity reflects "these memories actually link to each other".
 *
 * Both hemispheres are represented so the cloud fills the whole volume instead
 * of collapsing onto the midline.
 */
export const REGION_ANCHORS: ReadonlyArray<readonly [number, number, number]> = [
  [0.42, 0.32, 0.55], // right frontal
  [-0.42, 0.32, 0.55], // left frontal
  [0.52, 0.28, -0.08], // right parietal
  [-0.52, 0.28, -0.08], // left parietal
  [0.58, -0.18, 0.12], // right temporal
  [-0.58, -0.18, 0.12], // left temporal
  [0.28, 0.14, -0.58], // right occipital
  [-0.28, 0.14, -0.58], // left occipital
  [0.0, 0.52, 0.12], // superior medial / vertex
  [0.0, -0.32, -0.52], // cerebellar
]

/** Nodes with no community (isolated, no synapses) drift to the center. */
const UNASSIGNED_ANCHOR: readonly [number, number, number] = [0, 0, 0]

export interface CorticalZoneOptions {
  radii: readonly [number, number, number]
  /** node id → community id, from `detectCommunities`. */
  communityById: ReadonlyMap<string, string>
  /** Community ids largest-first; index into REGION_ANCHORS follows this order. */
  orderedCommunities: readonly string[]
  strength?: number
}

/**
 * Soft attraction toward each node's community region.
 *
 * Called per simulation tick; scales the nudge by current alpha so motion cools
 * as the layout settles. Capped at 0.03 to prevent early-tick overshoot when
 * alpha is near 1.0.
 *
 * The pull is deliberately weak — it biases where a cluster lands without
 * overriding the link forces that decide the cluster's internal shape.
 */
export function applyCorticalZones(
  nodes: ForceNode[],
  alpha: number,
  opts: CorticalZoneOptions,
): void {
  const [rx, ry, rz] = opts.radii
  const k = Math.min(0.03, (opts.strength ?? 0.05) * alpha)

  // Communities beyond the anchor count wrap around, so a brain with many
  // small clusters still spreads them out rather than piling the tail at 0,0,0.
  const anchorByCommunity = new Map<string, readonly [number, number, number]>()
  opts.orderedCommunities.forEach((community, i) => {
    anchorByCommunity.set(community, REGION_ANCHORS[i % REGION_ANCHORS.length])
  })

  for (const n of nodes) {
    const community = opts.communityById.get(n.id)
    const anchor =
      (community !== undefined ? anchorByCommunity.get(community) : undefined) ??
      UNASSIGNED_ANCHOR
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
