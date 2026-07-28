import type { ForceNode } from "./force-3d"
import { MAX_GYRI_INWARD, MIN_RADIUS_SCALE, brainRadiusScale } from "./brain-shape"

/**
 * Brain-surface containment force.
 *
 * For each node, evaluate the ellipsoid SDF at its position:
 *   f(p) = (p.x/rx)^2 + (p.y/ry)^2 + (p.z/rz)^2
 * If `f` exceeds the local limit, apply a radial velocity kick toward the
 * origin proportional to how far outside the surface the node has drifted.
 * Nodes already inside get no force, so the cloud is not squashed centrally.
 *
 * The limit is per-direction rather than a constant: it follows the same
 * `brainRadiusScale` the mesh is built from, so nodes tuck into the fissure
 * and under the cerebellum instead of poking through them. The smooth
 * (gyri-free) variant is used because this runs per node per tick — the
 * fold amplitude is subtracted instead, which is conservative.
 */
export interface BrainConstraintOptions {
  radii: readonly [number, number, number]
  // Target inset: nodes are nudged back to this fraction of the shell radius.
  insetFactor?: number
  // How strongly the kick pulls outside nodes back (0..1).
  strength?: number
}

export function applyBrainConstraint(
  nodes: ForceNode[],
  opts: BrainConstraintOptions,
): void {
  const [rx, ry, rz] = opts.radii
  const inset = opts.insetFactor ?? 0.95
  const strength = opts.strength ?? 0.25
  // Cheap early-out. Must use the conservative FLOOR of the surface, not
  // `inset` itself: inside a groove the real limit is smaller than `inset`, so
  // testing against `inset` would wave through nodes sitting in the fissure.
  const floorLimit = inset * MIN_RADIUS_SCALE
  const floorLimitSq = floorLimit * floorLimit

  for (const n of nodes) {
    const x = n.x ?? 0
    const y = n.y ?? 0
    const z = n.z ?? 0
    const fx = x / rx
    const fy = y / ry
    const fz = z / rz
    const f = fx * fx + fy * fy + fz * fz
    if (f <= floorLimitSq) continue

    // Local surface limit along this node's direction. The smooth term already
    // includes both grooves, so only the skipped fold amplitude is subtracted.
    const len = Math.sqrt(f) || 1
    const localScale =
      brainRadiusScale(fx / len, fy / len, fz / len, false) - MAX_GYRI_INWARD
    const localLimit = inset * Math.max(0.35, localScale)
    if (f <= localLimit * localLimit) continue

    // Amount node is outside (0 at boundary, scales with penetration depth).
    const excess = Math.sqrt(f) - localLimit
    // Pull back along normalized gradient direction (∇f points outward).
    const gLen = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1
    const kickX = (-fx / gLen) * excess * strength
    const kickY = (-fy / gLen) * excess * strength
    const kickZ = (-fz / gLen) * excess * strength

    n.vx = (n.vx ?? 0) + kickX * rx
    n.vy = (n.vy ?? 0) + kickY * ry
    n.vz = (n.vz ?? 0) + kickZ * rz
  }
}
