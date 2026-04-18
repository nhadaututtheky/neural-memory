import type { ForceNode } from "./force-3d"

/**
 * Ellipsoid containment force.
 *
 * For each node, evaluate the ellipsoid SDF at its position:
 *   f(p) = (p.x/rx)^2 + (p.y/ry)^2 + (p.z/rz)^2
 * If `f > insetFactor^2`, apply a radial velocity kick toward the origin
 * proportional to how far outside the surface the node has drifted. This
 * keeps nodes inside the brain without globally squashing them toward the
 * center (no force is applied to nodes already inside).
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
  const limitSq = inset * inset

  for (const n of nodes) {
    const x = n.x ?? 0
    const y = n.y ?? 0
    const z = n.z ?? 0
    const fx = x / rx
    const fy = y / ry
    const fz = z / rz
    const f = fx * fx + fy * fy + fz * fz
    if (f <= limitSq) continue

    // Amount node is outside (0 at boundary, scales with penetration depth).
    const excess = Math.sqrt(f) - inset
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
