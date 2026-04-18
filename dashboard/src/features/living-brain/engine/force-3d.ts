import {
  forceSimulation,
  forceManyBody,
  forceLink,
  forceCenter,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3-force-3d"
import { applyBrainConstraint } from "./brain-constraint"
import { applyCorticalZones } from "./cortical-zones"

export interface ForceNode extends SimulationNodeDatum {
  id: string
  x?: number
  y?: number
  z?: number
  vx?: number
  vy?: number
  vz?: number
}

export interface ForceLink extends SimulationLinkDatum<ForceNode> {
  source: string | ForceNode
  target: string | ForceNode
  weight: number
}

export interface LayoutResult {
  nodes: ReadonlyArray<Readonly<ForceNode>>
  links: ReadonlyArray<Readonly<ForceLink>>
}

export interface ForceLayoutOptions {
  iterations?: number
  /** Brain shell half-extents along X, Y, Z. Used for containment + cortical zones. */
  radii?: readonly [number, number, number]
  /** Whether to apply brain-shell containment. Default: true if radii set. */
  constrain?: boolean
  /** Whether to apply cortical-zone attraction. Default: true if type info present. */
  zones?: boolean
}

export const DEFAULT_BRAIN_RADII = [80, 60, 95] as const

export function runForceLayout(
  rawNodes: ReadonlyArray<{ id: string; type?: string }>,
  rawLinks: ReadonlyArray<{ source: string; target: string; weight: number }>,
  opts: ForceLayoutOptions = {},
): LayoutResult {
  const iterations = opts.iterations ?? 200
  const radii = opts.radii ?? DEFAULT_BRAIN_RADII
  const constrain = opts.constrain ?? true
  const zones = opts.zones ?? rawNodes.some((n) => n.type !== undefined)

  const nodes: ForceNode[] = rawNodes.map((n) => ({ id: n.id }))
  const nodeIndex = new Map(nodes.map((n) => [n.id, n]))
  const typeById = new Map<string, string>()
  for (const rn of rawNodes) {
    if (rn.type) typeById.set(rn.id, rn.type)
  }

  const links: ForceLink[] = rawLinks
    .filter((l) => nodeIndex.has(l.source) && nodeIndex.has(l.target))
    .map((l) => ({
      source: l.source,
      target: l.target,
      weight: l.weight,
    }))

  const sim = forceSimulation(nodes, 3)
    .force(
      "charge",
      forceManyBody<ForceNode>().strength(-40).theta(0.9),
    )
    .force(
      "link",
      forceLink<ForceNode, ForceLink>(links)
        .id((d) => d.id)
        .distance(25)
        .strength((l) => Math.min(1, l.weight)),
    )
    .force("center", forceCenter(0, 0, 0))
    .alpha(1)
    .alphaDecay(1 - Math.pow(0.001, 1 / iterations))
    .stop()

  for (let i = 0; i < iterations; i++) {
    sim.tick()
    const alpha = sim.alpha()
    // Constraint BEFORE zones so shell boundary is enforced first, then
    // zone attraction adds inward nudge on top (can't escape the shell).
    if (constrain) {
      applyBrainConstraint(nodes, { radii, insetFactor: 0.95, strength: 0.25 })
    }
    if (zones) {
      applyCorticalZones(nodes, alpha, { radii, typeById, strength: 0.05 })
    }
  }

  return { nodes, links }
}
