declare module "d3-force-3d" {
  export interface SimulationNodeDatum {
    index?: number
    x?: number
    y?: number
    z?: number
    vx?: number
    vy?: number
    vz?: number
    fx?: number | null
    fy?: number | null
    fz?: number | null
  }

  export interface SimulationLinkDatum<NodeDatum extends SimulationNodeDatum> {
    source: NodeDatum | string | number
    target: NodeDatum | string | number
    index?: number
  }

  export interface Simulation<
    NodeDatum extends SimulationNodeDatum,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    LinkDatum extends SimulationLinkDatum<NodeDatum> | undefined = undefined,
  > {
    tick(iterations?: number): this
    stop(): this
    alpha(): number
    alpha(value: number): this
    alphaDecay(): number
    alphaDecay(decay: number): this
    force(name: string): unknown
    force(name: string, force: unknown): this
    nodes(): NodeDatum[]
    nodes(nodes: NodeDatum[]): this
  }

  export function forceSimulation<NodeDatum extends SimulationNodeDatum>(
    nodes?: NodeDatum[],
    numDimensions?: number,
  ): Simulation<NodeDatum, undefined>

  interface ForceManyBody<NodeDatum extends SimulationNodeDatum> {
    strength(s: number | ((d: NodeDatum) => number)): this
    theta(t: number): this
    distanceMin(d: number): this
    distanceMax(d: number): this
  }
  export function forceManyBody<NodeDatum extends SimulationNodeDatum>(): ForceManyBody<NodeDatum>

  interface ForceLink<
    NodeDatum extends SimulationNodeDatum,
    LinkDatum extends SimulationLinkDatum<NodeDatum>,
  > {
    id(fn: (d: NodeDatum) => string | number): this
    distance(d: number | ((l: LinkDatum) => number)): this
    strength(s: number | ((l: LinkDatum) => number)): this
    links(links?: LinkDatum[]): this
  }
  export function forceLink<
    NodeDatum extends SimulationNodeDatum,
    LinkDatum extends SimulationLinkDatum<NodeDatum>,
  >(links?: LinkDatum[]): ForceLink<NodeDatum, LinkDatum>

  interface ForceCenter {
    x(x: number): this
    y(y: number): this
    z(z: number): this
  }
  export function forceCenter(x?: number, y?: number, z?: number): ForceCenter
}
