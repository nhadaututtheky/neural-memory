/**
 * Community detection over the synapse graph — weighted label propagation.
 *
 * This is what makes a node's POSITION carry information. The previous
 * `ZONE_ANCHORS` map pulled nodes toward a fixed point per neuron *type*, which
 * told the viewer nothing the color channel wasn't already saying, and actively
 * fought the force layout by dragging same-type nodes apart into one blob each.
 *
 * Grouping by graph structure instead means "memories that link to each other
 * sit together", which is the relationship the brain metaphor is claiming in
 * the first place.
 *
 * Why not embeddings: the SQLite brains have none. There is no `embedding`
 * column in that schema at all, the feature is Pro-gated and off by default,
 * and even when enabled the vector goes to a sidecar HNSW index that no API
 * exposes. Link structure is the strongest signal actually reachable from
 * `/api/graph` today.
 *
 * Label propagation is O(iterations · edges) and runs on at most a few hundred
 * nodes here, so it costs well under a frame.
 */

export interface CommunityInput {
  id: string
  source: string
  target: string
  weight: number
}

export interface CommunityResult {
  /** node id → community id (a node id, the community's representative). */
  communityById: Map<string, string>
  /** Community ids ordered by member count, largest first. */
  orderedCommunities: string[]
}

const MAX_ITERATIONS = 12

/**
 * Assigns each node a community label.
 *
 * Deterministic: nodes are visited in sorted id order and ties are broken by
 * the lexicographically smallest label, so the same graph always produces the
 * same layout. A random visit order would reshuffle the brain on every reload.
 */
export function detectCommunities(
  nodeIds: readonly string[],
  links: ReadonlyArray<{ source: string; target: string; weight: number }>,
): CommunityResult {
  const labels = new Map<string, string>()
  for (const id of nodeIds) labels.set(id, id)

  // Weighted adjacency, built once.
  const neighbors = new Map<string, Array<{ id: string; weight: number }>>()
  for (const id of nodeIds) neighbors.set(id, [])
  for (const l of links) {
    const a = neighbors.get(l.source)
    const b = neighbors.get(l.target)
    if (!a || !b) continue
    const w = l.weight > 0 ? l.weight : 0.1
    a.push({ id: l.target, weight: w })
    b.push({ id: l.source, weight: w })
  }

  const visitOrder = [...nodeIds].sort()

  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    let changed = false

    for (const id of visitOrder) {
      const adj = neighbors.get(id)
      if (!adj || adj.length === 0) continue

      // Sum edge weight per neighboring label; take the heaviest.
      const score = new Map<string, number>()
      for (const { id: other, weight } of adj) {
        const lab = labels.get(other)
        if (lab === undefined) continue
        score.set(lab, (score.get(lab) ?? 0) + weight)
      }

      let best: string | undefined
      let bestScore = -1
      for (const [lab, s] of score) {
        // Strict > keeps the first seen on ties; the explicit lexicographic
        // comparison then makes "first seen" independent of Map order.
        if (s > bestScore || (s === bestScore && best !== undefined && lab < best)) {
          best = lab
          bestScore = s
        }
      }

      if (best !== undefined && best !== labels.get(id)) {
        labels.set(id, best)
        changed = true
      }
    }

    if (!changed) break
  }

  const counts = new Map<string, number>()
  for (const lab of labels.values()) {
    counts.set(lab, (counts.get(lab) ?? 0) + 1)
  }
  const orderedCommunities = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .map(([lab]) => lab)

  return { communityById: labels, orderedCommunities }
}
