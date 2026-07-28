/**
 * Categorical color palette — the single source of truth.
 *
 * Rule: within one axis, no hex may appear twice. Two labels sharing a color
 * inside the same legend is the bug this module exists to prevent — before it,
 * `entity` rendered #06b6d4 on /graph but #14b8a6 on /visualize, and #14b8a6
 * canonically means `attribute`.
 *
 * Across axes, reuse is fine and often desirable: hot and error are both red
 * because they read the same way. Two axes never share a legend.
 *
 * Brand chrome (buttons, focus rings, active nav) does NOT live here — that is
 * `--color-primary` in index.css.
 */

/** Axis 1 — neuron types. Used by the graph, the 3D brain, and mindmaps. */
export const NEURON_TYPE_COLORS: Record<string, string> = {
  concept: "#6366f1", // indigo
  entity: "#06b6d4", // cyan
  time: "#f59e0b", // amber
  action: "#059669", // emerald
  state: "#8b5cf6", // violet
  relation: "#ec4899", // pink
  attribute: "#14b8a6", // teal
  other: "#a8a29e", // stone
}

/**
 * Axis 2 — memory types (the `nmem_remember` taxonomy).
 *
 * `concept` and `entity` intentionally match Axis 1: same word, same meaning,
 * so they must not change color when the user moves between pages. Every other
 * entry is deliberately offset from its Axis-1 neighbour (insight is yellow,
 * not `time`'s amber; workflow is sky, not `entity`'s cyan) so the two
 * taxonomies stay readable side by side.
 */
export const MEMORY_TYPE_COLORS: Record<string, string> = {
  concept: "#6366f1", // indigo — shared with Axis 1
  entity: "#06b6d4", // cyan   — shared with Axis 1
  fact: "#3b82f6", // blue
  decision: "#9333ea", // purple
  error: "#ef4444", // red
  insight: "#eab308", // yellow
  preference: "#10b981", // green
  workflow: "#0ea5e9", // sky
  instruction: "#f43f5e", // rose
  pattern: "#f97316", // orange
}

/** Axis 3 — storage tiers. Red/amber/blue reads as hot/warm/cold intuitively. */
export const TIER_COLORS: Record<string, string> = {
  hot: "#ef4444",
  warm: "#f59e0b",
  cold: "#3b82f6",
}

/**
 * Axis 4 — generic chart series, for data with no inherent category.
 *
 * These are `var()` references, so a series automatically picks up the
 * dark-theme override of `--color-chart-*` instead of staying a light-theme
 * pigment on a near-black background. Prefer `chartSeriesColor()` over adding
 * another local hex array to a page component.
 */
export const CHART_SERIES: readonly string[] = [
  "var(--color-chart-1)",
  "var(--color-chart-2)",
  "var(--color-chart-3)",
  "var(--color-chart-4)",
  "var(--color-chart-5)",
]

/**
 * Axis 5 — mindmap structural roles. These are not neuron types: `root` is the
 * fiber itself and `group` is a synthetic grouping node, so they deliberately
 * sit outside Axis 1 rather than competing with a real type for a hue.
 */
export const MINDMAP_ROLE_COLORS: Record<string, string> = {
  root: "#f97316", // orange
  group: "#64748b", // slate
}

/** Axis 6 — synapse types, for edge coloring in the mindmap. */
export const SYNAPSE_TYPE_COLORS: Record<string, string> = {
  CAUSED_BY: "#ef4444",
  RELATES_TO: "#6366f1",
  PART_OF: "#059669",
  LEADS_TO: "#f59e0b",
  CONTAINS: "#06b6d4",
  DEPENDS_ON: "#ec4899",
  SIMILAR_TO: "#8b5cf6",
  CONTRAST: "#f97316",
  RESOLVED_BY: "#10b981",
  TEMPORAL: "#eab308",
  SEMANTIC: "#a855f7",
}

/** Neutral used when a value falls outside its axis. */
export const EDGE_DEFAULT_COLOR = "#94a3b8"

export const NEURON_DEFAULT_COLOR = NEURON_TYPE_COLORS.other

export function colorForNeuronType(type: string): string {
  return NEURON_TYPE_COLORS[type] ?? NEURON_DEFAULT_COLOR
}

export function colorForMemoryType(type: string): string {
  return MEMORY_TYPE_COLORS[type] ?? NEURON_DEFAULT_COLOR
}

export function colorForSynapseType(type: string): string {
  return SYNAPSE_TYPE_COLORS[type] ?? EDGE_DEFAULT_COLOR
}

export function colorForTier(tier: string): string {
  return TIER_COLORS[tier] ?? NEURON_DEFAULT_COLOR
}

/** Cycles the chart ramp so series counts beyond 5 still get stable colors. */
export function chartSeriesColor(index: number): string {
  return CHART_SERIES[index % CHART_SERIES.length]
}

/**
 * Appends an 8-digit-hex alpha suffix (e.g. `"15"`) to a 6-digit hex.
 *
 * Lets callers derive a translucent fill from a palette entry instead of
 * hand-maintaining a parallel map of the same colors with alpha baked in.
 * Returns the input untouched if it is not a plain `#rrggbb` value.
 */
export function withAlpha(hex: string, alphaHex: string): string {
  return /^#[0-9a-fA-F]{6}$/.test(hex) ? `${hex}${alphaHex}` : hex
}
