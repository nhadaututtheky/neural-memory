/**
 * Procedural brain surface — the single definition of "where the shell is".
 *
 * Both the rendered mesh (`BrainShell`) and the layout containment force
 * (`applyBrainConstraint`) evaluate this, so a node can never end up outside a
 * sulcus that the geometry actually carved. Previously the mesh was a rippled
 * ellipsoid and the constraint was a plain ellipsoid SDF; they only agreed
 * because the ripple was too shallow to matter.
 *
 * Anatomy is procedural on purpose: no `.glb` asset, no new dependency. A real
 * scanned mesh would need licensing plus BVH point-in-mesh tests to replace the
 * closed-form containment below.
 *
 * Axes match the rest of the feature:
 *   +Z frontal · -Z occipital · +Y superior · -Y inferior · ±X left/right
 */

/* ---------------------------------------------------------------- */
/*  Noise                                                            */
/* ---------------------------------------------------------------- */

/** Integer hash → [0,1). Deterministic, so the brain looks the same each load. */
function hash3(i: number, j: number, k: number): number {
  let h = Math.imul(i, 374761393) + Math.imul(j, 668265263) + Math.imul(k, 1274126177)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

/** Quintic smoothstep — C2 continuous, so no facet banding on the surface. */
function smootherstep(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10)
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** 3D value noise in [-1, 1]. */
function valueNoise3(x: number, y: number, z: number): number {
  const i = Math.floor(x)
  const j = Math.floor(y)
  const k = Math.floor(z)
  const u = smootherstep(x - i)
  const v = smootherstep(y - j)
  const w = smootherstep(z - k)

  const c00 = lerp(hash3(i, j, k), hash3(i + 1, j, k), u)
  const c10 = lerp(hash3(i, j + 1, k), hash3(i + 1, j + 1, k), u)
  const c01 = lerp(hash3(i, j, k + 1), hash3(i + 1, j, k + 1), u)
  const c11 = lerp(hash3(i, j + 1, k + 1), hash3(i + 1, j + 1, k + 1), u)

  return lerp(lerp(c00, c10, v), lerp(c01, c11, v), w) * 2 - 1
}

/** Fractal sum. More octaves = finer folds. */
function fbm(x: number, y: number, z: number, octaves: number): number {
  let amp = 1
  let freq = 1
  let sum = 0
  let norm = 0
  for (let o = 0; o < octaves; o++) {
    sum += amp * valueNoise3(x * freq, y * freq, z * freq)
    norm += amp
    amp *= 0.5
    freq *= 2
  }
  return sum / norm
}

/* ---------------------------------------------------------------- */
/*  Shape constants                                                  */
/* ---------------------------------------------------------------- */

/** Fold frequency. Higher = more, tighter gyri. */
const GYRI_FREQ = 4.2
const GYRI_OCTAVES = 4
/** Peak-to-trough fold depth, as a fraction of radius. */
const GYRI_AMP = 0.085

/** Longitudinal fissure — the groove splitting the two hemispheres. */
const FISSURE_WIDTH = 0.16
const FISSURE_DEPTH = 0.13

/** Cerebellum — bulge low and behind the cerebrum. */
const CEREBELLUM_DIR: readonly [number, number, number] = [0, -0.6, -0.8]
const CEREBELLUM_AMP = 0.17
/** Transverse fissure separating cerebellum from the occipital lobe. */
const TRANSVERSE_DEPTH = 0.075

/**
 * Deepest the fold noise ever cuts inward. Callers that evaluate the smooth
 * (gyri-free) surface subtract exactly this to stay conservative — the
 * anatomical grooves are already included in the smooth term, so subtracting
 * anything more would double-count them.
 */
export const MAX_GYRI_INWARD = GYRI_AMP * 0.5

/**
 * Conservative floor for the radius multiplier: below this, a point is inside
 * the surface no matter which direction it lies in. Used as a cheap early-out
 * so the per-node check can skip evaluating the surface at all. The two
 * grooves cannot actually reach maximum depth in the same direction, so this
 * is a genuine lower bound rather than a tight one.
 */
export const MIN_RADIUS_SCALE = 1 - FISSURE_DEPTH - TRANSVERSE_DEPTH - MAX_GYRI_INWARD

/* ---------------------------------------------------------------- */
/*  Surface                                                          */
/* ---------------------------------------------------------------- */

/**
 * Radius multiplier along a unit direction — 1.0 is the base ellipsoid.
 *
 * `withGyri: false` skips the noise octaves and returns only the smooth
 * anatomical terms. The constraint force uses that cheap path (it runs per node
 * per tick) and subtracts `GYRI_AMP * 0.5` for a conservative lower bound.
 */
export function brainRadiusScale(
  nx: number,
  ny: number,
  nz: number,
  withGyri = true,
): number {
  let scale = 1

  // Longitudinal fissure: a groove at the midline (x≈0), deepest on the
  // superior surface and fading toward the frontal/occipital poles so the
  // hemispheres stay joined front and back rather than splitting in two.
  const midline = Math.exp(-(nx * nx) / (FISSURE_WIDTH * FISSURE_WIDTH))
  const superior = Math.max(0, ny)
  scale -= FISSURE_DEPTH * midline * superior

  // Cerebellum: bulge where the direction aligns with posterior-inferior.
  const cbDot =
    nx * CEREBELLUM_DIR[0] + ny * CEREBELLUM_DIR[1] + nz * CEREBELLUM_DIR[2]
  if (cbDot > 0) {
    scale += CEREBELLUM_AMP * Math.pow(cbDot, 2.4)
    // Groove where the cerebellum meets the occipital lobe.
    const seam = Math.exp(-Math.pow((cbDot - 0.62) / 0.13, 2))
    scale -= TRANSVERSE_DEPTH * seam
  }

  if (withGyri) {
    // Ridged noise reads as folds: broad crests with narrow valleys between,
    // which is the way round real gyri and sulci sit.
    const ridged = 1 - Math.abs(fbm(nx * GYRI_FREQ, ny * GYRI_FREQ, nz * GYRI_FREQ, GYRI_OCTAVES))
    scale += (ridged - 0.5) * GYRI_AMP
  }

  return scale
}

/** Brainstem placement, in radius fractions. Rendered as its own tapered mesh. */
export const BRAINSTEM = {
  topRadius: 0.13,
  bottomRadius: 0.085,
  /** Length as a fraction of ry. */
  length: 0.62,
  /** Center offset (fractions of rx, ry, rz). */
  offset: [0, -0.72, -0.12] as const,
  /** Forward tilt in radians — the stem angles slightly anterior. */
  tilt: 0.22,
} as const
