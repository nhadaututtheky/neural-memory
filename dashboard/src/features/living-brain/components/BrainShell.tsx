import { useEffect, useMemo } from "react"
import * as THREE from "three"
import { BRAINSTEM, brainRadiusScale } from "../engine/brain-shape"

interface BrainShellProps {
  // Half-extents along X, Y, Z — ellipsoid radii
  radii: readonly [number, number, number]
  color?: string
  rimColor?: string
  opacity?: number
}

/**
 * Fresnel-glow vertex shader — passes world-space view direction + normal
 * so the fragment shader can compute rim intensity.
 */
const VERTEX = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vViewDir;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vViewDir = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }
`

const FRAGMENT = /* glsl */ `
  uniform vec3 uColor;
  uniform vec3 uRimColor;
  uniform float uOpacity;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  void main() {
    // DoubleSide: flip normal for back faces so Fresnel stays correct inside.
    vec3 n = gl_FrontFacing ? vNormal : -vNormal;
    float fresnel = pow(1.0 - clamp(dot(n, vViewDir), 0.0, 1.0), 1.7);
    vec3 col = mix(uColor, uRimColor, fresnel);
    gl_FragColor = vec4(col, uOpacity + fresnel * 0.35);
  }
`

function makeBrainGeometry(radii: readonly [number, number, number]): THREE.BufferGeometry {
  const [rx, ry, rz] = radii
  // Detail 5 (vs 4) — the gyri are fine enough that level 4 aliases them into
  // faceted lumps. This is a one-off static mesh, so the cost is memory only.
  const geom = new THREE.IcosahedronGeometry(1, 5)
  const pos = geom.attributes.position
  const v = new THREE.Vector3()
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i)
    v.normalize()
    const scale = brainRadiusScale(v.x, v.y, v.z)
    pos.setXYZ(i, v.x * rx * scale, v.y * ry * scale, v.z * rz * scale)
  }
  geom.computeVertexNormals()
  return geom
}

function makeBrainstemGeometry(radii: readonly [number, number, number]): THREE.BufferGeometry {
  const [rx, ry, rz] = radii
  const meanR = (rx + ry + rz) / 3
  const geom = new THREE.CylinderGeometry(
    BRAINSTEM.topRadius * meanR,
    BRAINSTEM.bottomRadius * meanR,
    BRAINSTEM.length * ry,
    20,
    1,
    true,
  )
  geom.rotateX(BRAINSTEM.tilt)
  geom.translate(
    BRAINSTEM.offset[0] * rx,
    BRAINSTEM.offset[1] * ry,
    BRAINSTEM.offset[2] * rz,
  )
  return geom
}

export function BrainShell({
  radii,
  // Warm tissue base + the brand teal rim. Opacity is up from 0.12: at that
  // level the old shell was a faint ellipse with nothing to see, whereas the
  // folds and lobes now carry actual shape information.
  color = "#3b2b2b",
  rimColor = "#2dd4bf",
  opacity = 0.19,
}: BrainShellProps) {
  const geometry = useMemo(() => makeBrainGeometry(radii), [radii])
  const stemGeometry = useMemo(() => makeBrainstemGeometry(radii), [radii])
  // Dispose imperative geometry when it changes or the shell unmounts —
  // r3f only auto-disposes JSX-declared primitives.
  useEffect(() => () => geometry.dispose(), [geometry])
  useEffect(() => () => stemGeometry.dispose(), [stemGeometry])

  // Stable uniforms object — mutate `.value` in effects instead of rebuilding
  // the object on every prop change (avoids material churn in r3f reconciler).
  const uniforms = useMemo(
    () => ({
      uColor: { value: new THREE.Color(color) },
      uRimColor: { value: new THREE.Color(rimColor) },
      uOpacity: { value: opacity },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )
  useEffect(() => {
    uniforms.uColor.value.set(color)
  }, [color, uniforms])
  useEffect(() => {
    uniforms.uRimColor.value.set(rimColor)
  }, [rimColor, uniforms])
  useEffect(() => {
    uniforms.uOpacity.value = opacity
  }, [opacity, uniforms])

  // One material instance shared by cerebrum and brainstem so the rim glow is
  // continuous across the two meshes.
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: VERTEX,
        fragmentShader: FRAGMENT,
        uniforms,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    [uniforms],
  )
  useEffect(() => () => material.dispose(), [material])

  return (
    <group>
      <mesh geometry={geometry} material={material} frustumCulled={false} />
      <mesh geometry={stemGeometry} material={material} frustumCulled={false} />
    </group>
  )
}
