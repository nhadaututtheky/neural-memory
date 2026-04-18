import { useEffect, useMemo } from "react"
import * as THREE from "three"

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
    float fresnel = pow(1.0 - clamp(dot(n, vViewDir), 0.0, 1.0), 2.2);
    vec3 col = mix(uColor, uRimColor, fresnel);
    gl_FragColor = vec4(col, uOpacity + fresnel * 0.35);
  }
`

function makeBrainGeometry(radii: readonly [number, number, number]): THREE.BufferGeometry {
  const [rx, ry, rz] = radii
  const geom = new THREE.IcosahedronGeometry(1, 4)
  const pos = geom.attributes.position
  const v = new THREE.Vector3()
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i)
    v.normalize()
    // Anisotropic brain-ish scaling + subtle lobed perturbation
    const lobe =
      0.06 * Math.sin(4.0 * v.y) * Math.cos(3.0 * v.x) +
      0.04 * Math.sin(5.0 * v.z)
    v.x *= rx * (1 + lobe)
    v.y *= ry * (1 + 0.5 * lobe)
    v.z *= rz * (1 - lobe)
    pos.setXYZ(i, v.x, v.y, v.z)
  }
  geom.computeVertexNormals()
  return geom
}

export function BrainShell({
  radii,
  color = "#1e1b4b",
  rimColor = "#818cf8",
  opacity = 0.12,
}: BrainShellProps) {
  const geometry = useMemo(() => makeBrainGeometry(radii), [radii])
  // Dispose imperative geometry when it changes or the shell unmounts —
  // r3f only auto-disposes JSX-declared primitives.
  useEffect(() => () => geometry.dispose(), [geometry])

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

  return (
    <mesh geometry={geometry} frustumCulled={false}>
      <shaderMaterial
        attach="material"
        vertexShader={VERTEX}
        fragmentShader={FRAGMENT}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}
