import { useMemo, useEffect, useRef } from "react"
import { useThree } from "@react-three/fiber"
import * as THREE from "three"
import type { BrainLayout } from "../engine/types"
import { useLivingBrainStore } from "../stores/livingBrainStore"

interface SynapseEdgesProps {
  layout: BrainLayout
}

const BASE_R = 0x8a / 255
const BASE_G = 0x9e / 255
const BASE_B = 0xd0 / 255
const HIGHLIGHT_R = 0xe2 / 255
const HIGHLIGHT_G = 0xd0 / 255
const HIGHLIGHT_B = 0xff / 255
const DIM_R = 0x2a / 255
const DIM_G = 0x30 / 255
const DIM_B = 0x42 / 255

export function SynapseEdges({ layout }: SynapseEdgesProps) {
  const geomRef = useRef<THREE.BufferGeometry>(null)
  const invalidate = useThree((s) => s.invalidate)
  const selectedId = useLivingBrainStore((s) => s.selectedId)

  const positions = useMemo(() => {
    const buf = new Float32Array(layout.edges.length * 2 * 3)
    for (let i = 0; i < layout.edges.length; i++) {
      const e = layout.edges[i]
      const a = layout.neurons[e.sourceIdx]
      const b = layout.neurons[e.targetIdx]
      const offset = i * 6
      buf[offset] = a.x
      buf[offset + 1] = a.y
      buf[offset + 2] = a.z
      buf[offset + 3] = b.x
      buf[offset + 4] = b.y
      buf[offset + 5] = b.z
    }
    return buf
  }, [layout])

  // Set up position + color buffers once per layout.
  useEffect(() => {
    const g = geomRef.current
    if (!g) return
    const colors = new Float32Array(layout.edges.length * 6)
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3))
    g.attributes.position.needsUpdate = true
    g.computeBoundingSphere()
    invalidate()
  }, [positions, layout, invalidate])

  // Mutate color buffer in place on selection change — reading from the
  // geometry attribute (source of truth), not from a ref or useMemo value.
  useEffect(() => {
    const g = geomRef.current
    if (!g) return
    const colorAttr = g.getAttribute("color") as THREE.BufferAttribute | undefined
    if (!colorAttr) return
    const colors = colorAttr.array as Float32Array

    for (let i = 0; i < layout.edges.length; i++) {
      const e = layout.edges[i]
      const a = layout.neurons[e.sourceIdx]
      const b = layout.neurons[e.targetIdx]
      let r = BASE_R, gc = BASE_G, bc = BASE_B
      if (selectedId) {
        const touches = a.id === selectedId || b.id === selectedId
        if (touches) {
          r = HIGHLIGHT_R
          gc = HIGHLIGHT_G
          bc = HIGHLIGHT_B
        } else {
          r = DIM_R
          gc = DIM_G
          bc = DIM_B
        }
      }
      const off = i * 6
      colors[off] = r
      colors[off + 1] = gc
      colors[off + 2] = bc
      colors[off + 3] = r
      colors[off + 4] = gc
      colors[off + 5] = bc
    }
    colorAttr.needsUpdate = true
    invalidate()
  }, [selectedId, layout, invalidate])

  if (layout.edges.length === 0) return null

  return (
    // key on layout identity so THREE disposes the old BufferGeometry (including
    // its position + color attributes) when the graph data swaps — r3f's
    // reconciler handles the dispose call on unmount.
    <lineSegments key={layout.edges.length} frustumCulled={false}>
      <bufferGeometry ref={geomRef} />
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={selectedId ? 0.55 : 0.22}
        depthWrite={false}
      />
    </lineSegments>
  )
}
