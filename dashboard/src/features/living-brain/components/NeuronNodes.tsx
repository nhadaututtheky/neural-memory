import { useRef, useEffect, useMemo } from "react"
import { useThree, useFrame } from "@react-three/fiber"
import * as THREE from "three"
import type { BrainLayout, PositionedNeuron } from "../engine/types"
import { useLivingBrainStore } from "../stores/livingBrainStore"

interface NeuronNodesProps {
  layout: BrainLayout
  onNodeClick?: (neuron: PositionedNeuron) => void
  /** Read-only pulse map (id → startMs). Consumed per-frame. */
  getPulses?: () => ReadonlyMap<string, number>
  pulseDurationMs?: number
}

const DIM_COLOR = new THREE.Color("#1b1f33")
const WHITE_COLOR = new THREE.Color("#ffffff")
const DIM_FACTOR = 0.22
const HOVER_LERP = 0.22
const SELECT_LERP = 0.35
const HOVER_SCALE = 1.35
const SELECT_SCALE = 1.5
const PULSE_SCALE_GAIN = 0.4 // +40% peak
const PULSE_COLOR_GAIN = 0.5 // lerp 50% toward white at peak

const dummyObject = new THREE.Object3D()
const tmpColor = new THREE.Color()

function InstancedNeurons({
  layout,
  onNodeClick,
  getPulses,
  pulseDurationMs = 900,
}: {
  layout: BrainLayout
  onNodeClick?: (neuron: PositionedNeuron) => void
  getPulses?: () => ReadonlyMap<string, number>
  pulseDurationMs?: number
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const invalidate = useThree((s) => s.invalidate)
  const { neurons } = layout
  const count = neurons.length

  const hoveredId = useLivingBrainStore((s) => s.hoveredId)
  const selectedId = useLivingBrainStore((s) => s.selectedId)
  const setHovered = useLivingBrainStore((s) => s.setHovered)
  const setSelected = useLivingBrainStore((s) => s.setSelected)

  // Mirror state into refs so the useFrame pulse tick doesn't rely on React closures.
  const hoveredRef = useRef(hoveredId)
  const selectedRef = useRef(selectedId)
  useEffect(() => {
    hoveredRef.current = hoveredId
  }, [hoveredId])
  useEffect(() => {
    selectedRef.current = selectedId
  }, [selectedId])

  const baseColors = useMemo(
    () => neurons.map((n) => new THREE.Color(n.color)),
    [neurons],
  )

  const writeInstance = (index: number, scaleMul: number, colorLerp: number) => {
    const mesh = meshRef.current
    if (!mesh) return
    const n = neurons[index]
    let baseScale = n.radius
    const sel = selectedRef.current
    const hov = hoveredRef.current
    if (n.id === sel) baseScale *= SELECT_SCALE
    else if (n.id === hov) baseScale *= HOVER_SCALE
    dummyObject.position.set(n.x, n.y, n.z)
    dummyObject.scale.setScalar(baseScale * (1 + scaleMul))
    dummyObject.updateMatrix()
    mesh.setMatrixAt(index, dummyObject.matrix)
    tmpColor.copy(baseColors[index])
    const neighborSet = sel
      ? new Set([sel, ...(layout.neighbors.get(sel) ?? [])])
      : null
    if (neighborSet && !neighborSet.has(n.id)) {
      tmpColor.lerp(DIM_COLOR, 1 - DIM_FACTOR)
    } else if (n.id === sel) {
      tmpColor.lerp(WHITE_COLOR, SELECT_LERP)
    } else if (n.id === hov) {
      tmpColor.lerp(WHITE_COLOR, HOVER_LERP)
    }
    if (colorLerp > 0) tmpColor.lerp(WHITE_COLOR, colorLerp)
    mesh.setColorAt(index, tmpColor)
  }

  // Initial write of matrix + color — runs once per layout.
  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    for (let i = 0; i < neurons.length; i++) writeInstance(i, 0, 0)
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    invalidate()
    // writeInstance is stable relative to neurons/baseColors; depending only on layout identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [neurons, baseColors, invalidate])

  // Reactive rewrite on hover/select change. Mutates via writeInstance so pulse math stays in one place.
  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    for (let i = 0; i < neurons.length; i++) writeInstance(i, 0, 0)
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    invalidate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hoveredId, selectedId, layout.neighbors, invalidate])

  // Per-frame pulse envelope. Writes ONLY pulsed indices + restores freshly-expired ones.
  const pendingRestoreRef = useRef<Set<number>>(new Set())
  useFrame(() => {
    if (!getPulses) return
    const mesh = meshRef.current
    if (!mesh) return
    const pulses = getPulses()
    const pending = pendingRestoreRef.current
    if (pulses.size === 0 && pending.size === 0) return

    const now = performance.now()
    let changed = false

    if (pulses.size > 0) {
      for (const [id, start] of pulses) {
        const t = (now - start) / pulseDurationMs
        const idx = layout.indexById.get(id)
        if (idx === undefined) continue
        if (t >= 1) {
          pending.add(idx)
          continue
        }
        const env = Math.sin(Math.max(0, t) * Math.PI) // 0 → 1 → 0
        writeInstance(idx, env * PULSE_SCALE_GAIN, env * PULSE_COLOR_GAIN)
        changed = true
      }
    }

    if (pending.size > 0) {
      for (const idx of pending) writeInstance(idx, 0, 0)
      pending.clear()
      changed = true
    }

    if (changed) {
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
      invalidate()
    }
  })

  const handleClick = (e: {
    instanceId?: number
    stopPropagation?: () => void
  }) => {
    if (e.instanceId === undefined) return
    e.stopPropagation?.()
    const n = neurons[e.instanceId]
    if (!n) return
    setSelected(n.id)
    onNodeClick?.(n)
  }

  const handlePointerMove = (e: {
    instanceId?: number
    stopPropagation?: () => void
  }) => {
    if (e.instanceId === undefined) return
    const n = neurons[e.instanceId]
    if (n && n.id !== hoveredId) setHovered(n.id)
  }

  const handlePointerOut = () => {
    if (hoveredId !== null) setHovered(null)
  }

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, count]}
      onClick={handleClick}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
      frustumCulled={false}
    >
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial
        roughness={0.3}
        metalness={0}
        toneMapped={false}
      />
    </instancedMesh>
  )
}

export function NeuronNodes({
  layout,
  onNodeClick,
  getPulses,
  pulseDurationMs,
}: NeuronNodesProps) {
  if (layout.neurons.length === 0) return null
  return (
    <InstancedNeurons
      key={layout.neurons.length}
      layout={layout}
      onNodeClick={onNodeClick}
      getPulses={getPulses}
      pulseDurationMs={pulseDurationMs}
    />
  )
}
