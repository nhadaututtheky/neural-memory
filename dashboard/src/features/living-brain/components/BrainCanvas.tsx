import { Canvas, useThree } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import { Suspense, useEffect, useMemo, useRef } from "react"
import * as THREE from "three"
import { NeuronNodes } from "./NeuronNodes"
import { SynapseEdges } from "./SynapseEdges"
import { BrainShell } from "./BrainShell"
import { DEFAULT_BRAIN_RADII } from "../engine/force-3d"
import type { BrainLayout, PositionedNeuron } from "../engine/types"

const BRAIN_BG = "#050814"
const SHELL_PADDING = 1.08

interface BrainCanvasProps {
  layout: BrainLayout
  onNodeClick?: (neuron: PositionedNeuron) => void
  reducedMotion?: boolean
  showBrainShell?: boolean
  /**
   * When false, AutoRotate is disabled so the demand frameloop stays idle.
   * Defaults to true (respects reducedMotion separately).
   */
  autoRotate?: boolean
  getPulses?: () => ReadonlyMap<string, number>
  pulseDurationMs?: number
  pulseKickTick?: number
  /**
   * Imperative render hook — BrainCanvas registers a `() => gl.render(scene,
   * camera)` callback with the parent so features like "Share PNG" can force
   * a fresh draw before reading the drawing buffer. Essential when
   * `frameloop="demand"` means the buffer may be stale between pulses.
   */
  onRegisterRenderer?: (fn: (() => void) | null) => void
}

function AutoRotate({ enabled, invalidate }: { enabled: boolean; invalidate: () => void }) {
  const raf = useRef<number | null>(null)
  useEffect(() => {
    if (!enabled) return
    let running = true
    const tick = () => {
      if (!running) return
      invalidate()
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => {
      running = false
      if (raf.current !== null) cancelAnimationFrame(raf.current)
    }
  }, [enabled, invalidate])
  return null
}

function ControlsWithInvalidate({ autoRotate }: { autoRotate: boolean }) {
  const invalidate = useThree((s) => s.invalidate)
  return (
    <>
      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        autoRotate={autoRotate}
        autoRotateSpeed={0.35}
        minDistance={20}
        maxDistance={600}
        onChange={invalidate}
      />
      <AutoRotate enabled={autoRotate} invalidate={invalidate} />
    </>
  )
}

/**
 * Frames the camera around the brain bounds on first paint (+20% padding).
 * Runs ONCE per radii identity — a hasRun ref guards against strict-mode
 * double-mount re-firing after OrbitControls has cached its initial pose.
 */
function CameraAutoFit({ radii }: { radii: readonly [number, number, number] }) {
  const camera = useThree((s) => s.camera)
  const invalidate = useThree((s) => s.invalidate)
  const hasRun = useRef(false)
  useEffect(() => {
    if (hasRun.current) return
    hasRun.current = true
    const [rx, ry, rz] = radii
    const maxR = Math.max(rx, ry, rz)
    const perspective = camera as THREE.PerspectiveCamera
    const fovRad = ((perspective.fov ?? 60) * Math.PI) / 180
    const dist = (maxR * 1.2) / Math.tan(fovRad / 2)
    camera.position.set(0, ry * 0.3, dist)
    camera.lookAt(0, 0, 0)
    camera.updateProjectionMatrix()
    invalidate()
  }, [radii, camera, invalidate])
  return null
}

/**
 * Wakes the `frameloop="demand"` loop whenever a new pulse starts so
 * `useFrame` in NeuronNodes can animate the envelope. The pulse code
 * itself calls invalidate() every frame it does work, so this just
 * kickstarts the animation — subsequent frames fire on their own.
 */
function PulseKicker({ tick }: { tick: number }) {
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    if (tick > 0) invalidate()
  }, [tick, invalidate])
  return null
}

/**
 * Exposes an imperative `() => gl.render(scene, camera)` back to the parent.
 * Parent stores it in a ref so e.g. ShareBrain can force a fresh frame into
 * the drawing buffer before `toBlob` reads it.
 */
function RendererRegistrar({
  onRegister,
}: {
  onRegister?: (fn: (() => void) | null) => void
}) {
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)
  const camera = useThree((s) => s.camera)
  useEffect(() => {
    if (!onRegister) return
    onRegister(() => gl.render(scene, camera))
    return () => onRegister(null)
  }, [onRegister, gl, scene, camera])
  return null
}

export function BrainCanvas({
  layout,
  onNodeClick,
  reducedMotion = false,
  showBrainShell = true,
  autoRotate = true,
  getPulses,
  pulseDurationMs,
  pulseKickTick = 0,
  onRegisterRenderer,
}: BrainCanvasProps) {
  // Compute shell radii from the force-layout defaults, padded so the
  // mesh sits just outside the outermost node.
  const radii = useMemo<readonly [number, number, number]>(
    () =>
      [
        DEFAULT_BRAIN_RADII[0] * SHELL_PADDING,
        DEFAULT_BRAIN_RADII[1] * SHELL_PADDING,
        DEFAULT_BRAIN_RADII[2] * SHELL_PADDING,
      ] as const,
    [],
  )

  const rotateEnabled = autoRotate && !reducedMotion

  return (
    <Canvas
      frameloop="demand"
      dpr={reducedMotion ? 1 : [1, 2]}
      camera={{ position: [0, 20, 240], fov: 60, near: 0.1, far: 2000 }}
      gl={{ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true }}
      style={{ background: BRAIN_BG }}
    >
      <color attach="background" args={[BRAIN_BG]} />
      <fog attach="fog" args={[BRAIN_BG, 180, 520]} />

      <ambientLight intensity={0.25} />
      <pointLight position={[60, 80, 120]} intensity={1.2} color="#ffffff" />
      <pointLight position={[-100, -60, -80]} intensity={0.6} color="#7b61ff" />

      <Suspense fallback={null}>
        {showBrainShell && <BrainShell radii={radii} />}
        <NeuronNodes
          layout={layout}
          onNodeClick={onNodeClick}
          getPulses={getPulses}
          pulseDurationMs={pulseDurationMs}
        />
        <SynapseEdges layout={layout} />
      </Suspense>

      <CameraAutoFit radii={radii} />
      <ControlsWithInvalidate autoRotate={rotateEnabled} />
      <PulseKicker tick={pulseKickTick} />
      <RendererRegistrar onRegister={onRegisterRenderer} />
    </Canvas>
  )
}
