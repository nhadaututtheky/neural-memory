import { useMemo, useCallback, useState } from "react"
import {
  ReactFlow,
  type Node,
  type Edge,
  type NodeProps,
  type EdgeProps,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import Dagre from "@dagrejs/dagre"
import type { FiberDiagramResponse } from "@/api/types"

/* ------------------------------------------------------------------ */
/*  Color palette                                                      */
/* ------------------------------------------------------------------ */

const TYPE_COLORS: Record<string, string> = {
  concept: "#6366f1",
  entity: "#06b6d4",
  time: "#f59e0b",
  action: "#059669",
  state: "#8b5cf6",
  other: "#a8a29e",
  relation: "#ec4899",
  attribute: "#14b8a6",
  root: "#f97316",
  group: "#64748b",
}

const TYPE_BG: Record<string, string> = {
  concept: "#6366f115",
  entity: "#06b6d415",
  time: "#f59e0b15",
  action: "#05966915",
  state: "#8b5cf615",
  other: "#a8a29e15",
  relation: "#ec489915",
  attribute: "#14b8a615",
  root: "#f9731620",
  group: "#64748b15",
}

/* ------------------------------------------------------------------ */
/*  Synapse type styling                                               */
/* ------------------------------------------------------------------ */

const SYNAPSE_COLORS: Record<string, string> = {
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

const SYNAPSE_LABELS: Record<string, string> = {
  CAUSED_BY: "caused by",
  RELATES_TO: "relates to",
  PART_OF: "part of",
  LEADS_TO: "leads to",
  CONTAINS: "contains",
  DEPENDS_ON: "depends on",
  SIMILAR_TO: "similar to",
  CONTRAST: "contrast",
  RESOLVED_BY: "resolved by",
  TEMPORAL: "temporal",
  SEMANTIC: "semantic",
}

function getSynapseColor(type: string): string {
  return SYNAPSE_COLORS[type] ?? "#94a3b8"
}

function getSynapseLabel(type: string): string {
  return SYNAPSE_LABELS[type] ?? type.toLowerCase().replace(/_/g, " ")
}

/* ------------------------------------------------------------------ */
/*  Custom edge with label                                             */
/* ------------------------------------------------------------------ */

interface SynapseEdgeData {
  synapseType: string
  weight: number
  highlighted: boolean
  [key: string]: unknown
}

function SynapseEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  style,
}: EdgeProps<Edge<SynapseEdgeData>>) {
  const synapseType = data?.synapseType ?? ""
  const weight = data?.weight ?? 0.5
  const highlighted = data?.highlighted ?? false
  const color = getSynapseColor(synapseType)
  const label = getSynapseLabel(synapseType)

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    curvature: 0.3,
  })

  return (
    <>
      {/* Glow effect when highlighted */}
      {highlighted && (
        <BaseEdge
          id={`${id}-glow`}
          path={edgePath}
          style={{
            stroke: color,
            strokeWidth: Math.max(4, weight * 6),
            opacity: 0.2,
            filter: `blur(4px)`,
          }}
        />
      )}
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          ...style,
          stroke: color,
          strokeWidth: highlighted ? Math.max(2.5, weight * 4) : Math.max(1.5, weight * 3),
          opacity: highlighted ? 0.9 : 0.6,
          transition: "opacity 300ms ease, stroke-width 300ms ease",
        }}
      />
      <EdgeLabelRenderer>
        <div
          className="nodrag nopan pointer-events-auto"
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            opacity: highlighted ? 1 : 0.8,
            transition: "opacity 300ms ease",
          }}
        >
          <span
            className="rounded-md px-1.5 py-0.5 text-[9px] font-medium whitespace-nowrap"
            style={{
              backgroundColor: highlighted ? `${color}30` : `${color}18`,
              color,
              border: `1px solid ${color}${highlighted ? "70" : "40"}`,
              boxShadow: highlighted ? `0 0 6px ${color}30` : undefined,
            }}
          >
            {label}
          </span>
        </div>
      </EdgeLabelRenderer>
    </>
  )
}

/* ------------------------------------------------------------------ */
/*  Custom node components                                             */
/* ------------------------------------------------------------------ */

interface MindmapNodeData {
  label: string
  fullContent: string
  neuronType: string
  isGroup: boolean
  count?: number
  connectionCount?: number
  highlighted: boolean
  dimmed: boolean
  [key: string]: unknown
}

type MindmapNode = Node<MindmapNodeData>

function RootNode({ data }: NodeProps<MindmapNode>) {
  const color = TYPE_COLORS.root
  return (
    <div
      className="rounded-xl px-5 py-3 text-center shadow-md transition-opacity duration-300"
      style={{
        background: `linear-gradient(135deg, ${color}20, ${color}40)`,
        border: `2px solid ${color}`,
        minWidth: 120,
        maxWidth: 220,
        opacity: data.dimmed ? 0.25 : 1,
      }}
    >
      <p className="font-display text-xs font-bold truncate" style={{ color }}>
        {data.label}
      </p>
      <Handle type="source" position={Position.Right} className="!bg-transparent !border-0" />
    </div>
  )
}

function GroupNode({ data }: NodeProps<MindmapNode>) {
  const color = TYPE_COLORS[data.neuronType] ?? TYPE_COLORS.group
  return (
    <div
      className="rounded-lg px-4 py-2 shadow-sm transition-opacity duration-300"
      style={{
        background: TYPE_BG[data.neuronType] ?? TYPE_BG.group,
        border: `2px solid ${color}80`,
        minWidth: 100,
        opacity: data.dimmed ? 0.25 : 1,
      }}
    >
      <Handle type="target" position={Position.Left} className="!bg-transparent !border-0" />
      <div className="flex items-center gap-2">
        <div className="size-3 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-sm font-semibold">{data.neuronType}</span>
        {data.count !== undefined && (
          <span
            className="rounded-full px-1.5 py-0.5 font-mono text-[10px] font-bold"
            style={{ backgroundColor: `${color}30`, color }}
          >
            {data.count}
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-transparent !border-0" />
    </div>
  )
}

function LeafNode({ data }: NodeProps<MindmapNode>) {
  const color = TYPE_COLORS[data.neuronType] ?? TYPE_COLORS.other
  const hasConnections = (data.connectionCount ?? 0) > 0
  const highlighted = data.highlighted
  const dimmed = data.dimmed

  return (
    <div
      className="cursor-pointer rounded-lg px-3 py-1.5 transition-all duration-300"
      style={{
        background: highlighted
          ? `${color}25`
          : TYPE_BG[data.neuronType] ?? TYPE_BG.other,
        border: highlighted
          ? `2px solid ${color}`
          : `1.5px solid ${color}${hasConnections ? "90" : "50"}`,
        maxWidth: 260,
        minWidth: 80,
        opacity: dimmed ? 0.2 : 1,
        boxShadow: highlighted
          ? `0 0 12px ${color}40, 0 2px 8px ${color}20`
          : hasConnections
            ? `0 0 8px ${color}25`
            : "0 1px 3px rgba(0,0,0,0.1)",
        transform: highlighted ? "scale(1.04)" : "scale(1)",
      }}
    >
      <Handle type="target" position={Position.Left} className="!bg-transparent !border-0" />
      <Handle type="source" position={Position.Right} className="!bg-transparent !border-0" />
      <p className="text-xs leading-snug" style={{ color: "var(--color-foreground)" }}>
        {data.label}
      </p>
      {hasConnections && (
        <div className="mt-1 flex items-center gap-1">
          <div
            className="size-1.5 rounded-full"
            style={{
              backgroundColor: color,
              boxShadow: highlighted ? `0 0 4px ${color}` : undefined,
            }}
          />
          <span className="text-[9px] text-muted-foreground">
            {data.connectionCount} connection{(data.connectionCount ?? 0) > 1 ? "s" : ""}
          </span>
        </div>
      )}
    </div>
  )
}

const nodeTypes = {
  root: RootNode,
  group: GroupNode,
  leaf: LeafNode,
}

const edgeTypes = {
  synapse: SynapseEdge,
}

/* ------------------------------------------------------------------ */
/*  Dagre layout                                                       */
/* ------------------------------------------------------------------ */

function layoutTree(nodes: Node[], edges: Edge[]): { nodes: Node[]; edges: Edge[] } {
  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
  g.setGraph({
    rankdir: "LR",
    nodesep: 20,
    ranksep: 100,
    edgesep: 14,
  })

  for (const node of nodes) {
    const width = node.type === "root" ? 220 : node.type === "group" ? 140 : 220
    const height = node.type === "root" ? 50 : node.type === "group" ? 40 : 44
    g.setNode(node.id, { width, height })
  }

  for (const edge of edges) {
    g.setEdge(edge.source, edge.target)
  }

  Dagre.layout(g)

  const layoutedNodes = nodes.map((node) => {
    const pos = g.node(node.id)
    const width = node.type === "root" ? 220 : node.type === "group" ? 140 : 220
    const height = node.type === "root" ? 50 : node.type === "group" ? 40 : 44
    return {
      ...node,
      position: {
        x: pos.x - width / 2,
        y: pos.y - height / 2,
      },
    }
  })

  return { nodes: layoutedNodes, edges }
}

/* ------------------------------------------------------------------ */
/*  Adjacency map builder                                              */
/* ------------------------------------------------------------------ */

/** Build adjacency from synapses — maps each neuron to its neighbors and connecting edges */
function buildAdjacency(diagram: FiberDiagramResponse) {
  const neuronIds = new Set(diagram.neurons.map((n) => n.id))
  /** nodeId → Set of neighbor nodeIds */
  const neighbors = new Map<string, Set<string>>()
  /** edgeId → true (for synapse edges that connect two valid neurons) */
  const synapseEdgeIds = new Map<string, { source: string; target: string }>()

  for (const syn of diagram.synapses) {
    if (!neuronIds.has(syn.source_id) || !neuronIds.has(syn.target_id)) continue

    const edgeId = `syn-${syn.id}`
    synapseEdgeIds.set(edgeId, { source: syn.source_id, target: syn.target_id })

    if (!neighbors.has(syn.source_id)) neighbors.set(syn.source_id, new Set())
    if (!neighbors.has(syn.target_id)) neighbors.set(syn.target_id, new Set())
    neighbors.get(syn.source_id)!.add(syn.target_id)
    neighbors.get(syn.target_id)!.add(syn.source_id)
  }

  return { neighbors, synapseEdgeIds }
}

/* ------------------------------------------------------------------ */
/*  Build ReactFlow nodes/edges from FiberDiagramResponse              */
/* ------------------------------------------------------------------ */

function buildFlowData(diagram: FiberDiagramResponse, fiberName?: string): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []

  // Count synapse connections per neuron
  const connectionCounts = new Map<string, number>()
  const neuronIds = new Set(diagram.neurons.map((n) => n.id))
  for (const syn of diagram.synapses) {
    if (neuronIds.has(syn.source_id) && neuronIds.has(syn.target_id)) {
      connectionCounts.set(syn.source_id, (connectionCounts.get(syn.source_id) ?? 0) + 1)
      connectionCounts.set(syn.target_id, (connectionCounts.get(syn.target_id) ?? 0) + 1)
    }
  }

  // Root node
  const rootId = `root-${diagram.fiber_id}`
  nodes.push({
    id: rootId,
    type: "root",
    position: { x: 0, y: 0 },
    data: {
      label: fiberName ?? "Fiber",
      fullContent: fiberName ?? diagram.fiber_id,
      neuronType: "root",
      isGroup: false,
      highlighted: false,
      dimmed: false,
    },
  })

  // Group neurons by type
  const groups = new Map<string, typeof diagram.neurons>()
  for (const neuron of diagram.neurons) {
    const group = groups.get(neuron.type) ?? []
    group.push(neuron)
    groups.set(neuron.type, group)
  }

  // Create group + leaf nodes
  for (const [type, neurons] of groups) {
    const groupId = `group-${type}`
    nodes.push({
      id: groupId,
      type: "group",
      position: { x: 0, y: 0 },
      data: {
        label: type,
        fullContent: "",
        neuronType: type,
        isGroup: true,
        count: neurons.length,
        highlighted: false,
        dimmed: false,
      },
    })
    edges.push({
      id: `e-root-${type}`,
      source: rootId,
      target: groupId,
      style: { stroke: TYPE_COLORS[type] ?? TYPE_COLORS.other, strokeWidth: 2, opacity: 0.5 },
      type: "smoothstep",
    })

    for (const neuron of neurons) {
      const label = neuron.content.length > 80
        ? neuron.content.slice(0, 80) + "..."
        : neuron.content
      nodes.push({
        id: neuron.id,
        type: "leaf",
        position: { x: 0, y: 0 },
        data: {
          label,
          fullContent: neuron.content,
          neuronType: neuron.type,
          isGroup: false,
          connectionCount: connectionCounts.get(neuron.id) ?? 0,
          highlighted: false,
          dimmed: false,
        },
      })
      edges.push({
        id: `e-${groupId}-${neuron.id}`,
        source: groupId,
        target: neuron.id,
        style: {
          stroke: TYPE_COLORS[type] ?? TYPE_COLORS.other,
          strokeWidth: 1.5,
          opacity: 0.35,
        },
        type: "smoothstep",
      })
    }
  }

  // Synapse connections between leaf nodes — labeled with type
  for (const syn of diagram.synapses) {
    if (neuronIds.has(syn.source_id) && neuronIds.has(syn.target_id)) {
      edges.push({
        id: `syn-${syn.id}`,
        source: syn.source_id,
        target: syn.target_id,
        type: "synapse",
        data: {
          synapseType: syn.type,
          weight: syn.weight,
          highlighted: false,
        },
        animated: true,
      })
    }
  }

  return layoutTree(nodes, edges)
}

/* ------------------------------------------------------------------ */
/*  Synapse legend                                                     */
/* ------------------------------------------------------------------ */

function SynapseLegend({ synapseTypes }: { synapseTypes: string[] }) {
  if (synapseTypes.length === 0) return null

  return (
    <div className="absolute bottom-12 left-3 z-10 rounded-lg border border-border bg-card/90 px-3 py-2 backdrop-blur-sm shadow-sm">
      <p className="mb-1.5 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
        Relationships
      </p>
      <div className="flex flex-col gap-1">
        {synapseTypes.map((type) => (
          <div key={type} className="flex items-center gap-2">
            <div
              className="h-0.5 w-4 rounded-full"
              style={{ backgroundColor: getSynapseColor(type) }}
            />
            <span className="text-[10px]" style={{ color: getSynapseColor(type) }}>
              {getSynapseLabel(type)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

interface FiberMindmapProps {
  diagram: FiberDiagramResponse
  fiberName?: string
  onSelectNeuron?: (id: string, content: string, type: string) => void
}

function FiberMindmapInner({ diagram, fiberName, onSelectNeuron }: FiberMindmapProps) {
  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(
    () => buildFlowData(diagram, fiberName),
    [diagram, fiberName],
  )

  const { neighbors, synapseEdgeIds } = useMemo(
    () => buildAdjacency(diagram),
    [diagram],
  )

  // Collect unique synapse types for legend
  const synapseTypes = useMemo(() => {
    const neuronIds = new Set(diagram.neurons.map((n) => n.id))
    const types = new Set<string>()
    for (const syn of diagram.synapses) {
      if (neuronIds.has(syn.source_id) && neuronIds.has(syn.target_id)) {
        types.add(syn.type)
      }
    }
    return Array.from(types).sort()
  }, [diagram])

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges)
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null)

  /** Apply spreading activation highlight on click */
  const applyHighlight = useCallback(
    (clickedId: string | null) => {
      if (!clickedId) {
        // Clear all highlights
        setNodes((prev) =>
          prev.map((n) => ({
            ...n,
            data: { ...n.data, highlighted: false, dimmed: false },
          })),
        )
        setEdges((prev) =>
          prev.map((e) => {
            if (e.type === "synapse") {
              return {
                ...e,
                data: { ...e.data, highlighted: false },
                animated: true,
                style: { ...e.style, opacity: undefined },
              }
            }
            return { ...e, style: { ...e.style, opacity: 0.35 } }
          }),
        )
        return
      }

      const connectedNodes = neighbors.get(clickedId) ?? new Set<string>()
      const highlightSet = new Set([clickedId, ...connectedNodes])

      // Find which synapse edges connect to clicked node
      const highlightedEdgeIds = new Set<string>()
      for (const [edgeId, { source, target }] of synapseEdgeIds) {
        if (source === clickedId || target === clickedId) {
          highlightedEdgeIds.add(edgeId)
        }
      }

      // Also highlight the group node that the clicked leaf belongs to
      const clickedNeuron = diagram.neurons.find((n) => n.id === clickedId)
      if (clickedNeuron) {
        highlightSet.add(`group-${clickedNeuron.type}`)
      }
      // Highlight group nodes for connected neurons
      for (const neighborId of connectedNodes) {
        const neighbor = diagram.neurons.find((n) => n.id === neighborId)
        if (neighbor) highlightSet.add(`group-${neighbor.type}`)
      }

      setNodes((prev) =>
        prev.map((n) => ({
          ...n,
          data: {
            ...n.data,
            highlighted: n.id === clickedId || connectedNodes.has(n.id),
            dimmed: !highlightSet.has(n.id) && n.type !== "root",
          },
        })),
      )

      setEdges((prev) =>
        prev.map((e) => {
          if (e.type === "synapse") {
            const isHighlighted = highlightedEdgeIds.has(e.id)
            return {
              ...e,
              data: { ...e.data, highlighted: isHighlighted },
              animated: isHighlighted,
              style: {
                ...e.style,
                opacity: isHighlighted ? undefined : 0.1,
              },
            }
          }
          // Tree edges (root→group, group→leaf)
          const sourceHighlighted = highlightSet.has(e.source)
          const targetHighlighted = highlightSet.has(e.target)
          return {
            ...e,
            style: {
              ...e.style,
              opacity: sourceHighlighted && targetHighlighted ? 0.6 : 0.08,
            },
          }
        }),
      )
    },
    [neighbors, synapseEdgeIds, diagram, setNodes, setEdges],
  )

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const data = node.data as MindmapNodeData

      if (!data.isGroup && node.type === "leaf") {
        const newId = activeNodeId === node.id ? null : node.id
        setActiveNodeId(newId)
        applyHighlight(newId)
        if (newId) {
          onSelectNeuron?.(node.id, data.fullContent, data.neuronType)
        }
      }
    },
    [onSelectNeuron, activeNodeId, applyHighlight],
  )

  // Click on background to clear
  const onPaneClick = useCallback(() => {
    setActiveNodeId(null)
    applyHighlight(null)
  }, [applyHighlight])

  return (
    <div className="relative h-[calc(100vh-14rem)] min-h-[500px] w-full rounded-lg border border-border bg-background">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="var(--color-border)" gap={20} size={1} />
        <Controls
          showInteractive={false}
          className="!bg-card !border-border !shadow-sm [&>button]:!bg-card [&>button]:!border-border [&>button]:!fill-foreground"
        />
        <MiniMap
          nodeColor={(n) => TYPE_COLORS[(n.data as MindmapNodeData)?.neuronType] ?? "#a8a29e"}
          maskColor="rgba(0,0,0,0.15)"
          className="!bg-card !border-border"
        />
      </ReactFlow>
      <SynapseLegend synapseTypes={synapseTypes} />
    </div>
  )
}

export function FiberMindmap(props: FiberMindmapProps) {
  return (
    <ReactFlowProvider>
      <FiberMindmapInner {...props} />
    </ReactFlowProvider>
  )
}
