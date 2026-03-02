/* ------------------------------------------------------------------ */
/*  Dashboard API response types                                       */
/*  Matches backend Pydantic models in dashboard_api.py + models.py    */
/* ------------------------------------------------------------------ */

// GET /api/dashboard/stats
export interface DashboardStats {
  active_brain: string | null
  total_brains: number
  total_neurons: number
  total_synapses: number
  total_fibers: number
  health_grade: string
  purity_score: number
  brains: BrainSummary[]
}

// GET /api/dashboard/brains
export interface BrainSummary {
  id: string
  name: string
  neuron_count: number
  synapse_count: number
  fiber_count: number
  grade: string
  purity_score: number
  is_active: boolean
}

// POST /api/dashboard/brains/switch
export interface BrainSwitchResponse {
  status: string
  active_brain: string
}

// GET /api/dashboard/health
export interface HealthReport {
  grade: string
  purity_score: number
  connectivity: number
  diversity: number
  freshness: number
  consolidation_ratio: number
  orphan_rate: number
  activation_efficiency: number
  recall_confidence: number
  neuron_count: number
  synapse_count: number
  fiber_count: number
  warnings: HealthWarning[]
  recommendations: string[]
}

export interface HealthWarning {
  severity: "info" | "warning" | "critical"
  code: string
  message: string
  details: string
}

// GET /api/dashboard/timeline
export interface TimelineEntry {
  id: string
  content: string
  neuron_type: string
  created_at: string
  metadata: Record<string, unknown>
}

export interface TimelineResponse {
  entries: TimelineEntry[]
  total: number
}

// GET /api/dashboard/evolution
export interface EvolutionResponse {
  brain: string
  proficiency_level: string
  proficiency_index: number
  maturity_level: number
  plasticity: number
  density: number
  activity_score: number
  semantic_ratio: number
  reinforcement_days: number
  topology_coherence: number
  plasticity_index: number
  knowledge_density: number
  total_neurons: number
  total_synapses: number
  total_fibers: number
  fibers_at_semantic: number
  fibers_at_episodic: number
  stage_distribution: StageDistribution | null
  closest_to_semantic: SemanticProgressItem[]
}

export interface StageDistribution {
  short_term: number
  working: number
  episodic: number
  semantic: number
  total: number
}

export interface SemanticProgressItem {
  fiber_id: string
  stage: string
  days_in_stage: number
  days_required: number
  reinforcement_days: number
  reinforcement_required: number
  progress_pct: number
  next_step: string
}

// GET /api/dashboard/fibers
export interface FiberSummary {
  id: string
  summary: string
  neuron_count: number
}

export interface FiberListResponse {
  fibers: FiberSummary[]
}

// GET /api/dashboard/fiber/:id/diagram
export interface FiberDiagramResponse {
  fiber_id: string
  neurons: DiagramNeuron[]
  synapses: DiagramSynapse[]
}

export interface DiagramNeuron {
  id: string
  content: string
  type: string
  metadata: Record<string, unknown>
}

export interface DiagramSynapse {
  id: string
  source_id: string
  target_id: string
  type: string
  weight: number
  direction: string
}

// GET /api/graph
export interface GraphResponse {
  neurons: GraphNeuron[]
  synapses: GraphSynapse[]
  fibers: GraphFiber[]
  total_neurons: number
  total_synapses: number
  stats: {
    neuron_count: number
    synapse_count: number
    fiber_count: number
  }
}

export interface GraphNeuron {
  id: string
  content: string
  type: string
  metadata: Record<string, unknown>
}

export interface GraphSynapse {
  id: string
  source_id: string
  target_id: string
  type: string
  weight: number
  direction: string
}

export interface GraphFiber {
  id: string
  summary: string
  neuron_count: number
}

// GET /api/dashboard/brain-files
export interface BrainFileInfo {
  name: string
  path: string
  size_bytes: number
  is_active: boolean
}

export interface BrainFilesResponse {
  brains_dir: string
  brains: BrainFileInfo[]
  total_size_bytes: number
}

// GET /health
export interface HealthCheckResponse {
  status: string
  version: string
}

// Telegram (Phase 4)
export interface TelegramStatus {
  configured: boolean
  bot_name: string | null
  bot_username: string | null
  chat_ids: string[]
  backup_on_consolidation: boolean
  error: string | null
}

export interface TelegramTestResponse {
  status: string
  results: { chat_id: string; success: boolean; error?: string }[]
}

export interface TelegramBackupResponse {
  status: string
  brain: string
  size_mb: number
  sent_to: number
  failed: number
  errors?: string[]
}
