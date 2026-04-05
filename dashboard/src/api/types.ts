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
  top_penalties: PenaltyFactor[]
}

export interface HealthWarning {
  severity: "info" | "warning" | "critical"
  code: string
  message: string
  details: string
}

export interface PenaltyFactor {
  component: string
  current_score: number
  weight: number
  penalty_points: number
  estimated_gain: number
  action: string
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

// GET /api/dashboard/timeline/daily-stats
export interface DailyStats {
  date: string
  neurons_created: number
  fibers_created: number
  synapses_created: number
  neuron_types: Record<string, number>
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

// GET /api/dashboard/sync-status
export interface SyncStatusResponse {
  enabled: boolean
  hub_url: string
  api_key: string
  auto_sync: boolean
  conflict_strategy: string
  device_id: string
  change_log?: {
    total_changes: number
    synced_changes: number
    unsynced_changes: number
    latest_sequence: number
  }
  devices: SyncDevice[]
  device_count: number
}

export interface SyncDevice {
  device_id: string
  device_name: string
  last_sync_at: string | null
  last_sync_sequence: number
  registered_at: string
}

// POST /api/dashboard/sync-config
export interface SyncConfigUpdateResponse {
  status: string
  enabled: boolean
  hub_url: string
  api_key: string
  conflict_strategy: string
}

// GET /api/dashboard/storage/status
export interface StorageStatusResponse {
  current_backend: "sqlite" | "infinitydb"
  pro_installed: boolean
  is_pro_license: boolean
  sqlite_exists: boolean
  sqlite_size_bytes: number
  infinitydb_exists: boolean
  migration_job: MigrationJobStatus | null
}

// POST /api/dashboard/storage/migrate + GET /api/dashboard/storage/migrate/{job_id}
export interface MigrationJobStatus {
  job_id: string
  state: "running" | "done" | "error"
  direction: "to_infinitydb" | "to_sqlite"
  brain: string
  neurons_total: number
  neurons_done: number
  synapses_total: number
  synapses_done: number
  fibers_total: number
  fibers_done: number
  error: string | null
  started_at: string
  finished_at: string | null
}

// POST /api/dashboard/storage/backend
export interface SetBackendResponse {
  status: "switched" | "unchanged"
  backend: string
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

// GET /api/dashboard/tool-stats
export interface ToolStatsSummary {
  total_events: number
  success_rate: number
  top_tools: ToolMetric[]
}

export interface ToolMetric {
  tool_name: string
  server_name: string
  count: number
  success_rate: number
  avg_duration_ms: number
}

export interface ToolDailyEntry {
  date: string
  tool_name: string
  count: number
  success_rate: number
  avg_duration_ms: number
}

export interface ToolStatsResponse {
  summary: ToolStatsSummary
  daily: ToolDailyEntry[]
}

// GET /api/dashboard/config-status
export interface ConfigStatusItem {
  key: string
  label: string
  status: "configured" | "not_configured" | "warning" | "info"
  description: string
  command: string
  value: string
}

export interface ConfigStatusResponse {
  items: ConfigStatusItem[]
}

// PUT /api/dashboard/config
export interface EmbeddingConfigUpdate {
  enabled?: boolean
  provider?: string
  model?: string
  similarity_threshold?: number
}

export interface ConfigUpdateRequest {
  embedding?: EmbeddingConfigUpdate
}

export interface ConfigUpdateResponse {
  status: string
  embedding: {
    enabled: boolean
    provider: string
    model: string
    similarity_threshold: number
  }
}

// GET /api/dashboard/watcher/status
export interface WatcherStatusResponse {
  enabled: boolean
  running: boolean
  paths: string[]
  stats: Record<string, number>
  recent: Array<{
    path: string
    action: string
    neurons_created: number
  }>
}

// GET /api/dashboard/config/embedding
export interface EmbeddingConfigResponse {
  enabled: boolean
  provider: string
  model: string
  similarity_threshold: number
}

// POST /api/dashboard/config/embedding/test
export interface EmbeddingTestResponse {
  status: "ok" | "error"
  provider?: string
  dimension?: number
  error?: string
}

// POST /api/dashboard/visualize
export interface VisualizeRequest {
  query: string
  chart_type?: string
  format?: string
  limit?: number
}

export interface VisualizeResponse {
  query: string
  chart_type: string
  title?: string
  data_points_count?: number
  message?: string
  vega_lite?: Record<string, unknown>
  markdown?: string
  ascii?: string
  memories?: Array<{ id: string; content: string; type: string }>
}

// GET /api/dashboard/tier-stats
export interface TierDistribution {
  hot: number
  warm: number
  cold: number
  total: number
}

// GET /api/dashboard/license
export interface LicenseResponse {
  tier: "free" | "pro" | "team"
  is_pro: boolean
  activated_at: string
  expires_at: string
  pro_deps_installed?: boolean
  pro_deps_missing?: string[]
  pro_deps_install_hint?: string
}

// ── Brain Store ────────────────────────────────────────────────

export interface BrainManifest {
  id: string
  name: string
  display_name: string
  description: string
  version: string
  author: string
  license: string
  tags: string[]
  category: string
  created_at: string
  exported_at: string
  nmem_version: string
  stats: { neuron_count: number; synapse_count: number; fiber_count: number }
  size_bytes: number
  size_tier: string
  content_hash: string
  scan_summary: { risk_level: string; finding_count: number; safe: boolean }
  rating_avg: number
  rating_count: number
  download_count: number
}

export interface ScanFinding {
  category: string
  severity: string
  description: string
  location: string
}

// GET /api/dashboard/store/registry
export interface StoreRegistryResponse {
  brains: BrainManifest[]
  total: number
  cached: boolean
}

// GET /api/dashboard/store/registry/preview/:name
export interface BrainPreview {
  manifest: BrainManifest
  sample_neurons: {
    id: string
    type: string
    content: string
    created_at: string
  }[]
  neuron_type_breakdown: Record<string, number>
  top_tags: string[]
  scan_result: {
    safe: boolean
    risk_level: string
    finding_count: number
    findings?: ScanFinding[]
  }
}

// POST /api/dashboard/store/import + /import-remote
export interface StoreImportResponse {
  brain_id: string
  brain_name: string
  neurons_imported: number
  synapses_imported: number
  fibers_imported: number
  scan_result: { safe: boolean; risk_level: string; finding_count: number }
  warnings: string[]
}

// POST /api/dashboard/store/export
export interface StoreExportResponse {
  manifest: Record<string, unknown>
  scan_summary: Record<string, unknown>
  package_size_bytes: number
  size_tier: string
  package?: Record<string, unknown>
}

// POST /api/dashboard/store/rate
export interface StoreRatingResponse {
  brain_package_id: string
  rating_avg: number
  rating_count: number
}
