/**
 * TypeScript interfaces mirroring the NeuralMemory server Pydantic models.
 *
 * Source: neural_memory/server/models.py
 *         neural_memory/server/routes/sync.py
 *         neural_memory/server/app.py (graph endpoint)
 */

// ============ Request Types ============

export interface EncodeRequest {
  readonly content: string;
  readonly timestamp?: string;
  readonly metadata?: Record<string, unknown>;
  readonly tags?: readonly string[];
}

export interface QueryRequest {
  readonly query: string;
  readonly depth?: number; // 0=instant, 1=context, 2=habit, 3=deep
  readonly max_tokens?: number; // 50-5000, default 500
  readonly include_subgraph?: boolean;
  readonly reference_time?: string;
}

export interface CreateBrainRequest {
  readonly name: string;
  readonly owner_id?: string;
  readonly is_public?: boolean;
  readonly config?: BrainConfigModel;
}

export interface BrainConfigModel {
  readonly decay_rate?: number; // 0-1, default 0.1
  readonly reinforcement_delta?: number; // 0-0.5, default 0.05
  readonly activation_threshold?: number; // 0-1, default 0.2
  readonly max_spread_hops?: number; // 1-10, default 4
  readonly max_context_tokens?: number; // 100-10000, default 1500
}

// ============ Response Types ============

export interface EncodeResponse {
  readonly fiber_id: string;
  readonly neurons_created: number;
  readonly neurons_linked: number;
  readonly synapses_created: number;
}

export interface SubgraphResponse {
  readonly neuron_ids: readonly string[];
  readonly synapse_ids: readonly string[];
  readonly anchor_ids: readonly string[];
}

export interface QueryResponse {
  readonly answer: string | null;
  readonly confidence: number;
  readonly depth_used: number;
  readonly neurons_activated: number;
  readonly fibers_matched: readonly string[];
  readonly context: string;
  readonly latency_ms: number;
  readonly subgraph: SubgraphResponse | null;
  readonly metadata: Record<string, unknown>;
}

export interface BrainResponse {
  readonly id: string;
  readonly name: string;
  readonly owner_id: string | null;
  readonly is_public: boolean;
  readonly neuron_count: number;
  readonly synapse_count: number;
  readonly fiber_count: number;
  readonly created_at: string;
  readonly updated_at: string;
}

export interface BrainListResponse {
  readonly brains: readonly BrainResponse[];
  readonly total: number;
}

export interface StatsResponse {
  readonly brain_id: string;
  readonly neuron_count: number;
  readonly synapse_count: number;
  readonly fiber_count: number;
}

export interface HealthResponse {
  readonly status: string;
  readonly version: string;
}

export interface ErrorResponse {
  readonly error: string;
  readonly detail?: string;
}

// ============ Neuron / Fiber types (from route responses) ============

export type NeuronType =
  | "concept"
  | "entity"
  | "time"
  | "action"
  | "state"
  | "spatial"
  | "sensory"
  | "intent";

export interface NeuronItem {
  readonly id: string;
  readonly type: NeuronType;
  readonly content: string;
  readonly created_at: string;
}

export interface NeuronListResponse {
  readonly neurons: readonly NeuronItem[];
  readonly count: number;
}

export interface FiberResponse {
  readonly id: string;
  readonly neuron_ids: readonly string[];
  readonly synapse_ids: readonly string[];
  readonly anchor_neuron_id: string | null;
  readonly time_start: string | null;
  readonly time_end: string | null;
  readonly coherence: number;
  readonly salience: number;
  readonly frequency: number;
  readonly summary: string | null;
  readonly tags: readonly string[];
  readonly created_at: string;
}

// ============ Index types ============

export interface IndexRequest {
  readonly action: "scan" | "status";
  readonly path?: string;
  readonly extensions?: readonly string[];
}

export interface IndexResponse {
  readonly files_indexed: number;
  readonly neurons_created: number;
  readonly synapses_created: number;
  readonly path: string | null;
  readonly message: string;
  readonly indexed_files: readonly string[] | null;
}

// ============ Graph types (from /api/graph) ============

export interface GraphNeuron {
  readonly id: string;
  readonly type: NeuronType;
  readonly content: string;
  readonly metadata: Record<string, unknown>;
}

export type SynapseDirection = "forward" | "backward" | "bidirectional";

export interface GraphSynapse {
  readonly id: string;
  readonly source_id: string;
  readonly target_id: string;
  readonly type: string;
  readonly weight: number;
  readonly direction: SynapseDirection;
}

export interface GraphFiber {
  readonly id: string;
  readonly summary: string | null;
  readonly neuron_count: number;
}

export interface GraphStats {
  readonly neuron_count: number;
  readonly synapse_count: number;
  readonly fiber_count: number;
}

export interface GraphData {
  readonly neurons: readonly GraphNeuron[];
  readonly synapses: readonly GraphSynapse[];
  readonly fibers: readonly GraphFiber[];
  readonly stats: GraphStats;
}

// ============ Brain export/import ============

export interface BrainSnapshot {
  readonly brain_id: string;
  readonly brain_name: string;
  readonly exported_at: string;
  readonly version: string;
  readonly neurons: readonly Record<string, unknown>[];
  readonly synapses: readonly Record<string, unknown>[];
  readonly fibers: readonly Record<string, unknown>[];
  readonly config: Record<string, unknown>;
  readonly metadata: Record<string, unknown>;
}

export interface DeleteBrainResponse {
  readonly status: "deleted";
  readonly brain_id: string;
}

// ============ WebSocket Sync types (from sync.py) ============

export type SyncEventType =
  // Connection
  | "connected"
  | "disconnected"
  | "subscribed"
  | "unsubscribed"
  // Data events
  | "neuron_created"
  | "neuron_updated"
  | "neuron_deleted"
  | "synapse_created"
  | "synapse_updated"
  | "synapse_deleted"
  | "fiber_created"
  | "fiber_updated"
  | "fiber_deleted"
  // Memory events
  | "memory_encoded"
  | "memory_queried"
  // Sync events
  | "full_sync"
  | "partial_sync"
  // Error
  | "error";

export interface SyncEvent {
  readonly type: SyncEventType;
  readonly brain_id: string;
  readonly timestamp: string;
  readonly data: Record<string, unknown>;
  readonly source_client_id?: string;
}

export interface SyncHistoryResponse {
  readonly type: "history";
  readonly brain_id: string;
  readonly events: readonly SyncEvent[];
}

export interface SyncStatsResponse {
  readonly connected_clients: number;
  readonly brain_subscriptions: Record<string, number>;
  readonly event_history_size: number;
}
