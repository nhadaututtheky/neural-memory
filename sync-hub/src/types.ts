/**
 * TypeScript types matching Python sync protocol exactly.
 * See: src/neural_memory/sync/protocol.py
 */

// --- Enums (as const objects for TS 5.9 erasableSyntaxOnly) ---

export const SyncStatus = {
  SUCCESS: "success",
  PARTIAL: "partial",
  CONFLICT: "conflict",
  ERROR: "error",
} as const;
export type SyncStatus = (typeof SyncStatus)[keyof typeof SyncStatus];

export const ConflictStrategy = {
  PREFER_RECENT: "prefer_recent",
  PREFER_LOCAL: "prefer_local",
  PREFER_REMOTE: "prefer_remote",
  PREFER_STRONGER: "prefer_stronger",
} as const;
export type ConflictStrategy =
  (typeof ConflictStrategy)[keyof typeof ConflictStrategy];

// --- Data structures ---

export interface SyncChange {
  sequence: number;
  entity_type: string; // "neuron" | "synapse" | "fiber"
  entity_id: string;
  operation: string; // "insert" | "update" | "delete"
  device_id: string;
  changed_at: string; // ISO 8601
  payload: Record<string, unknown>;
}

export interface SyncRequest {
  device_id: string;
  brain_id: string;
  last_sequence: number;
  changes: SyncChange[];
  strategy: ConflictStrategy;
}

export interface SyncConflict {
  entity_type: string;
  entity_id: string;
  local_device: string;
  remote_device: string;
  resolution: string;
  details: string;
}

export interface SyncResponse {
  hub_sequence: number;
  changes: SyncChange[];
  conflicts: SyncConflict[];
  status: SyncStatus;
  message: string;
}

export interface RegisterDeviceRequest {
  device_id: string;
  brain_id: string;
  device_name: string;
}

export interface RegisterDeviceResponse {
  device_id: string;
  device_name: string;
  brain_id: string;
  registered_at: string;
  last_sync_sequence: number;
}

export interface HubStatusResponse {
  brain_id: string;
  device_count: number;
  change_log: {
    total_changes: number;
    synced_changes: number;
    unsynced_changes: number;
    latest_sequence: number;
  };
}

export interface DeviceRecord {
  device_id: string;
  device_name: string;
  brain_id: string;
  registered_at: string;
  last_sync_at: string | null;
  last_sync_sequence: number;
}

// --- Team types ---

export const TeamRole = {
  OWNER: "owner",
  ADMIN: "admin",
  EDITOR: "editor",
  VIEWER: "viewer",
} as const;
export type TeamRole = (typeof TeamRole)[keyof typeof TeamRole];

export interface Team {
  id: string;
  name: string;
  owner_id: string;
  max_seats: number;
  created_at: string;
  updated_at: string;
}

export interface TeamMember {
  team_id: string;
  user_id: string;
  role: TeamRole;
  invited_by: string | null;
  joined_at: string;
}

export interface TeamInvite {
  id: string;
  team_id: string;
  email: string;
  role: TeamRole;
  token: string;
  status: string;
  invited_by: string;
  created_at: string;
  expires_at: string;
}

export interface TeamBrain {
  team_id: string;
  brain_id: string;
  added_by: string;
  created_at: string;
}

export interface AuditEntry {
  id: number;
  team_id: string;
  user_id: string;
  action: string;
  brain_id: string | null;
  details: string;
  ip: string | null;
  created_at: string;
}

// --- Auth context (attached by middleware) ---

export interface AuthContext {
  userId: string;
  tier: string;
}

// --- Cloudflare bindings ---

export interface Env {
  SYNC_DB: D1Database;
  XLABS_API_KEY: string;
  GITHUB_BOT_TOKEN: string; // For Brain Store publish (creates PRs)
  // BRAIN_SNAPSHOTS: R2Bucket;  // Phase 1 placeholder
  // RATE_LIMITS: KVNamespace;   // Phase 2
}

// License context (attached by license middleware)
export interface LicenseContext {
  tier: "free" | "pro" | "team";
  licenseId: string | null;
  expiresAt: string | null;
}

// Hono Variables type for c.set/c.get
export interface HonoVariables {
  auth: AuthContext;
  license: LicenseContext;
}

// Combined type for Hono app
export type AppEnv = {
  Bindings: Env;
  Variables: HonoVariables;
};
