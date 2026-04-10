/**
 * D1 prepared statement wrappers for sync hub.
 */

import type {
  SyncChange,
  DeviceRecord,
  Team,
  TeamMember,
  TeamInvite,
  TeamBrain,
  TeamRole,
  AuditEntry,
} from "../types.js";

const MAX_CHANGES_RETURN = 1000;

// --- Brain ---

export async function getOrCreateBrain(
  db: D1Database,
  brainId: string,
): Promise<void> {
  const existing = await db
    .prepare("SELECT id FROM brains WHERE id = ?")
    .bind(brainId)
    .first<{ id: string }>();

  if (!existing) {
    const now = new Date().toISOString();
    await db
      .prepare(
        "INSERT INTO brains (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
      )
      .bind(brainId, brainId, now, now)
      .run();
  }
}

export async function brainExists(
  db: D1Database,
  brainId: string,
): Promise<boolean> {
  const row = await db
    .prepare("SELECT id FROM brains WHERE id = ?")
    .bind(brainId)
    .first<{ id: string }>();
  return row !== null;
}

// --- Device ---

export async function upsertDevice(
  db: D1Database,
  deviceId: string,
  brainId: string,
  deviceName: string,
): Promise<DeviceRecord> {
  const now = new Date().toISOString();

  await db
    .prepare(
      `INSERT INTO devices (device_id, brain_id, device_name, registered_at, last_sync_sequence)
       VALUES (?, ?, ?, ?, 0)
       ON CONFLICT (device_id, brain_id) DO UPDATE SET device_name = excluded.device_name`,
    )
    .bind(deviceId, brainId, deviceName, now)
    .run();

  return getDevice(db, deviceId, brainId);
}

export async function getDevice(
  db: D1Database,
  deviceId: string,
  brainId: string,
): Promise<DeviceRecord> {
  const row = await db
    .prepare(
      "SELECT device_id, brain_id, device_name, registered_at, last_sync_at, last_sync_sequence FROM devices WHERE device_id = ? AND brain_id = ?",
    )
    .bind(deviceId, brainId)
    .first<DeviceRecord>();

  if (!row) {
    throw new Error(`Device ${deviceId} not found for brain ${brainId}`);
  }
  return row;
}

export async function listDevices(
  db: D1Database,
  brainId: string,
): Promise<DeviceRecord[]> {
  const result = await db
    .prepare(
      "SELECT device_id, brain_id, device_name, registered_at, last_sync_at, last_sync_sequence FROM devices WHERE brain_id = ? ORDER BY registered_at ASC",
    )
    .bind(brainId)
    .all<DeviceRecord>();

  return result.results;
}

export async function updateDeviceSync(
  db: D1Database,
  deviceId: string,
  brainId: string,
  sequence: number,
): Promise<void> {
  const now = new Date().toISOString();
  await db
    .prepare(
      "UPDATE devices SET last_sync_at = ?, last_sync_sequence = ? WHERE device_id = ? AND brain_id = ?",
    )
    .bind(now, sequence, deviceId, brainId)
    .run();
}

// --- Change Log ---

export async function insertChanges(
  db: D1Database,
  brainId: string,
  changes: SyncChange[],
): Promise<void> {
  // D1 batch: up to 500 statements per batch
  const stmts = changes.map((c) =>
    db
      .prepare(
        "INSERT INTO change_log (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
      )
      .bind(
        brainId,
        c.entity_type,
        c.entity_id,
        c.operation,
        c.device_id,
        c.changed_at,
        JSON.stringify(c.payload),
      ),
  );

  if (stmts.length > 0) {
    await db.batch(stmts);
  }
}

export async function getChangesSince(
  db: D1Database,
  brainId: string,
  afterSequence: number,
  excludeDeviceId: string,
): Promise<SyncChange[]> {
  const result = await db
    .prepare(
      "SELECT id, entity_type, entity_id, operation, device_id, changed_at, payload FROM change_log WHERE brain_id = ? AND id > ? AND device_id != ? ORDER BY id ASC LIMIT ?",
    )
    .bind(brainId, afterSequence, excludeDeviceId, MAX_CHANGES_RETURN)
    .all<{
      id: number;
      entity_type: string;
      entity_id: string;
      operation: string;
      device_id: string;
      changed_at: string;
      payload: string;
    }>();

  return result.results.map((row) => ({
    sequence: row.id,
    entity_type: row.entity_type,
    entity_id: row.entity_id,
    operation: row.operation,
    device_id: row.device_id,
    changed_at: row.changed_at,
    payload: safeJsonParse(row.payload),
  }));
}

export async function getMaxSequence(
  db: D1Database,
  brainId: string,
): Promise<number> {
  const row = await db
    .prepare("SELECT MAX(id) as max_id FROM change_log WHERE brain_id = ?")
    .bind(brainId)
    .first<{ max_id: number | null }>();

  return row?.max_id ?? 0;
}

export async function getChangeLogStats(
  db: D1Database,
  brainId: string,
): Promise<{
  total_changes: number;
  latest_sequence: number;
}> {
  const row = await db
    .prepare(
      "SELECT COUNT(*) as total, MAX(id) as latest FROM change_log WHERE brain_id = ?",
    )
    .bind(brainId)
    .first<{ total: number; latest: number | null }>();

  return {
    total_changes: row?.total ?? 0,
    latest_sequence: row?.latest ?? 0,
  };
}

// --- Brain Ownership ---

export async function getBrainOwner(
  db: D1Database,
  brainId: string,
): Promise<string | null> {
  const row = await db
    .prepare("SELECT user_id FROM brains WHERE id = ?")
    .bind(brainId)
    .first<{ user_id: string }>();

  return row?.user_id || null;
}

export async function setBrainOwner(
  db: D1Database,
  brainId: string,
  userId: string,
): Promise<void> {
  await db
    .prepare("UPDATE brains SET user_id = ?, updated_at = ? WHERE id = ?")
    .bind(userId, new Date().toISOString(), brainId)
    .run();
}

// --- Teams ---

export async function createTeam(
  db: D1Database,
  id: string,
  name: string,
  ownerId: string,
  maxSeats: number = 5,
): Promise<Team> {
  const now = new Date().toISOString();
  await db.batch([
    db
      .prepare(
        "INSERT INTO teams (id, name, owner_id, max_seats, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
      )
      .bind(id, name, ownerId, maxSeats, now, now),
    db
      .prepare(
        "INSERT INTO team_members (team_id, user_id, role, invited_by, joined_at) VALUES (?, ?, 'owner', NULL, ?)",
      )
      .bind(id, ownerId, now),
  ]);
  return { id, name, owner_id: ownerId, max_seats: maxSeats, created_at: now, updated_at: now };
}

export async function getTeam(
  db: D1Database,
  teamId: string,
): Promise<Team | null> {
  return db
    .prepare("SELECT id, name, owner_id, max_seats, created_at, updated_at FROM teams WHERE id = ?")
    .bind(teamId)
    .first<Team>();
}

export async function listUserTeams(
  db: D1Database,
  userId: string,
): Promise<Array<Team & { role: TeamRole }>> {
  const result = await db
    .prepare(
      `SELECT t.id, t.name, t.owner_id, t.max_seats, t.created_at, t.updated_at, tm.role
       FROM teams t
       JOIN team_members tm ON tm.team_id = t.id
       WHERE tm.user_id = ?
       ORDER BY t.created_at DESC`,
    )
    .bind(userId)
    .all<Team & { role: TeamRole }>();
  return result.results;
}

export async function updateTeam(
  db: D1Database,
  teamId: string,
  updates: { name?: string; max_seats?: number },
): Promise<void> {
  const sets: string[] = ["updated_at = ?"];
  const values: unknown[] = [new Date().toISOString()];
  if (updates.name !== undefined) {
    sets.push("name = ?");
    values.push(updates.name);
  }
  if (updates.max_seats !== undefined) {
    sets.push("max_seats = ?");
    values.push(updates.max_seats);
  }
  values.push(teamId);
  await db
    .prepare(`UPDATE teams SET ${sets.join(", ")} WHERE id = ?`)
    .bind(...values)
    .run();
}

export async function deleteTeam(
  db: D1Database,
  teamId: string,
): Promise<void> {
  await db.prepare("DELETE FROM teams WHERE id = ?").bind(teamId).run();
}

// --- Team Members ---

export async function getMember(
  db: D1Database,
  teamId: string,
  userId: string,
): Promise<TeamMember | null> {
  return db
    .prepare("SELECT team_id, user_id, role, invited_by, joined_at FROM team_members WHERE team_id = ? AND user_id = ?")
    .bind(teamId, userId)
    .first<TeamMember>();
}

export async function listMembers(
  db: D1Database,
  teamId: string,
): Promise<TeamMember[]> {
  const result = await db
    .prepare("SELECT team_id, user_id, role, invited_by, joined_at FROM team_members WHERE team_id = ? ORDER BY joined_at ASC")
    .bind(teamId)
    .all<TeamMember>();
  return result.results;
}

export async function countMembers(
  db: D1Database,
  teamId: string,
): Promise<number> {
  const row = await db
    .prepare("SELECT COUNT(*) as cnt FROM team_members WHERE team_id = ?")
    .bind(teamId)
    .first<{ cnt: number }>();
  return row?.cnt ?? 0;
}

export async function addMember(
  db: D1Database,
  teamId: string,
  userId: string,
  role: TeamRole,
  invitedBy: string,
): Promise<void> {
  await db
    .prepare("INSERT INTO team_members (team_id, user_id, role, invited_by, joined_at) VALUES (?, ?, ?, ?, ?)")
    .bind(teamId, userId, role, invitedBy, new Date().toISOString())
    .run();
}

export async function updateMemberRole(
  db: D1Database,
  teamId: string,
  userId: string,
  role: TeamRole,
): Promise<void> {
  await db
    .prepare("UPDATE team_members SET role = ? WHERE team_id = ? AND user_id = ?")
    .bind(role, teamId, userId)
    .run();
}

export async function removeMember(
  db: D1Database,
  teamId: string,
  userId: string,
): Promise<void> {
  await db
    .prepare("DELETE FROM team_members WHERE team_id = ? AND user_id = ?")
    .bind(teamId, userId)
    .run();
}

// --- Team Invites ---

export async function createInvite(
  db: D1Database,
  invite: {
    id: string;
    team_id: string;
    email: string;
    role: TeamRole;
    token: string;
    invited_by: string;
    expires_at: string;
  },
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO team_invites (id, team_id, email, role, token, status, invited_by, created_at, expires_at)
       VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)`,
    )
    .bind(
      invite.id,
      invite.team_id,
      invite.email,
      invite.role,
      invite.token,
      invite.invited_by,
      new Date().toISOString(),
      invite.expires_at,
    )
    .run();
}

export async function getInviteByToken(
  db: D1Database,
  token: string,
): Promise<TeamInvite | null> {
  return db
    .prepare("SELECT id, team_id, email, role, token, status, invited_by, created_at, expires_at FROM team_invites WHERE token = ?")
    .bind(token)
    .first<TeamInvite>();
}

export async function listTeamInvites(
  db: D1Database,
  teamId: string,
): Promise<TeamInvite[]> {
  const result = await db
    .prepare("SELECT id, team_id, email, role, token, status, invited_by, created_at, expires_at FROM team_invites WHERE team_id = ? AND status = 'pending' ORDER BY created_at DESC")
    .bind(teamId)
    .all<TeamInvite>();
  return result.results;
}

export async function updateInviteStatus(
  db: D1Database,
  inviteId: string,
  status: string,
): Promise<void> {
  await db
    .prepare("UPDATE team_invites SET status = ? WHERE id = ?")
    .bind(status, inviteId)
    .run();
}

// --- Team Brains ---

export async function assignBrainToTeam(
  db: D1Database,
  teamId: string,
  brainId: string,
  addedBy: string,
): Promise<void> {
  await db
    .prepare(
      "INSERT OR IGNORE INTO team_brains (team_id, brain_id, added_by, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(teamId, brainId, addedBy, new Date().toISOString())
    .run();
}

export async function listTeamBrains(
  db: D1Database,
  teamId: string,
): Promise<TeamBrain[]> {
  const result = await db
    .prepare("SELECT team_id, brain_id, added_by, created_at FROM team_brains WHERE team_id = ? ORDER BY created_at ASC")
    .bind(teamId)
    .all<TeamBrain>();
  return result.results;
}

export async function removeTeamBrain(
  db: D1Database,
  teamId: string,
  brainId: string,
): Promise<void> {
  await db
    .prepare("DELETE FROM team_brains WHERE team_id = ? AND brain_id = ?")
    .bind(teamId, brainId)
    .run();
}

export async function getUserTeamRoleForBrain(
  db: D1Database,
  userId: string,
  brainId: string,
): Promise<TeamRole | null> {
  const row = await db
    .prepare(
      `SELECT tm.role FROM team_members tm
       JOIN team_brains tb ON tb.team_id = tm.team_id
       WHERE tm.user_id = ? AND tb.brain_id = ?
       ORDER BY CASE tm.role
         WHEN 'owner' THEN 1 WHEN 'admin' THEN 2 WHEN 'editor' THEN 3 WHEN 'viewer' THEN 4
       END ASC
       LIMIT 1`,
    )
    .bind(userId, brainId)
    .first<{ role: TeamRole }>();
  return row?.role ?? null;
}

// --- Audit Log ---

export async function logAudit(
  db: D1Database,
  entry: {
    team_id: string;
    user_id: string;
    action: string;
    brain_id?: string;
    details?: Record<string, unknown>;
    ip?: string;
  },
): Promise<void> {
  await db
    .prepare(
      "INSERT INTO audit_log (team_id, user_id, action, brain_id, details, ip, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(
      entry.team_id,
      entry.user_id,
      entry.action,
      entry.brain_id ?? null,
      JSON.stringify(entry.details ?? {}),
      entry.ip ?? null,
      new Date().toISOString(),
    )
    .run();
}

export async function getAuditLog(
  db: D1Database,
  teamId: string,
  opts: { limit?: number; offset?: number; action?: string; brain_id?: string },
): Promise<{ entries: AuditEntry[]; total: number }> {
  const limit = Math.min(opts.limit ?? 50, 200);
  const offset = opts.offset ?? 0;

  let where = "WHERE team_id = ?";
  const params: unknown[] = [teamId];
  if (opts.action) {
    where += " AND action = ?";
    params.push(opts.action);
  }
  if (opts.brain_id) {
    where += " AND brain_id = ?";
    params.push(opts.brain_id);
  }

  const countRow = await db
    .prepare(`SELECT COUNT(*) as cnt FROM audit_log ${where}`)
    .bind(...params)
    .first<{ cnt: number }>();

  const result = await db
    .prepare(
      `SELECT id, team_id, user_id, action, brain_id, details, ip, created_at FROM audit_log ${where} ORDER BY created_at DESC LIMIT ? OFFSET ?`,
    )
    .bind(...params, limit, offset)
    .all<AuditEntry>();

  return { entries: result.results, total: countRow?.cnt ?? 0 };
}

export async function pruneAuditLog(
  db: D1Database,
  teamId: string,
  maxRows: number = 10000,
): Promise<number> {
  const countRow = await db
    .prepare("SELECT COUNT(*) as cnt FROM audit_log WHERE team_id = ?")
    .bind(teamId)
    .first<{ cnt: number }>();

  const total = countRow?.cnt ?? 0;
  if (total <= maxRows) return 0;

  const toDelete = total - maxRows;
  await db
    .prepare(
      `DELETE FROM audit_log WHERE id IN (
        SELECT id FROM audit_log WHERE team_id = ? ORDER BY created_at ASC LIMIT ?
      )`,
    )
    .bind(teamId, toDelete)
    .run();

  return toDelete;
}

// --- Seat Limits ---

const TIER_SEAT_LIMITS: Record<string, number> = {
  free: 0,
  pro: 3,
  team: 10,
};

export function getMaxSeatsForTier(tier: string): number {
  return TIER_SEAT_LIMITS[tier] ?? 0;
}

// --- Helpers ---

function safeJsonParse(str: string): Record<string, unknown> {
  try {
    return JSON.parse(str) as Record<string, unknown>;
  } catch {
    return {};
  }
}
