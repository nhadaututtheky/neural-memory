/**
 * Team Sharing routes — CRUD, members, invites, brains, audit.
 */

import { Hono } from "hono";
import type { AppEnv, TeamRole } from "../types.js";
import { HubError } from "../errors.js";
import {
  createTeam,
  getTeam,
  listUserTeams,
  updateTeam,
  deleteTeam,
  getMember,
  listMembers,
  countMembers,
  addMember,
  updateMemberRole,
  removeMember,
  createInvite,
  getInviteByToken,
  listTeamInvites,
  updateInviteStatus,
  assignBrainToTeam,
  listTeamBrains,
  removeTeamBrain,
  logAudit,
  getAuditLog,
  pruneAuditLog,
  getMaxSeatsForTier,
} from "../db/queries.js";

const teams = new Hono<AppEnv>();

// --- Helpers ---

function generateId(): string {
  return `tm_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function generateToken(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function getClientIp(c: { req: { header: (name: string) => string | undefined } }): string {
  return c.req.header("CF-Connecting-IP") ?? c.req.header("X-Forwarded-For") ?? "unknown";
}

type RoleLevel = "owner" | "admin" | "editor" | "viewer";
const ROLE_RANK: Record<RoleLevel, number> = { owner: 1, admin: 2, editor: 3, viewer: 4 };

function hasRole(actual: string, required: RoleLevel): boolean {
  return (ROLE_RANK[actual as RoleLevel] ?? 99) <= ROLE_RANK[required];
}

async function requireTeamRole(
  db: D1Database,
  teamId: string,
  userId: string,
  minRole: RoleLevel,
): Promise<{ role: TeamRole }> {
  const member = await getMember(db, teamId, userId);
  if (!member) {
    throw new HubError(403, "Not a team member");
  }
  if (!hasRole(member.role, minRole)) {
    throw new HubError(403, `Requires ${minRole} role or higher`);
  }
  return { role: member.role };
}

// ── Team CRUD ─────────────────────────────────────────────────────────────

// POST / — Create team
teams.post("/", async (c) => {
  const auth = c.get("auth");
  const license = c.get("license");

  // Only team tier can create teams (pro gets 3 seats, team gets 10)
  const maxSeats = getMaxSeatsForTier(license.tier);
  if (maxSeats === 0) {
    throw new HubError(403, "Team sharing requires Pro or Team tier");
  }

  const body = await c.req.json<{ name: string }>();
  if (!body.name || body.name.trim().length === 0) {
    throw new HubError(400, "Team name is required");
  }
  if (body.name.length > 100) {
    throw new HubError(400, "Team name too long (max 100)");
  }

  const id = generateId();
  const team = await createTeam(c.env.SYNC_DB, id, body.name.trim(), auth.userId, maxSeats);

  await logAudit(c.env.SYNC_DB, {
    team_id: id,
    user_id: auth.userId,
    action: "team.create",
    details: { name: team.name },
    ip: getClientIp(c),
  });

  return c.json({ team }, 201);
});

// GET / — List user's teams
teams.get("/", async (c) => {
  const auth = c.get("auth");
  const userTeams = await listUserTeams(c.env.SYNC_DB, auth.userId);
  return c.json({ teams: userTeams });
});

// GET /:id — Get team details
teams.get("/:id", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "viewer");

  const team = await getTeam(db, teamId);
  if (!team) throw new HubError(404, "Team not found");

  const members = await listMembers(db, teamId);
  const brains = await listTeamBrains(db, teamId);
  const invites = await listTeamInvites(db, teamId);

  return c.json({ team, members, brains, pending_invites: invites.length });
});

// PATCH /:id — Update team
teams.patch("/:id", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "admin");

  const body = await c.req.json<{ name?: string }>();
  if (body.name !== undefined) {
    if (body.name.trim().length === 0) throw new HubError(400, "Name cannot be empty");
    if (body.name.length > 100) throw new HubError(400, "Name too long");
  }

  await updateTeam(db, teamId, { name: body.name?.trim() });

  await logAudit(db, {
    team_id: teamId,
    user_id: auth.userId,
    action: "team.update",
    details: body,
    ip: getClientIp(c),
  });

  return c.json({ success: true });
});

// DELETE /:id — Delete team (owner only)
teams.delete("/:id", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "owner");

  await deleteTeam(db, teamId);
  return c.json({ success: true });
});

// ── Members ───────────────────────────────────────────────────────────────

// GET /:id/members
teams.get("/:id/members", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "viewer");
  const members = await listMembers(db, teamId);
  return c.json({ members });
});

// PATCH /:id/members/:userId — Change role
teams.patch("/:id/members/:userId", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const targetUserId = c.req.param("userId");
  const db = c.env.SYNC_DB;

  const { role: callerRole } = await requireTeamRole(db, teamId, auth.userId, "admin");

  const target = await getMember(db, teamId, targetUserId);
  if (!target) throw new HubError(404, "Member not found");

  // Cannot modify owner
  if (target.role === "owner") throw new HubError(403, "Cannot modify owner");
  // Admin cannot modify other admins (only owner can)
  if (target.role === "admin" && callerRole !== "owner") {
    throw new HubError(403, "Only owner can modify admin roles");
  }

  const body = await c.req.json<{ role: TeamRole }>();
  const newRole = body.role;
  if (!["admin", "editor", "viewer"].includes(newRole)) {
    throw new HubError(400, "Invalid role");
  }

  await updateMemberRole(db, teamId, targetUserId, newRole);

  await logAudit(db, {
    team_id: teamId,
    user_id: auth.userId,
    action: "member.role_change",
    details: { target_user: targetUserId, old_role: target.role, new_role: newRole },
    ip: getClientIp(c),
  });

  return c.json({ success: true });
});

// DELETE /:id/members/:userId — Remove member
teams.delete("/:id/members/:userId", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const targetUserId = c.req.param("userId");
  const db = c.env.SYNC_DB;

  const { role: callerRole } = await requireTeamRole(db, teamId, auth.userId, "admin");

  const target = await getMember(db, teamId, targetUserId);
  if (!target) throw new HubError(404, "Member not found");

  if (target.role === "owner") throw new HubError(403, "Cannot remove owner");
  if (target.role === "admin" && callerRole !== "owner") {
    throw new HubError(403, "Only owner can remove admins");
  }

  await removeMember(db, teamId, targetUserId);

  await logAudit(db, {
    team_id: teamId,
    user_id: auth.userId,
    action: "member.remove",
    details: { target_user: targetUserId, role: target.role },
    ip: getClientIp(c),
  });

  return c.json({ success: true });
});

// ── Invites ───────────────────────────────────────────────────────────────

// POST /:id/invite — Send invite
teams.post("/:id/invite", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "admin");

  const body = await c.req.json<{ email: string; role?: TeamRole }>();
  if (!body.email || !body.email.includes("@")) {
    throw new HubError(400, "Valid email is required");
  }

  const role = body.role ?? "editor";
  if (!["admin", "editor", "viewer"].includes(role)) {
    throw new HubError(400, "Invalid role for invite");
  }

  // Check seat limit
  const team = await getTeam(db, teamId);
  if (!team) throw new HubError(404, "Team not found");

  const memberCount = await countMembers(db, teamId);
  if (memberCount >= team.max_seats) {
    throw new HubError(403, `Team seat limit reached (${team.max_seats})`);
  }

  const token = generateToken();
  const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();

  await createInvite(db, {
    id: generateId(),
    team_id: teamId,
    email: body.email.toLowerCase().trim(),
    role,
    token,
    invited_by: auth.userId,
    expires_at: expiresAt,
  });

  await logAudit(db, {
    team_id: teamId,
    user_id: auth.userId,
    action: "invite.create",
    details: { email: body.email, role },
    ip: getClientIp(c),
  });

  return c.json({ token, expires_at: expiresAt }, 201);
});

// GET /:id/invites — List pending invites
teams.get("/:id/invites", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "admin");
  const invites = await listTeamInvites(db, teamId);
  return c.json({ invites });
});

// DELETE /:id/invites/:inviteId — Revoke invite
teams.delete("/:id/invites/:inviteId", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const inviteId = c.req.param("inviteId");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "admin");
  await updateInviteStatus(db, inviteId, "revoked");

  await logAudit(db, {
    team_id: teamId,
    user_id: auth.userId,
    action: "invite.revoke",
    details: { invite_id: inviteId },
    ip: getClientIp(c),
  });

  return c.json({ success: true });
});

// ── Accept Invite (PUBLIC — token is auth) ────────────────────────────────

// POST /accept — Accept invite via token (no auth required — used separately)
teams.post("/accept", async (c) => {
  const body = await c.req.json<{ token: string }>();
  if (!body.token) throw new HubError(400, "Token is required");

  const db = c.env.SYNC_DB;
  const invite = await getInviteByToken(db, body.token);

  if (!invite) throw new HubError(404, "Invalid invite token");
  if (invite.status !== "pending") throw new HubError(410, "Invite already used or revoked");

  if (new Date(invite.expires_at) < new Date()) {
    await updateInviteStatus(db, invite.id, "expired");
    throw new HubError(410, "Invite has expired");
  }

  // Accept requires auth (user must be logged in to join)
  const auth = c.get("auth");

  // Check if already a member
  const existing = await getMember(db, invite.team_id, auth.userId);
  if (existing) throw new HubError(409, "Already a team member");

  // Check seat limit
  const team = await getTeam(db, invite.team_id);
  if (!team) throw new HubError(404, "Team not found");

  const memberCount = await countMembers(db, invite.team_id);
  if (memberCount >= team.max_seats) {
    throw new HubError(403, "Team seat limit reached");
  }

  await addMember(db, invite.team_id, auth.userId, invite.role as TeamRole, invite.invited_by);
  await updateInviteStatus(db, invite.id, "accepted");

  await logAudit(db, {
    team_id: invite.team_id,
    user_id: auth.userId,
    action: "invite.accept",
    details: { role: invite.role, email: invite.email },
    ip: getClientIp(c),
  });

  return c.json({ team_id: invite.team_id, role: invite.role });
});

// ── Team Brains ───────────────────────────────────────────────────────────

// POST /:id/brains — Assign brain to team
teams.post("/:id/brains", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "admin");

  const body = await c.req.json<{ brain_id: string }>();
  if (!body.brain_id) throw new HubError(400, "brain_id is required");

  await assignBrainToTeam(db, teamId, body.brain_id, auth.userId);

  await logAudit(db, {
    team_id: teamId,
    user_id: auth.userId,
    action: "brain.assign",
    brain_id: body.brain_id,
    ip: getClientIp(c),
  });

  return c.json({ success: true }, 201);
});

// GET /:id/brains — List team brains
teams.get("/:id/brains", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "viewer");
  const brains = await listTeamBrains(db, teamId);
  return c.json({ brains });
});

// DELETE /:id/brains/:brainId — Remove brain from team
teams.delete("/:id/brains/:brainId", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const brainId = c.req.param("brainId");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "admin");
  await removeTeamBrain(db, teamId, brainId);

  await logAudit(db, {
    team_id: teamId,
    user_id: auth.userId,
    action: "brain.remove",
    brain_id: brainId,
    ip: getClientIp(c),
  });

  return c.json({ success: true });
});

// ── Audit Log ─────────────────────────────────────────────────────────────

// GET /:id/audit — Paginated audit log
teams.get("/:id/audit", async (c) => {
  const auth = c.get("auth");
  const teamId = c.req.param("id");
  const db = c.env.SYNC_DB;

  await requireTeamRole(db, teamId, auth.userId, "editor");

  const limit = Math.min(parseInt(c.req.query("limit") ?? "50", 10), 200);
  const offset = parseInt(c.req.query("offset") ?? "0", 10);
  const action = c.req.query("action");
  const brainId = c.req.query("brain_id");

  const result = await getAuditLog(db, teamId, { limit, offset, action, brain_id: brainId });

  // Auto-prune in background if needed
  c.executionCtx.waitUntil(pruneAuditLog(db, teamId));

  return c.json(result);
});

export default teams;
