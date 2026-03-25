/**
 * GET /v1/hub/license — Current license status for authenticated user.
 *
 * Returns tier, activation date, expiry, and features.
 * Used by XLabs dashboard and NM CLI/Dashboard to show license info.
 */

import { Hono } from "hono";
import type { AppEnv } from "../types.js";
import { handleError } from "../errors.js";

const license = new Hono<AppEnv>();

license.get("/", async (c) => {
  try {
    const { userId } = c.get("auth");
    const db = c.env.SYNC_DB;

    // Get active license
    const row = await db
      .prepare(
        `SELECT id, tier, status, created_at, expires_at
         FROM licenses
         WHERE user_id = ? AND status = 'active'
         ORDER BY created_at DESC
         LIMIT 1`,
      )
      .bind(userId)
      .first<{
        id: string;
        tier: string;
        status: string;
        created_at: string;
        expires_at: string | null;
      }>();

    if (!row) {
      return c.json({
        tier: "free",
        status: "none",
        license_id: null,
        activated_at: null,
        expires_at: null,
        features: [],
        is_pro: false,
      });
    }

    // Check expiry
    const isExpired =
      row.expires_at && new Date(row.expires_at) < new Date();

    if (isExpired) {
      // Fire-and-forget: mark expired
      c.executionCtx.waitUntil(
        db
          .prepare("UPDATE licenses SET status = 'expired' WHERE id = ?")
          .bind(row.id)
          .run(),
      );

      return c.json({
        tier: "free",
        status: "expired",
        license_id: row.id,
        activated_at: row.created_at,
        expires_at: row.expires_at,
        features: [],
        is_pro: false,
      });
    }

    const isPro = row.tier === "pro" || row.tier === "team";

    return c.json({
      tier: row.tier,
      status: "active",
      license_id: row.id,
      activated_at: row.created_at,
      expires_at: row.expires_at,
      features: isPro
        ? ["sync", "infinitydb", "cone_queries", "smart_merge", "directional_compress"]
        : [],
      is_pro: isPro,
    });
  } catch (err) {
    return handleError(c, err);
  }
});

export default license;
