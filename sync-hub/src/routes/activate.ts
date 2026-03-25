/**
 * POST /v1/hub/activate — Activate a purchased license key.
 *
 * User sends their NM license key (nm_pro_xxx or nm_team_xxx).
 * Hub verifies via pay.theio.vn/verify, then upgrades user tier in D1.
 * One-time activation — tier is cached in D1 for fast subsequent checks.
 */

import { Hono } from "hono";
import type { AppEnv, LicenseContext } from "../types.js";
import { handleError, HubError } from "../errors.js";

const activate = new Hono<AppEnv>();

const PAY_VERIFY_URL = "https://pay.theio.vn/verify";

const LICENSE_KEY_PATTERN = /^nm_(pro|team)_[a-zA-Z0-9_-]+$/;
const VALID_TIERS = new Set(["free", "pro", "team"]);

activate.post("/", async (c) => {
  try {
    const body = await c.req.json<{ license_key: string }>();
    const licenseKey = (body.license_key || "").trim();

    if (!licenseKey) {
      throw new HubError(400, "Missing license_key");
    }

    // Normalize XLabs format (NM-PRO-XXXX) → nm_pro_XXXX
    const normalizedKey = licenseKey.startsWith("NM-")
      ? licenseKey.replaceAll("-", "_").toLowerCase()
      : licenseKey;

    // Validate license key format strictly
    if (!LICENSE_KEY_PATTERN.test(normalizedKey)) {
      throw new HubError(400, "Invalid license key format");
    }

    // Verify against pay.theio.vn — send key in POST body, not query string
    const verifyRes = await fetch(PAY_VERIFY_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: normalizedKey }),
    });
    if (!verifyRes.ok) {
      throw new HubError(502, "License verification service unavailable");
    }

    const verifyData = await verifyRes.json<{
      valid: boolean;
      tier?: string;
      expiresAt?: string;
      features?: string[];
      error?: string;
    }>();

    if (!verifyData.valid) {
      throw new HubError(403, "Invalid or expired license key");
    }

    // Validate tier from upstream — never trust external input
    const tier = VALID_TIERS.has(verifyData.tier ?? "") ? verifyData.tier! : "pro";
    const expiresAt = verifyData.expiresAt || null;
    const { userId } = c.get("auth");
    const db = c.env.SYNC_DB;
    const now = new Date().toISOString();

    // Upsert license in D1 (deactivate old ones first)
    await db.batch([
      db
        .prepare("UPDATE licenses SET status = 'replaced' WHERE user_id = ? AND status = 'active'")
        .bind(userId),
      db
        .prepare(
          `INSERT INTO licenses (id, user_id, tier, status, payment_provider, payment_id, created_at, expires_at)
           VALUES (?, ?, ?, 'active', 'pay.theio.vn', ?, ?, ?)`,
        )
        .bind(normalizedKey, userId, tier, normalizedKey, now, expiresAt),
    ]);

    // Update user tier
    await db
      .prepare("UPDATE users SET tier = ? WHERE id = ?")
      .bind(tier, userId)
      .run();

    return c.json({
      status: "activated",
      tier,
      expiresAt,
      features: verifyData.features || [],
      message: `License activated! You now have ${tier} tier access.`,
    });
  } catch (err) {
    return handleError(c, err);
  }
});

export default activate;
