/**
 * POST /v1/hub/activate — Activate a purchased license key.
 *
 * User sends their NM license key (NM-PRO-xxx or nm_pro_xxx).
 * Hub verifies via XLabs license API, then upgrades user tier in D1.
 * One-time activation — tier is cached in D1 for fast subsequent checks.
 */

import { Hono } from "hono";
import type { AppEnv } from "../types.js";
import { handleError, HubError } from "../errors.js";

const activate = new Hono<AppEnv>();

const XLABS_API = "https://admin.theio.vn/api/licenses";

const LICENSE_KEY_PATTERN = /^nm_(pro|team)_[a-zA-Z0-9_-]+$/;
const XLABS_KEY_PATTERN = /^NM-(PRO|TEAM)-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$/;
const VALID_TIERS = new Set(["free", "pro", "team"]);

activate.post("/", async (c) => {
  try {
    const body = await c.req.json<{ license_key: string }>();
    const licenseKey = (body.license_key || "").trim();

    if (!licenseKey) {
      throw new HubError(400, "Missing license_key");
    }

    // Accept both formats: NM-PRO-XXXX-XXXX-XXXX or nm_pro_xxxx_xxxx_xxxx
    const isXLabsFormat = XLABS_KEY_PATTERN.test(licenseKey);
    const isNmFormat = LICENSE_KEY_PATTERN.test(licenseKey);

    if (!isXLabsFormat && !isNmFormat) {
      throw new HubError(400, "Invalid license key format");
    }

    // Normalize to nm_ format for D1 storage
    const normalizedKey = isXLabsFormat
      ? licenseKey.replaceAll("-", "_").toLowerCase()
      : licenseKey;

    // Use XLabs format for API lookup (XLabs stores keys as NM-PRO-*)
    const xlabsKey = isXLabsFormat
      ? licenseKey
      : licenseKey.replaceAll("_", "-").toUpperCase();

    // Verify against XLabs license API
    const xlabsToken = c.env.XLABS_API_KEY;
    if (!xlabsToken) {
      throw new HubError(500, "License verification not configured");
    }

    const verifyRes = await fetch(XLABS_API, {
      headers: { Authorization: `Bearer ${xlabsToken}` },
    });

    if (!verifyRes.ok) {
      throw new HubError(502, "License verification service unavailable");
    }

    const licenseData = await verifyRes.json<{
      success: boolean;
      data: Array<{
        license_key: string;
        project_slug: string;
        tier: string;
        status: string;
        features_json: string;
        expires_at: string | null;
      }>;
    }>();

    // Find matching license
    const match = licenseData.data?.find(
      (l) =>
        l.license_key === xlabsKey &&
        l.project_slug === "neural-memory" &&
        l.status === "active",
    );

    if (!match) {
      throw new HubError(403, "Invalid or expired license key");
    }

    // Extract tier and features
    const tier = VALID_TIERS.has(match.tier) ? match.tier : "pro";
    const expiresAt = match.expires_at || null;
    let features: string[] = [];
    try {
      features = JSON.parse(match.features_json || "[]");
    } catch {
      features = [];
    }

    const { userId } = c.get("auth");
    const db = c.env.SYNC_DB;
    const now = new Date().toISOString();

    // Upsert license in D1 (deactivate old ones first)
    await db.batch([
      db
        .prepare(
          "UPDATE licenses SET status = 'replaced' WHERE user_id = ? AND status = 'active'",
        )
        .bind(userId),
      db
        .prepare(
          `INSERT INTO licenses (id, user_id, tier, status, payment_provider, payment_id, created_at, expires_at)
           VALUES (?, ?, ?, 'active', 'xlabs', ?, ?, ?)`,
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
      activated_at: now,
      expires_at: expiresAt,
      features,
      message: `License activated! You now have ${tier} tier access.`,
    });
  } catch (err) {
    return handleError(c, err);
  }
});

export default activate;
