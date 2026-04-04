import { Hono } from "hono";
import { cors } from "hono/cors";
import type { AppEnv, Order } from "./types.js";
import checkout from "./routes/checkout.js";
import order from "./routes/order.js";
import webhook from "./routes/webhook.js";
import { fulfillOrder } from "./lib/license.js";

const app = new Hono<AppEnv>();

// CORS for landing page
app.use(
  "*",
  cors({
    origin: [
      "https://neuralmemory.theio.vn",
      "https://companion.theio.vn",
      "https://nhadaututtheky.github.io",
      "http://localhost:3000",
    ],
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
  }),
);

// Health
app.get("/", (c) =>
  c.json({
    name: "Neural Memory Pay Hub",
    version: "1.0.0",
    status: "healthy",
  }),
);

app.get("/health", (c) => c.json({ status: "ok" }));

// Routes
app.route("/checkout", checkout);
app.route("/order", order);
app.route("/webhook", webhook);

// Pro features granted on all verified licenses
const PRO_FEATURES = [
  "merkle_sync",
  "cone_queries",
  "directional_compression",
  "cross_encoder",
  "smart_merge",
  "infinity_db",
];

// ── Shared verify logic ────────────────────────────────────────────────────

async function verifyKey(c: any, key: string) {
  if (!key) {
    return c.json({ valid: false, error: "Missing key" });
  }

  const normalizedKey = key.startsWith("nm_")
    ? key.replaceAll("_", "-").toUpperCase()
    : key.toUpperCase();

  // Companion keys (CPN-*) → forward to companion-verify via Service Binding
  if (normalizedKey.startsWith("CPN-")) {
    try {
      const worker = c.env.COMPANION_WORKER;
      const res = await worker.fetch(
        new Request(`https://companion.theio.vn/verify?key=${encodeURIComponent(key)}`),
      );
      const data = await res.json();
      return c.json(data, res.status);
    } catch {
      return c.json({ valid: false, error: "Companion verify unreachable" }, 502);
    }
  }

  // NM keys → D1 primary, XLabs fallback
  const db = c.env.PAY_DB;
  const d1Order = await db
    .prepare(
      "SELECT product, license_key, fulfilled_at FROM orders WHERE UPPER(license_key) = ? AND status = 'fulfilled'",
    )
    .bind(normalizedKey)
    .first<{ product: string; license_key: string; fulfilled_at: string }>();

  if (d1Order) {
    const tier = d1Order.product.includes("TEAM") ? "team" : "pro";
    return c.json({
      valid: true,
      tier,
      expires_at: null,
      features: PRO_FEATURES,
    });
  }

  try {
    const res = await fetch("https://admin.theio.vn/api/licenses", {
      headers: { Authorization: `Bearer ${c.env.XLABS_API_KEY}` },
    });

    if (res.ok) {
      const data = await res.json<{
        data: Array<{
          license_key: string;
          project_slug: string;
          tier: string;
          status: string;
          features_json: string;
          expires_at: string | null;
        }>;
      }>();

      const match = data.data?.find(
        (l) =>
          l.license_key.toUpperCase() === normalizedKey &&
          l.project_slug === "neural-memory" &&
          l.status === "active",
      );

      if (match) {
        let features: string[] = [];
        try {
          features = JSON.parse(match.features_json || "[]");
        } catch {
          features = PRO_FEATURES;
        }
        return c.json({
          valid: true,
          tier: match.tier,
          expires_at: match.expires_at,
          features,
        });
      }
    }
  } catch {
    // XLabs API unreachable
  }

  return c.json({ valid: false, error: "Invalid or expired license key" });
}

// ── License sync relay (XLabs → companion-verify via Service Binding) ──────

app.post("/admin/license/sync", async (c) => {
  // Auth: XLabs webhook secret or companion admin secret
  const auth = c.req.header("Authorization");
  const expected = `Bearer ${c.env.COMPANION_ADMIN_SECRET}`;
  if (auth !== expected) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  const body = await c.req.json<{
    action: "create" | "revoke";
    license_key: string;
    tier?: string;
    email?: string;
    name?: string;
    max_sessions?: number;
    expires_at?: string;
    duration_days?: number;
  }>();

  const worker = c.env.COMPANION_WORKER;

  if (body.action === "create") {
    let durationDays = body.duration_days ?? 370;
    if (body.expires_at) {
      const ms = new Date(body.expires_at).getTime() - Date.now();
      durationDays = Math.max(1, Math.ceil(ms / 86_400_000));
    }

    // 1. Relay to companion worker
    const res = await worker.fetch(
      new Request("https://companion.theio.vn/admin/create", {
        method: "POST",
        headers: {
          Authorization: expected,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          key: body.license_key,
          tier: body.tier ?? "pro",
          email: body.email ?? "",
          durationDays,
        }),
      }),
    );
    const data = await res.json();

    // 2. Also insert into D1 so /verify finds it without XLabs fallback
    try {
      const db = c.env.PAY_DB;
      const now = new Date().toISOString();
      const product = (body.tier === "team") ? "NM-TEAM-YEARLY" : "NM-PRO-YEARLY";
      const existing = await db
        .prepare("SELECT id FROM orders WHERE UPPER(license_key) = ?")
        .bind(body.license_key.toUpperCase())
        .first();
      if (!existing) {
        await db
          .prepare(
            `INSERT INTO orders (id, product, email, source, status, amount_vnd, license_key, created_at, paid_at, fulfilled_at)
             VALUES (?, ?, ?, 'polar', 'fulfilled', 0, ?, ?, ?, ?)`,
          )
          .bind(`sync-${Date.now()}`, product, body.email ?? "", body.license_key, now, now, now)
          .run();
      }
    } catch {
      // D1 insert is best-effort — companion is primary for sync keys
    }

    return c.json(data, res.status);
  }

  if (body.action === "revoke") {
    const res = await worker.fetch(
      new Request("https://companion.theio.vn/admin/revoke", {
        method: "POST",
        headers: {
          Authorization: expected,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ key: body.license_key }),
      }),
    );
    const data = await res.json();
    return c.json(data, res.status);
  }

  return c.json({ error: "Invalid action" }, 400);
});

// ── Admin: grant license directly into D1 ──────────────────────────────────

app.post("/admin/license/grant", async (c) => {
  const auth = c.req.header("Authorization");
  const expected = `Bearer ${c.env.COMPANION_ADMIN_SECRET}`;
  if (auth !== expected) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  const body = await c.req.json<{
    license_key: string;
    product?: string;
    email?: string;
    tier?: string;
  }>();

  const key = (body.license_key || "").trim();
  if (!key) {
    return c.json({ error: "Missing license_key" }, 400);
  }

  const db = c.env.PAY_DB;
  const product = body.product ?? (body.tier === "team" ? "NM-TEAM-YEARLY" : "NM-PRO-YEARLY");
  const email = body.email ?? "admin@theio.vn";
  const id = `grant-${Date.now()}`;
  const now = new Date().toISOString();

  // Upsert: check if key already exists
  const existing = await db
    .prepare("SELECT id FROM orders WHERE UPPER(license_key) = ?")
    .bind(key.toUpperCase())
    .first();

  if (existing) {
    // Update to fulfilled
    await db
      .prepare("UPDATE orders SET status = 'fulfilled', fulfilled_at = ? WHERE UPPER(license_key) = ?")
      .bind(now, key.toUpperCase())
      .run();
    return c.json({ success: true, action: "updated", license_key: key });
  }

  await db
    .prepare(
      `INSERT INTO orders (id, product, email, source, status, amount_vnd, license_key, created_at, paid_at, fulfilled_at)
       VALUES (?, ?, ?, 'polar', 'fulfilled', 0, ?, ?, ?, ?)`,
    )
    .bind(id, product, email, key, now, now, now)
    .run();

  return c.json({ success: true, action: "created", license_key: key, id });
});

// ── Admin: list orders ─────────────────────────────────────────────────────

app.get("/admin/orders", async (c) => {
  const auth = c.req.header("Authorization");
  const expected = `Bearer ${c.env.COMPANION_ADMIN_SECRET}`;
  if (auth !== expected) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  const status = c.req.query("status"); // optional filter
  const db = c.env.PAY_DB;
  const query = status
    ? db.prepare("SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC LIMIT 50").bind(status)
    : db.prepare("SELECT * FROM orders ORDER BY created_at DESC LIMIT 50");

  const { results } = await query.all<Order>();
  return c.json({ orders: results, count: results.length });
});

// ── Admin: retry fulfillment for stuck orders ──────────────────────────────

app.post("/admin/orders/fulfill", async (c) => {
  const auth = c.req.header("Authorization");
  const expected = `Bearer ${c.env.COMPANION_ADMIN_SECRET}`;
  if (auth !== expected) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  const body = await c.req.json<{ order_id?: string; status_filter?: string }>();
  const db = c.env.PAY_DB;

  if (body.order_id) {
    // Fulfill specific order
    try {
      const { licenseKey } = await fulfillOrder(c.env, db, body.order_id);
      return c.json({ success: true, order_id: body.order_id, license_key: licenseKey });
    } catch (err) {
      return c.json({ success: false, error: String(err) }, 500);
    }
  }

  // Bulk: fulfill all paid-but-not-fulfilled orders
  const filter = body.status_filter || "paid";
  const { results } = await db
    .prepare("SELECT id FROM orders WHERE status = ? AND license_key IS NULL")
    .bind(filter)
    .all<{ id: string }>();

  const fulfilled: string[] = [];
  const failed: Array<{ id: string; error: string }> = [];

  for (const row of results) {
    try {
      await fulfillOrder(c.env, db, row.id);
      fulfilled.push(row.id);
    } catch (err) {
      failed.push({ id: row.id, error: String(err) });
    }
  }

  return c.json({ fulfilled, failed, total: results.length });
});

// Verify endpoint — supports both GET (Companion app) and POST (legacy)
app.get("/verify", async (c) => {
  const key = (c.req.query("key") || "").trim();
  return verifyKey(c, key);
});

app.post("/verify", async (c) => {
  const body = await c.req.json<{ key: string }>();
  const key = (body.key || "").trim();
  return verifyKey(c, key);
});

export default app;
