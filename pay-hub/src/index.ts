import { Hono } from "hono";
import { cors } from "hono/cors";
import type { AppEnv } from "./types.js";
import checkout from "./routes/checkout.js";
import order from "./routes/order.js";
import webhook from "./routes/webhook.js";

const app = new Hono<AppEnv>();

// CORS for landing page
app.use(
  "*",
  cors({
    origin: [
      "https://neuralmemory.theio.vn",
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

// Verify endpoint (for sync hub backward compat)
app.post("/verify", async (c) => {
  const body = await c.req.json<{ key: string }>();
  const key = (body.key || "").trim();

  if (!key) {
    return c.json({ valid: false, error: "Missing key" });
  }

  // Normalize to XLabs format for lookup
  const xlabsKey = key.startsWith("nm_")
    ? key.replaceAll("_", "-").toUpperCase()
    : key;

  // Check XLabs
  const res = await fetch("https://admin.theio.vn/api/licenses", {
    headers: { Authorization: `Bearer ${c.env.XLABS_API_KEY}` },
  });

  if (!res.ok) {
    return c.json({ valid: false, error: "Verification service unavailable" });
  }

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
      l.license_key === xlabsKey &&
      l.project_slug === "neural-memory" &&
      l.status === "active",
  );

  if (!match) {
    return c.json({ valid: false, error: "Invalid or expired license key" });
  }

  let features: string[] = [];
  try {
    features = JSON.parse(match.features_json || "[]");
  } catch {
    features = [];
  }

  return c.json({
    valid: true,
    tier: match.tier,
    expires_at: match.expires_at,
    features,
  });
});

export default app;
