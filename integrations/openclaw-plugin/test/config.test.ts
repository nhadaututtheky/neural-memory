import { describe, it, expect } from "vitest";
import { resolveConfig, BRAIN_NAME_RE, MAX_AUTO_CAPTURE_CHARS } from "../src/index.js";

describe("resolveConfig", () => {
  it("returns defaults when called with no input", () => {
    const cfg = resolveConfig();
    expect(cfg).toEqual({
      pythonPath: "python",
      brain: "default",
      autoContext: true,
      autoCapture: true,
      contextDepth: 1,
      maxContextTokens: 500,
      timeout: 30_000,
    });
  });

  it("returns defaults for undefined input", () => {
    const cfg = resolveConfig(undefined);
    expect(cfg).toEqual(resolveConfig());
  });

  it("returns defaults for empty object", () => {
    const cfg = resolveConfig({});
    expect(cfg).toEqual(resolveConfig());
  });

  it("merges valid partial config", () => {
    const cfg = resolveConfig({
      pythonPath: "/usr/bin/python3",
      brain: "my-brain",
      autoContext: false,
    });
    expect(cfg.pythonPath).toBe("/usr/bin/python3");
    expect(cfg.brain).toBe("my-brain");
    expect(cfg.autoContext).toBe(false);
    expect(cfg.autoCapture).toBe(true);
    expect(cfg.contextDepth).toBe(1);
  });

  // ── pythonPath ──

  it("rejects non-string pythonPath → falls back to default", () => {
    const cfg = resolveConfig({ pythonPath: 123 });
    expect(cfg.pythonPath).toBe("python");
  });

  it("rejects empty string pythonPath → falls back to default", () => {
    const cfg = resolveConfig({ pythonPath: "" });
    expect(cfg.pythonPath).toBe("python");
  });

  // ── brain ──

  it("rejects invalid brain name → falls back to default", () => {
    const cfg = resolveConfig({ brain: "has spaces!" });
    expect(cfg.brain).toBe("default");
  });

  it("rejects brain name longer than 64 chars → falls back", () => {
    const cfg = resolveConfig({ brain: "a".repeat(65) });
    expect(cfg.brain).toBe("default");
  });

  it("rejects non-string brain → falls back to default", () => {
    const cfg = resolveConfig({ brain: 42 });
    expect(cfg.brain).toBe("default");
  });

  it("accepts valid brain names with dots, hyphens, underscores", () => {
    const cfg = resolveConfig({ brain: "my_brain-v2.1" });
    expect(cfg.brain).toBe("my_brain-v2.1");
  });

  // ── autoContext / autoCapture ──

  it("rejects non-boolean autoContext → falls back to default", () => {
    const cfg = resolveConfig({ autoContext: "yes" });
    expect(cfg.autoContext).toBe(true);
  });

  it("rejects non-boolean autoCapture → falls back to default", () => {
    const cfg = resolveConfig({ autoCapture: 1 });
    expect(cfg.autoCapture).toBe(true);
  });

  it("accepts false for autoContext/autoCapture", () => {
    const cfg = resolveConfig({ autoContext: false, autoCapture: false });
    expect(cfg.autoContext).toBe(false);
    expect(cfg.autoCapture).toBe(false);
  });

  // ── contextDepth ──

  it("rejects contextDepth below 0 → falls back to default", () => {
    const cfg = resolveConfig({ contextDepth: -1 });
    expect(cfg.contextDepth).toBe(1);
  });

  it("rejects contextDepth above 3 → falls back to default", () => {
    const cfg = resolveConfig({ contextDepth: 4 });
    expect(cfg.contextDepth).toBe(1);
  });

  it("rejects non-integer contextDepth → falls back to default", () => {
    const cfg = resolveConfig({ contextDepth: 1.5 });
    expect(cfg.contextDepth).toBe(1);
  });

  it("accepts valid contextDepth values (0, 1, 2, 3)", () => {
    for (const depth of [0, 1, 2, 3]) {
      const cfg = resolveConfig({ contextDepth: depth });
      expect(cfg.contextDepth).toBe(depth);
    }
  });

  // ── maxContextTokens ──

  it("rejects maxContextTokens below 100 → falls back", () => {
    const cfg = resolveConfig({ maxContextTokens: 50 });
    expect(cfg.maxContextTokens).toBe(500);
  });

  it("rejects maxContextTokens above 10000 → falls back", () => {
    const cfg = resolveConfig({ maxContextTokens: 20_000 });
    expect(cfg.maxContextTokens).toBe(500);
  });

  // ── timeout ──

  it("rejects timeout below 5000 → falls back to default", () => {
    const cfg = resolveConfig({ timeout: 1000 });
    expect(cfg.timeout).toBe(30_000);
  });

  it("rejects timeout above 120000 → falls back to default", () => {
    const cfg = resolveConfig({ timeout: 200_000 });
    expect(cfg.timeout).toBe(30_000);
  });

  it("rejects non-finite timeout → falls back to default", () => {
    const cfg = resolveConfig({ timeout: Infinity });
    expect(cfg.timeout).toBe(30_000);
  });

  it("accepts valid timeout at boundaries", () => {
    expect(resolveConfig({ timeout: 5_000 }).timeout).toBe(5_000);
    expect(resolveConfig({ timeout: 120_000 }).timeout).toBe(120_000);
  });

  // ── null / undefined values for individual fields ──

  it("handles null values for fields → falls back to defaults", () => {
    const cfg = resolveConfig({
      pythonPath: null,
      brain: null,
      autoContext: null,
      timeout: null,
    } as unknown as Record<string, unknown>);
    expect(cfg.pythonPath).toBe("python");
    expect(cfg.brain).toBe("default");
    expect(cfg.autoContext).toBe(true);
    expect(cfg.timeout).toBe(30_000);
  });
});

describe("BRAIN_NAME_RE", () => {
  it("matches valid brain names", () => {
    expect(BRAIN_NAME_RE.test("default")).toBe(true);
    expect(BRAIN_NAME_RE.test("my-brain")).toBe(true);
    expect(BRAIN_NAME_RE.test("brain_v2.0")).toBe(true);
    expect(BRAIN_NAME_RE.test("A")).toBe(true);
  });

  it("rejects invalid brain names", () => {
    expect(BRAIN_NAME_RE.test("")).toBe(false);
    expect(BRAIN_NAME_RE.test("has space")).toBe(false);
    expect(BRAIN_NAME_RE.test("special@char")).toBe(false);
    expect(BRAIN_NAME_RE.test("a".repeat(65))).toBe(false);
  });
});

describe("MAX_AUTO_CAPTURE_CHARS", () => {
  it("is 50000", () => {
    expect(MAX_AUTO_CAPTURE_CHARS).toBe(50_000);
  });
});
