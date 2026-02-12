import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { buildChildEnv, ALLOWED_ENV_KEYS } from "../src/mcp-client.js";

describe("buildChildEnv", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Clear all env vars for predictable tests
    for (const key of Object.keys(process.env)) {
      delete process.env[key];
    }
  });

  afterEach(() => {
    // Restore original env
    for (const key of Object.keys(process.env)) {
      delete process.env[key];
    }
    Object.assign(process.env, originalEnv);
  });

  it("returns empty env when no allowed keys are set", () => {
    const env = buildChildEnv("default");
    expect(env).toEqual({});
  });

  it("does not set NEURALMEMORY_BRAIN for default brain", () => {
    const env = buildChildEnv("default");
    expect(env).not.toHaveProperty("NEURALMEMORY_BRAIN");
  });

  it("sets NEURALMEMORY_BRAIN for custom brain", () => {
    const env = buildChildEnv("my-brain");
    expect(env.NEURALMEMORY_BRAIN).toBe("my-brain");
  });

  it("only passes through whitelisted keys", () => {
    process.env.PATH = "/usr/bin";
    process.env.HOME = "/home/user";
    process.env.SECRET_KEY = "should-not-pass";
    process.env.API_KEY = "should-not-pass";
    process.env.AWS_SECRET_ACCESS_KEY = "should-not-pass";

    const env = buildChildEnv("default");

    expect(env.PATH).toBe("/usr/bin");
    expect(env.HOME).toBe("/home/user");
    expect(env).not.toHaveProperty("SECRET_KEY");
    expect(env).not.toHaveProperty("API_KEY");
    expect(env).not.toHaveProperty("AWS_SECRET_ACCESS_KEY");
  });

  it("excludes secret-like keys (TOKEN, PASSWORD, CREDENTIAL)", () => {
    process.env.GITHUB_TOKEN = "ghp_xxx";
    process.env.DATABASE_PASSWORD = "pass123";
    process.env.AWS_CREDENTIAL = "cred";

    const env = buildChildEnv("default");

    expect(env).not.toHaveProperty("GITHUB_TOKEN");
    expect(env).not.toHaveProperty("DATABASE_PASSWORD");
    expect(env).not.toHaveProperty("AWS_CREDENTIAL");
  });

  it("omits undefined env vars (no undefined values in result)", () => {
    process.env.PATH = "/usr/bin";
    // HOME is not set

    const env = buildChildEnv("default");

    expect(env.PATH).toBe("/usr/bin");
    expect(Object.values(env).every((v) => v !== undefined)).toBe(true);
    // HOME should not be present since it's not set
    if (!("HOME" in env)) {
      expect(env).not.toHaveProperty("HOME");
    }
  });

  it("passes through all NeuralMemory-specific env vars", () => {
    process.env.NEURALMEMORY_DIR = "/data/nm";
    process.env.NEURAL_MEMORY_DIR = "/alt/dir";
    process.env.NEURAL_MEMORY_JSON = "/config.json";
    process.env.NEURAL_MEMORY_DEBUG = "1";

    const env = buildChildEnv("default");

    expect(env.NEURALMEMORY_DIR).toBe("/data/nm");
    expect(env.NEURAL_MEMORY_DIR).toBe("/alt/dir");
    expect(env.NEURAL_MEMORY_JSON).toBe("/config.json");
    expect(env.NEURAL_MEMORY_DEBUG).toBe("1");
  });

  it("passes through Python env vars", () => {
    process.env.VIRTUAL_ENV = "/venv";
    process.env.CONDA_PREFIX = "/conda";
    process.env.PYTHONPATH = "/lib";
    process.env.PYTHONHOME = "/python";

    const env = buildChildEnv("default");

    expect(env.VIRTUAL_ENV).toBe("/venv");
    expect(env.CONDA_PREFIX).toBe("/conda");
    expect(env.PYTHONPATH).toBe("/lib");
    expect(env.PYTHONHOME).toBe("/python");
  });
});

describe("ALLOWED_ENV_KEYS", () => {
  it("is a ReadonlySet with expected keys", () => {
    expect(ALLOWED_ENV_KEYS).toBeInstanceOf(Set);
    expect(ALLOWED_ENV_KEYS.has("PATH")).toBe(true);
    expect(ALLOWED_ENV_KEYS.has("HOME")).toBe(true);
    expect(ALLOWED_ENV_KEYS.has("VIRTUAL_ENV")).toBe(true);
    expect(ALLOWED_ENV_KEYS.has("NEURALMEMORY_BRAIN")).toBe(true);
  });

  it("does not contain secret-like keys", () => {
    for (const key of ALLOWED_ENV_KEYS) {
      expect(key).not.toMatch(/SECRET|TOKEN|PASSWORD|CREDENTIAL/i);
    }
  });
});
