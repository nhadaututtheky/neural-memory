/**
 * NeuralMemory — OpenClaw Memory Plugin
 *
 * Brain-inspired persistent memory for AI agents.
 * Occupies the exclusive "memory" plugin slot.
 *
 * Architecture:
 *   OpenClaw ←→ Plugin (TypeScript) ←→ MCP stdio ←→ NeuralMemory (Python)
 *
 * Registers:
 *   6 tools    — nmem_remember, nmem_recall, nmem_context, nmem_todo, nmem_stats, nmem_health
 *   1 service  — MCP process lifecycle (start/stop)
 *   2 hooks    — before_agent_start (auto-context), agent_end (auto-capture)
 */

import type {
  OpenClawPluginDefinition,
  OpenClawPluginApi,
  BeforeAgentStartEvent,
  AgentContext,
  AgentEndEvent,
} from "./types.js";
import { NeuralMemoryMcpClient } from "./mcp-client.js";
import { createTools } from "./tools.js";

// ── Config ─────────────────────────────────────────────────

type PluginConfig = {
  pythonPath: string;
  brain: string;
  autoContext: boolean;
  autoCapture: boolean;
  contextDepth: number;
  maxContextTokens: number;
  timeout: number;
};

const DEFAULT_CONFIG: Readonly<PluginConfig> = {
  pythonPath: "python",
  brain: "default",
  autoContext: true,
  autoCapture: true,
  contextDepth: 1,
  maxContextTokens: 500,
  timeout: 30_000,
};

export const BRAIN_NAME_RE = /^[a-zA-Z0-9_\-.]{1,64}$/;
export const MAX_AUTO_CAPTURE_CHARS = 50_000;

export function resolveConfig(raw?: Record<string, unknown>): PluginConfig {
  const merged = { ...DEFAULT_CONFIG, ...(raw ?? {}) };

  return {
    pythonPath:
      typeof merged.pythonPath === "string" && merged.pythonPath.length > 0
        ? merged.pythonPath
        : DEFAULT_CONFIG.pythonPath,
    brain:
      typeof merged.brain === "string" && BRAIN_NAME_RE.test(merged.brain)
        ? merged.brain
        : DEFAULT_CONFIG.brain,
    autoContext:
      typeof merged.autoContext === "boolean"
        ? merged.autoContext
        : DEFAULT_CONFIG.autoContext,
    autoCapture:
      typeof merged.autoCapture === "boolean"
        ? merged.autoCapture
        : DEFAULT_CONFIG.autoCapture,
    contextDepth:
      typeof merged.contextDepth === "number" &&
      Number.isInteger(merged.contextDepth) &&
      merged.contextDepth >= 0 &&
      merged.contextDepth <= 3
        ? merged.contextDepth
        : DEFAULT_CONFIG.contextDepth,
    maxContextTokens:
      typeof merged.maxContextTokens === "number" &&
      Number.isInteger(merged.maxContextTokens) &&
      merged.maxContextTokens >= 100 &&
      merged.maxContextTokens <= 10_000
        ? merged.maxContextTokens
        : DEFAULT_CONFIG.maxContextTokens,
    timeout:
      typeof merged.timeout === "number" &&
      Number.isFinite(merged.timeout) &&
      merged.timeout >= 5_000 &&
      merged.timeout <= 120_000
        ? merged.timeout
        : DEFAULT_CONFIG.timeout,
  };
}

// ── Plugin definition ──────────────────────────────────────

const plugin: OpenClawPluginDefinition = {
  id: "neuralmemory",
  name: "NeuralMemory",
  description:
    "Brain-inspired persistent memory for AI agents — neurons, synapses, and fibers",
  version: "1.4.1",
  kind: "memory",

  register(api: OpenClawPluginApi): void {
    const cfg = resolveConfig(api.pluginConfig);

    const mcp = new NeuralMemoryMcpClient({
      pythonPath: cfg.pythonPath,
      brain: cfg.brain,
      logger: api.logger,
      timeout: cfg.timeout,
    });

    // ── Service: MCP process lifecycle ───────────────────

    api.registerService({
      id: "neuralmemory-mcp",

      async start(): Promise<void> {
        try {
          await mcp.connect();
          api.logger.info("NeuralMemory MCP service started");
        } catch (err) {
          api.logger.error(
            `Failed to start NeuralMemory MCP: ${(err as Error).message}`,
          );
          throw err;
        }
      },

      async stop(): Promise<void> {
        await mcp.close();
        api.logger.info("NeuralMemory MCP service stopped");
      },
    });

    // ── Tools: 6 core memory tools ──────────────────────

    const tools = createTools(mcp);

    for (const t of tools) {
      api.registerTool(t, { name: t.name });
    }

    // ── Hook: auto-context before agent start ───────────

    if (cfg.autoContext) {
      api.on(
        "before_agent_start",
        async (
          event: unknown,
          _ctx: unknown,
        ): Promise<{ prependContext?: string } | void> => {
          if (!mcp.connected) return;

          const ev = event as BeforeAgentStartEvent;

          try {
            const raw = await mcp.callTool("nmem_recall", {
              query: ev.prompt,
              depth: cfg.contextDepth,
              max_tokens: cfg.maxContextTokens,
            });

            const data = JSON.parse(raw) as {
              answer?: string;
              confidence?: number;
            };

            if (data.answer && (data.confidence ?? 0) > 0.1) {
              return {
                prependContext: `[NeuralMemory — relevant context]\n${data.answer}`,
              };
            }
          } catch (err) {
            api.logger.warn(
              `Auto-context failed: ${(err as Error).message}`,
            );
          }
        },
        { priority: 10 },
      );
    }

    // ── Hook: auto-capture after agent completes ────────

    if (cfg.autoCapture) {
      api.on(
        "agent_end",
        async (event: unknown, _ctx: unknown): Promise<void> => {
          if (!mcp.connected) return;

          const ev = event as AgentEndEvent;
          if (!ev.success) return;

          try {
            const messages = ev.messages?.slice(-5) ?? [];
            const text = messages
              .filter(
                (m: unknown): m is { role: string; content: string } =>
                  typeof m === "object" &&
                  m !== null &&
                  (m as { role?: string }).role === "assistant" &&
                  typeof (m as { content?: unknown }).content === "string",
              )
              .map((m) => m.content)
              .join("\n")
              .slice(0, MAX_AUTO_CAPTURE_CHARS);

            if (text.length > 50) {
              await mcp.callTool("nmem_auto", {
                action: "process",
                text,
              });
            }
          } catch (err) {
            api.logger.warn(
              `Auto-capture failed: ${(err as Error).message}`,
            );
          }
        },
        { priority: 90 },
      );
    }

    // ── Done ────────────────────────────────────────────

    api.logger.info(
      `NeuralMemory registered (brain: ${cfg.brain}, tools: ${tools.length}, ` +
        `autoContext: ${cfg.autoContext}, autoCapture: ${cfg.autoCapture})`,
    );
  },
};

export default plugin;
