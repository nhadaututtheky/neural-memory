/**
 * NeuralMemory — OpenClaw Memory Plugin
 *
 * Brain-inspired persistent memory for AI agents.
 * Occupies the exclusive "memory" plugin slot.
 *
 * Architecture:
 *   OpenClaw ←→ Plugin (TypeScript) ←→ MCP stdio ←→ NeuralMemory (Python)
 *
 * v1.7.0: Dynamic tool proxy — fetches all tools from MCP `tools/list`
 * instead of hardcoding 6 tools. Automatically exposes every tool the
 * MCP server provides (39+ tools in NM v2.28.0).
 *
 * v1.8.0: Compatible with NM v2.29.0 — RRF score fusion, graph-based
 * query expansion, and Personalized PageRank activation.
 *
 * v1.8.1: Fix async register() — OpenClaw requires synchronous registration.
 * Fallback tools registered sync; MCP connection deferred to service.start().
 *
 * v1.9.0: Backward-compat shim tools (memory_search, memory_get) to prevent
 * "allowList contains unknown entries" warnings when NM replaces memory-core.
 *
 * v1.10.0: Singleton MCP client — multiple workspaces (multi-agent) share
 * the same connected client instance, keyed by (pythonPath, brain). Fixes
 * "NeuralMemory service not running" when OpenClaw registers the plugin
 * for a second workspace after gateway startup.
 *
 * Registers:
 *   N tools    — dynamically from MCP server (fallback: 5 core + 2 compat)
 *   1 service  — MCP process lifecycle (start/stop)
 *   5 hooks    — before_prompt_build (auto-context), agent_end (auto-capture),
 *                before_compaction (flush), before_reset (flush),
 *                gateway_start (consolidation)
 */

import type {
  OpenClawPluginDefinition,
  OpenClawPluginApi,
  BeforePromptBuildEvent,
  BeforePromptBuildResult,
  AgentEndEvent,
  SessionCompactEvent,
  CommandEvent,
  GatewayStartupEvent,
} from "./types.js";
import { NeuralMemoryMcpClient } from "./mcp-client.js";
import type { PluginLogger } from "./types.js";
import { createToolsFromMcp, createFallbackTools, createCompatibilityTools } from "./tools.js";
import type { ToolDefinition } from "./tools.js";

// ── Prompt metadata stripping ─────────────────────────────

/**
 * Strip metadata preamble from raw prompts before recall.
 *
 * OpenClaw + Telegram injects JSON metadata, NeuralMemory context blocks,
 * env vars, and system boilerplate into ev.prompt. Passing these raw to
 * nmem_recall creates junk neurons like "[concept] json message id".
 *
 * Stripping order matters — later passes clean up residue from earlier ones.
 */
export function stripPromptMetadata(raw: string): string {
  let cleaned = raw;

  // 1. Remove JSON blocks (Telegram metadata, conversation info)
  cleaned = cleaned.replace(
    /^\{[\s\S]*?"(?:conversation|message_id|sender_id|sender|chat_id|update_id)"[\s\S]*?\}$/gm,
    "",
  );

  // 2. Remove NeuralMemory context sections (## Relevant Memories, etc.)
  //    The |$ ensures sections at end-of-string are also stripped.
  cleaned = cleaned.replace(
    /^#{1,3}\s*(?:Relevant Memories|Related Information|Relevant Context|Neural Memory)[\s\S]*?(?=\n#{1,3}\s|\n\n(?![-•*\s])|$)/gim,
    "",
  );

  // 3. Remove neuron-type bullet lines injected by NM context
  cleaned = cleaned.replace(
    /^-\s*\[(?:concept|entity|decision|error|preference|insight|memory|fact|workflow|instruction|pattern)\].*$/gim,
    "",
  );

  // 4. Remove [NeuralMemory — ...] wrapper lines
  cleaned = cleaned.replace(/^\[NeuralMemory\s*[—–-].*\]$/gm, "");

  // 5. Remove metadata labels (untrusted metadata lines)
  cleaned = cleaned.replace(
    /^(?:Conversation info|Sender|Context|System)\s*\(.*?\)\s*:?\s*$/gim,
    "",
  );

  // 6. Remove env/export lines
  cleaned = cleaned.replace(/^export\s+\w+=.*$/gm, "");

  // 7. Collapse whitespace runs
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n").trim();

  // Fallback: if everything was stripped, use last non-empty line of raw
  if (!cleaned) {
    const lines = raw.split("\n").filter((l) => l.trim());
    cleaned = lines[lines.length - 1]?.trim() ?? raw.trim();
  }

  return cleaned;
}

// ── Auto-capture sanitization ─────────────────────────────

/**
 * Strip NeuralMemory context noise and metadata from auto-capture text.
 *
 * When agent_end forwards assistant messages to nmem_auto, those messages
 * may contain NM context wrappers that were injected by before_prompt_build.
 * Re-ingesting these creates junk neurons like "[concept] json message id".
 *
 * This is defense-in-depth — the Python input_firewall also strips these,
 * but catching them here avoids wasting network round-trips.
 */
export function sanitizeAutoCapture(raw: string): string {
  let cleaned = raw;

  // Strip NM context section headers
  cleaned = cleaned.replace(
    /^#{1,3}\s*(?:Relevant Memories|Related Information|Relevant Context|Neural Memory)\b.*$/gim,
    "",
  );

  // Strip [NeuralMemory — ...] wrapper lines
  cleaned = cleaned.replace(/^\[NeuralMemory\s*[—–-].*\]$/gm, "");

  // Strip neuron-type bullet lines (- [concept] ..., - [error] ...)
  cleaned = cleaned.replace(
    /^-\s*\[(?:concept|entity|decision|error|preference|insight|memory|fact|workflow|instruction|pattern)\]\s.*$/gim,
    "",
  );

  // Strip metadata labels
  cleaned = cleaned.replace(
    /^(?:Conversation info|Sender|Context)\s*\(.*?\)\s*:?\s*$/gim,
    "",
  );

  // Strip short acknowledgement lines (< 20 chars, common filler)
  cleaned = cleaned.replace(
    /^(?:OK|Sure|Done|Got it|Understood|Noted|Alright|I see|Thanks|Thank you|Okay)\.?\s*$/gim,
    "",
  );

  // Collapse whitespace
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n").trim();

  return cleaned;
}

// ── System prompt for tool awareness ──────────────────────

/**
 * Build a system prompt listing all registered tool names.
 * This makes the agent aware of which nmem_* tools are available.
 */
function buildToolInstructions(tools: ToolDefinition[]): string {
  const toolList = tools
    .map((t) => `- ${t.name}: ${t.description.slice(0, 100)}`)
    .join("\n");

  return `Neural Memory gives you persistent memory across sessions. Use it proactively — each session starts fresh, so without explicit saves ALL discoveries are lost.

These are TOOL CALLS, not CLI commands. Do NOT run "nmem remember" in terminal.

## Available Tools
${toolList}

nmem_* is your primary memory system. memory_search/memory_get are legacy aliases for nmem_recall.

## WHEN TO RECALL
- New session starts → nmem_recall("current project context")
- User references past event → nmem_recall("<that topic>")
- Prefix queries with project name for precision

## WHEN TO SAVE
After each task: did you make a decision (type="decision", priority=7), fix a bug (type="error", priority=7), learn a preference (type="preference", priority=8), or discover an insight (type="insight", priority=6)?

Save with: nmem_remember(content="Chose X over Y because Z", type="decision", priority=7, tags=["project", "topic"])

## CONTENT QUALITY
- Max 1-3 sentences. Use causal language: "Chose X because Y", "Root cause was X, fixed by Y".
- Always include project name + topic in tags (lowercase).
- For temporary scratch notes: nmem_remember(content="...", ephemeral=true) — auto-expires, never synced.

## SESSION END
nmem_auto(action="process", text="<brief session summary>")

## COMPACT MODE
All tools support compact=true (saves 60-80% tokens) and token_budget=N.`;
}

// ── Config ─────────────────────────────────────────────────

type PluginConfig = {
  pythonPath: string;
  brain: string;
  autoContext: boolean;
  autoCapture: boolean;
  autoFlush: boolean;
  autoConsolidate: boolean;
  contextDepth: number;
  maxContextTokens: number;
  timeout: number;
  initTimeout: number;
};

const DEFAULT_CONFIG: Readonly<PluginConfig> = {
  pythonPath: "python",
  brain: "default",
  autoContext: true,
  autoCapture: true,
  autoFlush: true,
  autoConsolidate: true,
  contextDepth: 1,
  maxContextTokens: 500,
  timeout: 30_000,
  initTimeout: 90_000,
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
    autoFlush:
      typeof merged.autoFlush === "boolean"
        ? merged.autoFlush
        : DEFAULT_CONFIG.autoFlush,
    autoConsolidate:
      typeof merged.autoConsolidate === "boolean"
        ? merged.autoConsolidate
        : DEFAULT_CONFIG.autoConsolidate,
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
    initTimeout:
      typeof merged.initTimeout === "number" &&
      Number.isFinite(merged.initTimeout) &&
      merged.initTimeout >= 10_000 &&
      merged.initTimeout <= 300_000
        ? merged.initTimeout
        : DEFAULT_CONFIG.initTimeout,
  };
}

// ── Singleton MCP client pool ────────────────────────────────
// Multiple workspaces may call register() independently, but all
// should share the same MCP process per (pythonPath, brain) combo.

const mcpClients = new Map<string, NeuralMemoryMcpClient>();

function getOrCreateMcpClient(
  cfg: PluginConfig,
  logger: PluginLogger,
): NeuralMemoryMcpClient {
  const key = `${cfg.pythonPath}::${cfg.brain}`;

  const existing = mcpClients.get(key);
  if (existing) {
    logger.debug?.(`Reusing existing MCP client for brain "${cfg.brain}"`);
    return existing;
  }

  const mcp = new NeuralMemoryMcpClient({
    pythonPath: cfg.pythonPath,
    brain: cfg.brain,
    logger,
    timeout: cfg.timeout,
    initTimeout: cfg.initTimeout,
  });

  mcpClients.set(key, mcp);
  return mcp;
}

// ── Plugin definition ──────────────────────────────────────

const plugin: OpenClawPluginDefinition = {
  id: "neuralmemory",
  name: "Neural Memory",
  description:
    "Brain-inspired persistent memory for AI agents — neurons, synapses, and fibers",
  version: "1.16.1",
  kind: "memory",

  register(api: OpenClawPluginApi): void {
    const cfg = resolveConfig(api.pluginConfig);

    const mcp = getOrCreateMcpClient(cfg, api.logger);

    // ── Register fallback tools synchronously ────────────
    // OpenClaw requires register() to be synchronous.
    // Register stable fallback tools immediately; MCP connection
    // and dynamic tool discovery happen in service.start().
    // Fallback tools auto-reconnect MCP on first call.

    const registeredTools = createFallbackTools(mcp);
    const compatTools = createCompatibilityTools(mcp);

    for (const t of [...registeredTools, ...compatTools]) {
      api.registerTool(t, { name: t.name });
    }

    api.logger.info(
      `Registered ${registeredTools.length} NeuralMemory tools + ${compatTools.length} compat shims (sync)`,
    );

    // ── Service: MCP process lifecycle ───────────────────

    api.registerService({
      id: "neuralmemory-mcp",

      async start(): Promise<void> {
        if (!mcp.connected) {
          try {
            await mcp.connect();
            api.logger.info("NeuralMemory MCP connected in service.start()");

            // Log discovered tools for diagnostics (cannot re-register
            // after register() — OpenClaw freezes the tool list).
            try {
              const dynamicTools = await createToolsFromMcp(mcp);
              api.logger.info(
                `NeuralMemory MCP discovered ${dynamicTools.length} tools`,
              );
            } catch (err) {
              api.logger.warn(
                `Tool discovery failed: ${(err as Error).message}`,
              );
            }
          } catch (err) {
            api.logger.error(
              `Failed to start NeuralMemory MCP: ${(err as Error).message}`,
            );
            throw err;
          }
        }
      },

      async stop(): Promise<void> {
        // Remove from singleton pool so next register() creates fresh client
        const key = `${cfg.pythonPath}::${cfg.brain}`;
        mcpClients.delete(key);
        await mcp.close();
        api.logger.info("NeuralMemory MCP service stopped");
      },
    });

    // ── Hook: tool awareness + auto-context before prompt build ──
    // Migrated from legacy before_agent_start to before_prompt_build
    // per OpenClaw compatibility guidance (issue #116).

    api.on(
      "before_prompt_build",
      async (
        event: unknown,
        _ctx: unknown,
      ): Promise<BeforePromptBuildResult | void> => {
        const result: BeforePromptBuildResult = {
          systemPrompt: buildToolInstructions(registeredTools),
        };

        if (cfg.autoContext && mcp.connected) {
          const ev = event as BeforePromptBuildEvent;

          try {
            const query = stripPromptMetadata(ev.prompt);
            const raw = await mcp.callTool("nmem_recall", {
              query,
              depth: cfg.contextDepth,
              max_tokens: cfg.maxContextTokens,
              clean_for_prompt: true,
            });

            const data = JSON.parse(raw) as {
              answer?: string;
              confidence?: number;
            };

            if (data.answer && (data.confidence ?? 0) > 0.1) {
              result.prependContext = `[NeuralMemory — relevant context]\n${data.answer}`;
            }
          } catch (err) {
            api.logger.warn(
              `Auto-context failed: ${(err as Error).message}`,
            );
          }
        }

        return result;
      },
      { priority: 10 },
    );

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
            const rawText = messages
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

            // Strip NM context noise and short acknowledgements before re-ingest
            const text = sanitizeAutoCapture(rawText);

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

    // ── Hook: flush memories before context compaction ──
    // Migrated from legacy session:compact:before to before_compaction

    if (cfg.autoFlush) {
      api.on(
        "before_compaction",
        async (_event: unknown, _ctx: unknown): Promise<void> => {
          if (!mcp.connected) return;

          try {
            await mcp.callTool("nmem_auto", {
              action: "process",
              text: "[pre-compact emergency flush]",
            });
            api.logger.info("Pre-compact flush completed");
          } catch (err) {
            api.logger.warn(
              `Pre-compact flush failed: ${(err as Error).message}`,
            );
          }
        },
        { priority: 5 },
      );

      // Flush on session boundary (/new and /reset)
      // Migrated from legacy command:new + command:reset to before_reset
      api.on(
        "before_reset",
        async (_event: unknown, _ctx: unknown): Promise<void> => {
          if (!mcp.connected) return;

          try {
            await mcp.callTool("nmem_auto", {
              action: "process",
              text: "[session boundary — reset]",
            });
          } catch (err) {
            api.logger.warn(
              `Session boundary flush failed: ${(err as Error).message}`,
            );
          }
        },
        { priority: 10 },
      );
    }

    // ── Hook: consolidation on gateway start ─────────────
    // Migrated from legacy gateway:startup to gateway_start

    if (cfg.autoConsolidate) {
      api.on(
        "gateway_start",
        async (_event: unknown, _ctx: unknown): Promise<void> => {
          if (!mcp.connected) {
            try {
              await mcp.ensureConnected();
            } catch (err) {
              api.logger.warn(
                `MCP connect on startup failed: ${(err as Error).message}`,
              );
              return;
            }
          }

          try {
            await mcp.callTool("nmem_consolidate", {
              strategy: "enrich",
              compact: true,
            });
            api.logger.info("Startup consolidation completed");
          } catch (err) {
            api.logger.warn(
              `Startup consolidation failed: ${(err as Error).message}`,
            );
          }
        },
        { priority: 50 },
      );
    }

    // ── Done ────────────────────────────────────────────

    api.logger.info(
      `NeuralMemory registered (brain: ${cfg.brain}, ` +
        `autoContext: ${cfg.autoContext}, autoCapture: ${cfg.autoCapture}, ` +
        `autoFlush: ${cfg.autoFlush}, autoConsolidate: ${cfg.autoConsolidate}) — ` +
        `tools will be loaded dynamically from MCP on service start`,
    );
  },
};

export default plugin;
