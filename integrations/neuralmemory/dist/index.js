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
 *   2 hooks    — before_agent_start (auto-context), agent_end (auto-capture)
 */
import { NeuralMemoryMcpClient } from "./mcp-client.js";
import { createToolsFromMcp, createFallbackTools, createCompatibilityTools } from "./tools.js";
// ── System prompt for tool awareness ──────────────────────
/**
 * Build a system prompt listing all registered tool names.
 * This makes the agent aware of which nmem_* tools are available.
 */
function buildToolInstructions(tools) {
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
const DEFAULT_CONFIG = {
    pythonPath: "python",
    brain: "default",
    autoContext: true,
    autoCapture: true,
    contextDepth: 1,
    maxContextTokens: 500,
    timeout: 30_000,
    initTimeout: 90_000,
};
export const BRAIN_NAME_RE = /^[a-zA-Z0-9_\-.]{1,64}$/;
export const MAX_AUTO_CAPTURE_CHARS = 50_000;
export function resolveConfig(raw) {
    const merged = { ...DEFAULT_CONFIG, ...(raw ?? {}) };
    return {
        pythonPath: typeof merged.pythonPath === "string" && merged.pythonPath.length > 0
            ? merged.pythonPath
            : DEFAULT_CONFIG.pythonPath,
        brain: typeof merged.brain === "string" && BRAIN_NAME_RE.test(merged.brain)
            ? merged.brain
            : DEFAULT_CONFIG.brain,
        autoContext: typeof merged.autoContext === "boolean"
            ? merged.autoContext
            : DEFAULT_CONFIG.autoContext,
        autoCapture: typeof merged.autoCapture === "boolean"
            ? merged.autoCapture
            : DEFAULT_CONFIG.autoCapture,
        contextDepth: typeof merged.contextDepth === "number" &&
            Number.isInteger(merged.contextDepth) &&
            merged.contextDepth >= 0 &&
            merged.contextDepth <= 3
            ? merged.contextDepth
            : DEFAULT_CONFIG.contextDepth,
        maxContextTokens: typeof merged.maxContextTokens === "number" &&
            Number.isInteger(merged.maxContextTokens) &&
            merged.maxContextTokens >= 100 &&
            merged.maxContextTokens <= 10_000
            ? merged.maxContextTokens
            : DEFAULT_CONFIG.maxContextTokens,
        timeout: typeof merged.timeout === "number" &&
            Number.isFinite(merged.timeout) &&
            merged.timeout >= 5_000 &&
            merged.timeout <= 120_000
            ? merged.timeout
            : DEFAULT_CONFIG.timeout,
        initTimeout: typeof merged.initTimeout === "number" &&
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
const mcpClients = new Map();
function getOrCreateMcpClient(cfg, logger) {
    const key = `${cfg.pythonPath}::${cfg.brain}`;
    const existing = mcpClients.get(key);
    if (existing) {
        logger.info(`Reusing existing MCP client for brain "${cfg.brain}"`);
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
const plugin = {
    id: "neuralmemory",
    name: "NeuralMemory",
    description: "Brain-inspired persistent memory for AI agents — neurons, synapses, and fibers",
    version: "1.15.0",
    kind: "memory",
    register(api) {
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
        api.logger.info(`Registered ${registeredTools.length} NeuralMemory tools + ${compatTools.length} compat shims (sync)`);
        // ── Service: MCP process lifecycle ───────────────────
        api.registerService({
            id: "neuralmemory-mcp",
            async start() {
                if (!mcp.connected) {
                    try {
                        await mcp.connect();
                        api.logger.info("NeuralMemory MCP connected in service.start()");
                        // Log discovered tools for diagnostics (cannot re-register
                        // after register() — OpenClaw freezes the tool list).
                        try {
                            const dynamicTools = await createToolsFromMcp(mcp);
                            api.logger.info(`NeuralMemory MCP discovered ${dynamicTools.length} tools`);
                        }
                        catch (err) {
                            api.logger.warn(`Tool discovery failed: ${err.message}`);
                        }
                    }
                    catch (err) {
                        api.logger.error(`Failed to start NeuralMemory MCP: ${err.message}`);
                        throw err;
                    }
                }
            },
            async stop() {
                // Remove from singleton pool so next register() creates fresh client
                const key = `${cfg.pythonPath}::${cfg.brain}`;
                mcpClients.delete(key);
                await mcp.close();
                api.logger.info("NeuralMemory MCP service stopped");
            },
        });
        // ── Hook: tool awareness + auto-context before agent start ───
        api.on("before_agent_start", async (event, _ctx) => {
            const result = {
                systemPrompt: buildToolInstructions(registeredTools),
            };
            if (cfg.autoContext && mcp.connected) {
                const ev = event;
                try {
                    const raw = await mcp.callTool("nmem_recall", {
                        query: ev.prompt,
                        depth: cfg.contextDepth,
                        max_tokens: cfg.maxContextTokens,
                    });
                    const data = JSON.parse(raw);
                    if (data.answer && (data.confidence ?? 0) > 0.1) {
                        result.prependContext = `[NeuralMemory — relevant context]\n${data.answer}`;
                    }
                }
                catch (err) {
                    api.logger.warn(`Auto-context failed: ${err.message}`);
                }
            }
            return result;
        }, { priority: 10 });
        // ── Hook: auto-capture after agent completes ────────
        if (cfg.autoCapture) {
            api.on("agent_end", async (event, _ctx) => {
                if (!mcp.connected)
                    return;
                const ev = event;
                if (!ev.success)
                    return;
                try {
                    const messages = ev.messages?.slice(-5) ?? [];
                    const text = messages
                        .filter((m) => typeof m === "object" &&
                        m !== null &&
                        m.role === "assistant" &&
                        typeof m.content === "string")
                        .map((m) => m.content)
                        .join("\n")
                        .slice(0, MAX_AUTO_CAPTURE_CHARS);
                    if (text.length > 50) {
                        await mcp.callTool("nmem_auto", {
                            action: "process",
                            text,
                        });
                    }
                }
                catch (err) {
                    api.logger.warn(`Auto-capture failed: ${err.message}`);
                }
            }, { priority: 90 });
        }
        // ── Done ────────────────────────────────────────────
        api.logger.info(`NeuralMemory registered (brain: ${cfg.brain}, ` +
            `autoContext: ${cfg.autoContext}, autoCapture: ${cfg.autoCapture}) — ` +
            `tools will be loaded dynamically from MCP on service start`);
    },
};
export default plugin;
//# sourceMappingURL=index.js.map