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
import type { OpenClawPluginDefinition } from "./types.js";
type PluginConfig = {
    pythonPath: string;
    brain: string;
    autoContext: boolean;
    autoCapture: boolean;
    contextDepth: number;
    maxContextTokens: number;
    timeout: number;
    initTimeout: number;
};
export declare const BRAIN_NAME_RE: RegExp;
export declare const MAX_AUTO_CAPTURE_CHARS = 50000;
export declare function resolveConfig(raw?: Record<string, unknown>): PluginConfig;
declare const plugin: OpenClawPluginDefinition;
export default plugin;
