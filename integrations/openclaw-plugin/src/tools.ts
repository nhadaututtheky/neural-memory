/**
 * NeuralMemory tool definitions for OpenClaw.
 *
 * Each tool proxies to the MCP server via JSON-RPC.
 * Uses zod for parameter schemas (peer dependency from OpenClaw runtime).
 *
 * Registers 6 core tools:
 *   nmem_remember  — Store a memory
 *   nmem_recall    — Query/search memories
 *   nmem_context   — Get recent context
 *   nmem_todo      — Quick TODO shortcut
 *   nmem_stats     — Brain statistics
 *   nmem_health    — Brain health diagnostics
 */

import { z } from "zod";
import type { NeuralMemoryMcpClient } from "./mcp-client.js";

// ── Types ──────────────────────────────────────────────────

export type ToolDefinition = {
  readonly name: string;
  readonly description: string;
  readonly parameters: z.ZodTypeAny;
  readonly execute: (args: Record<string, unknown>) => Promise<unknown>;
};

// ── Tool factory ───────────────────────────────────────────

export function createTools(mcp: NeuralMemoryMcpClient): ToolDefinition[] {
  const call = async (
    toolName: string,
    args: Record<string, unknown>,
  ): Promise<unknown> => {
    if (!mcp.connected) {
      return {
        error: true,
        message: "NeuralMemory service not running. Start the service first.",
      };
    }

    const raw = await mcp.callTool(toolName, args);

    try {
      return JSON.parse(raw);
    } catch {
      return { text: raw };
    }
  };

  return [
    {
      name: "nmem_remember",
      description:
        "Store a memory in NeuralMemory. Use this to remember facts, decisions, " +
        "insights, todos, errors, and other information that should persist across sessions.",
      parameters: z.object({
        content: z
          .string()
          .describe("The content to remember"),
        type: z
          .enum([
            "fact",
            "decision",
            "preference",
            "todo",
            "insight",
            "context",
            "instruction",
            "error",
            "workflow",
            "reference",
          ])
          .optional()
          .describe("Memory type (auto-detected if not specified)"),
        priority: z
          .number()
          .int()
          .min(0)
          .max(10)
          .optional()
          .describe("Priority 0-10 (5=normal, 10=critical)"),
        tags: z
          .array(z.string())
          .optional()
          .describe("Tags for categorization"),
        expires_days: z
          .number()
          .int()
          .optional()
          .describe("Days until memory expires"),
      }),
      execute: (args) => call("nmem_remember", args),
    },

    {
      name: "nmem_recall",
      description:
        "Query memories from NeuralMemory. Use this to recall past information, " +
        "decisions, patterns, or context relevant to the current task.",
      parameters: z.object({
        query: z
          .string()
          .describe("The query to search memories"),
        depth: z
          .number()
          .int()
          .min(0)
          .max(3)
          .optional()
          .describe(
            "Search depth: 0=instant, 1=context, 2=habit, 3=deep",
          ),
        max_tokens: z
          .number()
          .int()
          .min(1)
          .max(10000)
          .optional()
          .describe("Maximum tokens in response (default: 500)"),
        min_confidence: z
          .number()
          .min(0)
          .max(1)
          .optional()
          .describe("Minimum confidence threshold"),
      }),
      execute: (args) => call("nmem_recall", args),
    },

    {
      name: "nmem_context",
      description:
        "Get recent context from NeuralMemory. Use this at the start of " +
        "tasks to inject relevant recent memories.",
      parameters: z.object({
        limit: z
          .number()
          .int()
          .optional()
          .describe("Number of recent memories (default: 10)"),
        fresh_only: z
          .boolean()
          .optional()
          .describe("Only include memories less than 30 days old"),
      }),
      execute: (args) => call("nmem_context", args),
    },

    {
      name: "nmem_todo",
      description:
        "Quick shortcut to add a TODO memory with 30-day expiry.",
      parameters: z.object({
        task: z
          .string()
          .describe("The task to remember"),
        priority: z
          .number()
          .int()
          .min(0)
          .max(10)
          .optional()
          .describe("Priority 0-10 (default: 5)"),
      }),
      execute: (args) => call("nmem_todo", args),
    },

    {
      name: "nmem_stats",
      description:
        "Get brain statistics including memory counts and freshness.",
      parameters: z.object({}),
      execute: (args) => call("nmem_stats", args),
    },

    {
      name: "nmem_health",
      description:
        "Get brain health diagnostics including grade, purity score, " +
        "and recommendations.",
      parameters: z.object({}),
      execute: (args) => call("nmem_health", args),
    },
  ];
}
