import { describe, it, expect, vi } from "vitest";
import { z } from "zod";
import { createTools, type ToolDefinition } from "../src/tools.js";
import type { NeuralMemoryMcpClient } from "../src/mcp-client.js";

function makeMockMcp(
  overrides: Partial<NeuralMemoryMcpClient> = {},
): NeuralMemoryMcpClient {
  return {
    connected: true,
    callTool: vi.fn().mockResolvedValue("{}"),
    connect: vi.fn(),
    close: vi.fn(),
    ...overrides,
  } as unknown as NeuralMemoryMcpClient;
}

describe("createTools", () => {
  it("returns 6 tools", () => {
    const tools = createTools(makeMockMcp());
    expect(tools).toHaveLength(6);
  });

  it("returns tools with expected names", () => {
    const tools = createTools(makeMockMcp());
    const names = tools.map((t) => t.name);
    expect(names).toEqual([
      "nmem_remember",
      "nmem_recall",
      "nmem_context",
      "nmem_todo",
      "nmem_stats",
      "nmem_health",
    ]);
  });

  it("each tool has name, description, parameters, and execute", () => {
    const tools = createTools(makeMockMcp());
    for (const tool of tools) {
      expect(typeof tool.name).toBe("string");
      expect(typeof tool.description).toBe("string");
      expect(tool.parameters).toBeDefined();
      expect(typeof tool.execute).toBe("function");
    }
  });
});

describe("tool schemas", () => {
  let tools: ToolDefinition[];

  function findTool(name: string): ToolDefinition {
    const tool = tools.find((t) => t.name === name);
    if (!tool) throw new Error(`Tool ${name} not found`);
    return tool;
  }

  function parse(tool: ToolDefinition, input: unknown): unknown {
    return (tool.parameters as z.ZodTypeAny).parse(input);
  }

  function safeParse(
    tool: ToolDefinition,
    input: unknown,
  ): z.SafeParseReturnType<unknown, unknown> {
    return (tool.parameters as z.ZodTypeAny).safeParse(input);
  }

  tools = createTools(makeMockMcp());

  describe("nmem_remember", () => {
    it("accepts valid input", () => {
      const tool = findTool("nmem_remember");
      const result = parse(tool, { content: "hello world" });
      expect(result).toEqual({ content: "hello world" });
    });

    it("accepts full input with all optional fields", () => {
      const tool = findTool("nmem_remember");
      const result = parse(tool, {
        content: "test",
        type: "fact",
        priority: 5,
        tags: ["tag1", "tag2"],
        expires_days: 30,
      });
      expect(result).toHaveProperty("content", "test");
      expect(result).toHaveProperty("type", "fact");
    });

    it("rejects content over 100K chars", () => {
      const tool = findTool("nmem_remember");
      const result = safeParse(tool, { content: "x".repeat(100_001) });
      expect(result.success).toBe(false);
    });

    it("rejects priority out of range", () => {
      const tool = findTool("nmem_remember");
      expect(safeParse(tool, { content: "x", priority: -1 }).success).toBe(
        false,
      );
      expect(safeParse(tool, { content: "x", priority: 11 }).success).toBe(
        false,
      );
    });

    it("rejects invalid type enum", () => {
      const tool = findTool("nmem_remember");
      const result = safeParse(tool, { content: "x", type: "invalid" });
      expect(result.success).toBe(false);
    });
  });

  describe("nmem_recall", () => {
    it("accepts valid query", () => {
      const tool = findTool("nmem_recall");
      const result = parse(tool, { query: "search term" });
      expect(result).toEqual({ query: "search term" });
    });

    it("rejects query over 10K chars", () => {
      const tool = findTool("nmem_recall");
      const result = safeParse(tool, { query: "x".repeat(10_001) });
      expect(result.success).toBe(false);
    });

    it("rejects depth out of range", () => {
      const tool = findTool("nmem_recall");
      expect(
        safeParse(tool, { query: "x", depth: -1 }).success,
      ).toBe(false);
      expect(
        safeParse(tool, { query: "x", depth: 4 }).success,
      ).toBe(false);
    });
  });

  describe("nmem_context", () => {
    it("accepts empty object", () => {
      const tool = findTool("nmem_context");
      const result = parse(tool, {});
      expect(result).toEqual({});
    });

    it("rejects limit over 200", () => {
      const tool = findTool("nmem_context");
      const result = safeParse(tool, { limit: 201 });
      expect(result.success).toBe(false);
    });
  });

  describe("nmem_todo", () => {
    it("accepts valid task", () => {
      const tool = findTool("nmem_todo");
      const result = parse(tool, { task: "do this" });
      expect(result).toEqual({ task: "do this" });
    });

    it("rejects task over 10K chars", () => {
      const tool = findTool("nmem_todo");
      const result = safeParse(tool, { task: "x".repeat(10_001) });
      expect(result.success).toBe(false);
    });
  });

  describe("nmem_stats / nmem_health", () => {
    it("accepts empty object", () => {
      expect(parse(findTool("nmem_stats"), {})).toEqual({});
      expect(parse(findTool("nmem_health"), {})).toEqual({});
    });
  });
});

describe("tool execution", () => {
  it("returns error when service not connected", async () => {
    const mcp = makeMockMcp({ connected: false });
    const tools = createTools(mcp);
    const result = await tools[0].execute({ content: "test" });
    expect(result).toEqual({
      error: true,
      message: "NeuralMemory service not running. Start the service first.",
    });
  });

  it("catches callTool exceptions → structured error", async () => {
    const mcp = makeMockMcp({
      callTool: vi.fn().mockRejectedValue(new Error("connection lost")),
    });
    const tools = createTools(mcp);
    const result = await tools[0].execute({ content: "test" });
    expect(result).toEqual({
      error: true,
      message: "Tool nmem_remember failed: connection lost",
    });
  });

  it("parses JSON response correctly", async () => {
    const mcp = makeMockMcp({
      callTool: vi
        .fn()
        .mockResolvedValue('{"answer": "hello", "confidence": 0.9}'),
    });
    const tools = createTools(mcp);
    const result = await tools[1].execute({ query: "test" });
    expect(result).toEqual({ answer: "hello", confidence: 0.9 });
  });

  it("handles non-JSON response → {text: raw}", async () => {
    const mcp = makeMockMcp({
      callTool: vi.fn().mockResolvedValue("plain text response"),
    });
    const tools = createTools(mcp);
    const result = await tools[0].execute({ content: "test" });
    expect(result).toEqual({ text: "plain text response" });
  });

  it("passes correct tool name and args to callTool", async () => {
    const callTool = vi.fn().mockResolvedValue("{}");
    const mcp = makeMockMcp({ callTool });
    const tools = createTools(mcp);

    await tools[0].execute({ content: "remember this", priority: 5 });
    expect(callTool).toHaveBeenCalledWith("nmem_remember", {
      content: "remember this",
      priority: 5,
    });
  });
});
