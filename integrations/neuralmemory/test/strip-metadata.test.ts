import { describe, it, expect } from "vitest";
import { stripPromptMetadata } from "../src/index.js";

describe("stripPromptMetadata", () => {
  it("returns plain text unchanged", () => {
    expect(stripPromptMetadata("Hello, how are you?")).toBe(
      "Hello, how are you?",
    );
  });

  it("strips [NeuralMemory — ...] context blocks", () => {
    const input = `[NeuralMemory — relevant context]
## Relevant Memories

- [concept] some concept
- [entity] Tyler

## Related Information

- [concept] another concept

Actual user message here`;

    const result = stripPromptMetadata(input);
    expect(result).toBe("Actual user message here");
  });

  it("strips JSON metadata blocks", () => {
    const input = `{
  "conversation": { "chat_id": 123 },
  "sender": { "name": "Tyler" }
}

What is the weather today?`;

    const result = stripPromptMetadata(input);
    expect(result).toBe("What is the weather today?");
  });

  it("strips [Subagent Context] and [Subagent Task] blocks", () => {
    const input = `[Subagent Context] You are running as a subagent (depth 1/1).

[Subagent Task]: Fix the bug in index.ts

Please fix issue #104`;

    const result = stripPromptMetadata(input);
    expect(result).toBe("Please fix issue #104");
  });

  it("strips timestamp lines", () => {
    const input = `[Mon 2026-03-23 18:47 GMT+11] some context info

Do the thing`;

    const result = stripPromptMetadata(input);
    expect(result).toBe("Do the thing");
  });

  it("strips IMPORTANT: lines and export lines", () => {
    const input = `IMPORTANT: Use curl + GitHub REST API, not gh CLI.

export GH_TOKEN="abc123"

Fix the bug please`;

    const result = stripPromptMetadata(input);
    expect(result).toBe("Fix the bug please");
  });

  it("strips media attachment lines", () => {
    const input = `[image] photo_123.jpg
📎 document.pdf

What does this show?`;

    const result = stripPromptMetadata(input);
    expect(result).toBe("What does this show?");
  });

  it("strips URL-only lines", () => {
    const input = `URL: https://github.com/example/repo/issues/1
https://example.com/foo

Check this issue`;

    const result = stripPromptMetadata(input);
    expect(result).toBe("Check this issue");
  });

  it("falls back to last non-empty line when all content stripped", () => {
    const input = `[NeuralMemory — context]
- [concept] only metadata here`;

    const result = stripPromptMetadata(input);
    // Should return something non-empty
    expect(result.length).toBeGreaterThan(0);
  });

  it("handles a realistic full prompt with all metadata types", () => {
    const input = `[NeuralMemory — relevant context]
## Relevant Memories

- [concept] OpenClaw
- [entity] Tyler là Telegram

## Related Information

- [concept] brain maiai

[Mon 2026-03-23 18:47 GMT+11] [Subagent Context] You are running as a subagent (depth 1/1).

[Subagent Task]: Fix GitHub issue #104

IMPORTANT: Use curl + GitHub REST API, not gh CLI.

export GH_TOKEN="gho_abc123"

URL: https://github.com/nhadaututtheky/neural-memory/issues/104

Can you fix the autoContext recall bug?`;

    const result = stripPromptMetadata(input);
    expect(result).toContain("fix the autoContext recall bug");
  });
});
