#!/usr/bin/env node

/**
 * neural-memory-mcp — thin npm wrapper for the Python MCP server.
 *
 * Usage:
 *   npx neural-memory-mcp          # runs nmem-mcp (stdio transport)
 *   npx neural-memory-mcp --help   # shows help
 *
 * Requires: pip install neural-memory
 */

import { spawn } from "node:child_process";

const args = process.argv.slice(2);

// Try nmem-mcp first (installed via pip), fall back to python -m
const child = spawn("nmem-mcp", args, {
  stdio: "inherit",
  shell: true,
});

child.on("error", () => {
  // Fallback: python -m neural_memory.mcp
  const fallback = spawn("python", ["-m", "neural_memory.mcp", ...args], {
    stdio: "inherit",
    shell: true,
  });

  fallback.on("error", () => {
    process.stderr.write(
      "Error: neural-memory not found.\n" +
      "Install: pip install neural-memory\n" +
      "Docs: https://neuralmemory.theio.vn\n"
    );
    process.exit(1);
  });

  fallback.on("exit", (code) => process.exit(code ?? 1));
});

child.on("exit", (code) => process.exit(code ?? 0));
