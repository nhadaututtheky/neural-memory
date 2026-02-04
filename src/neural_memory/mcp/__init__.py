"""MCP (Model Context Protocol) server for NeuralMemory.

This module provides an MCP server that exposes NeuralMemory tools
to Claude Code, Claude Desktop, and other MCP-compatible clients.
"""

from neural_memory.mcp.server import create_mcp_server, main, run_mcp_server

__all__ = ["create_mcp_server", "main", "run_mcp_server"]
