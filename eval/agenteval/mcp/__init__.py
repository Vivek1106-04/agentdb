"""MCP client surface. agenteval speaks to servers; it never implements one."""

from agenteval.mcp.base import McpError, McpSession, ToolResult, ToolSpec
from agenteval.mcp.config import (
    McpConfigError,
    McpServerConfig,
    load_servers,
    parse_server,
)
from agenteval.mcp.stdio import StdioSession, connect

__all__ = [
    "McpConfigError",
    "McpError",
    "McpServerConfig",
    "McpSession",
    "StdioSession",
    "ToolResult",
    "ToolSpec",
    "connect",
    "load_servers",
    "parse_server",
]
