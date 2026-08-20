"""Talking to an MCP server over stdio, through the official SDK.

Field names follow the ``mcp`` 2.x models (``input_schema``, ``is_error``). The
1.x camelCase attributes are gone, and reading a missing ``isError`` with a
default of ``False`` would have scored every failed tool call as a success —
a benchmark reporting a number that is quietly too high.

The SDK is reached by injected import, like every other vendor dependency here,
so the harness imports and its tests run without it. Only a Family S run needs
it installed.

The SDK models a session as nested async context managers, which suits a script
and not a benchmark that must keep one server alive across many tasks. An
``AsyncExitStack`` holds them open, and :meth:`StdioSession.close` unwinds it —
including when the run fails, because a leaked server process quietly poisons
the next arm's timings.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from agenteval.mcp.base import McpError, McpSession, ToolResult, ToolSpec
from agenteval.mcp.config import McpServerConfig

Importer = Callable[[str], ModuleType]

SDK = "mcp"


@dataclass(frozen=True, slots=True)
class StdioSession:
    """An :class:`McpSession` over one child process."""

    session: Any
    stack: AsyncExitStack

    async def list_tools(self) -> tuple[ToolSpec, ...]:
        listing = await self.session.list_tools()
        return tuple(
            ToolSpec(
                name=str(tool.name),
                description=str(tool.description or ""),
                input_schema=dict(tool.input_schema or {}),
            )
            for tool in listing.tools
        )

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Invoke a tool, flattening its content blocks into text.

        A transport failure is raised; a tool that *reports* failure comes back
        as a result, because that is a measurement of the system under test.
        """
        try:
            outcome = await self.session.call_tool(name, dict(arguments))
        except Exception as exc:
            raise McpError(f"calling tool {name!r} failed: {exc}") from exc

        return ToolResult(
            content=_flatten(outcome.content),
            is_error=bool(getattr(outcome, "is_error", False)),
        )

    async def close(self) -> None:
        await self.stack.aclose()


async def connect(
    config: McpServerConfig,
    *,
    environ: Mapping[str, str] | None = None,
    importer: Importer = importlib.import_module,
) -> McpSession:
    """Launch ``config``'s server and complete the MCP handshake."""
    env = config.resolve_env(os.environ if environ is None else environ)

    try:
        sdk = importer(SDK)
        stdio = importer("mcp.client.stdio")
    except ImportError as exc:
        raise McpError(
            "the 'mcp' package is not installed; install the optional extra "
            "with: uv sync --extra mcp"
        ) from exc

    parameters = sdk.StdioServerParameters(command=config.command, args=list(config.args), env=env)
    stack = AsyncExitStack()
    try:
        streams = await stack.enter_async_context(stdio.stdio_client(parameters))
        session = await stack.enter_async_context(sdk.ClientSession(*streams))
        await session.initialize()
    except Exception as exc:
        await stack.aclose()
        raise McpError(f"could not start MCP server {config.name!r}: {exc}") from exc

    return StdioSession(session=session, stack=stack)


def _flatten(content: object) -> str:
    """Join the text blocks of a tool result, ignoring blocks that carry none."""
    if not isinstance(content, list):
        return str(content)
    parts = [str(block.text) for block in content if hasattr(block, "text")]
    return "\n".join(parts)
