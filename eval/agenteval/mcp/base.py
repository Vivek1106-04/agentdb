"""What agenteval needs from an MCP server, and nothing more.

The harness scores servers it did not write. It therefore models a server as
three things — the tools it advertises, a way to call one, and a way to shut it
down — and refuses to know anything else. Every assumption beyond that would be
an assumption about a specific vendor's server, and the moment the harness has
one of those it stops being a scoreboard.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class McpError(RuntimeError):
    """A server could not be reached, started, or spoken to."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool as the server advertises it."""

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What came back from one tool call.

    ``is_error`` is the server's own judgement, kept separate from transport
    failures: a tool that reports a bad argument is data about the system under
    test, while a dead server is a hole in the run.
    """

    content: str
    is_error: bool = False


class McpSession(Protocol):
    """A live connection to one MCP server."""

    async def list_tools(self) -> tuple[ToolSpec, ...]:
        """Every tool the server offers, in the order it offers them."""
        ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Invoke one tool. Tool-level failures come back, not up."""
        ...

    async def close(self) -> None:
        """Shut the server down. Called even when a run fails."""
        ...
