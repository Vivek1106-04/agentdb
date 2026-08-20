"""The tool catalog and its dispatch (SPEC §13).

No MCP SDK is imported here and no transport is opened. The catalog is a plain
object: it lists tools and it calls one. That is what lets the contract tests
prove every ``outputSchema`` against a real response with nothing running, and
it is why adding a second transport later is wiring rather than a rewrite.

MCP's 2026-07-28 core is stateless, so every tool is independently callable and
the catalog holds no per-conversation state. The only thing it keeps between
calls is the adapter, which is a connection, not a session.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from agentdb.adapters import Adapter, AdapterError
from agentdb.config import Config
from agentdb.server.base import ServerContext, ToolDef, ToolError
from agentdb.server.schemas import JsonValue
from agentdb.server.tools import all_tools

__all__ = ["ToolCatalog", "ToolDef", "ToolError", "ToolResponse", "build_catalog"]


@dataclass(frozen=True, slots=True)
class ToolResponse:
    """One tool result, in both the shapes a client may read.

    ``structured`` is what a client with a schema does; ``text`` is what a
    client without one does. They are generated from one value, so a client
    cannot get two different answers depending on which field it trusts.
    """

    structured: dict[str, JsonValue]
    is_error: bool = False

    @property
    def text(self) -> str:
        return json.dumps(self.structured, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    """Every tool this server serves, and the one way to call one."""

    tools: tuple[ToolDef, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    def get(self, name: str) -> ToolDef:
        """The tool called ``name``, or a ``ToolError`` listing what does exist."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise ToolError(
            f"no such tool: {name!r}",
            suggestion=f"available tools: {', '.join(self.names)}",
        )

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> ToolResponse:
        """Invoke ``name``, turning every expected failure into a structured result.

        Tool-level failures come back as results rather than exceptions because
        that is what an agent can act on: a bad argument or an unsupported
        capability is information about the engine, and a traceback across a
        protocol boundary is not. Unexpected failures are left to propagate —
        swallowing those would turn a bug into a quiet wrong answer.
        """
        try:
            return ToolResponse(structured=await self.get(name).handler(arguments))
        except ToolError as exc:
            return ToolResponse(structured=exc.as_dict(), is_error=True)
        except AdapterError as exc:
            structured: dict[str, JsonValue] = dict(exc.as_dict())
            return ToolResponse(structured=structured, is_error=True)


def build_catalog(adapter: Adapter, config: Config | None = None) -> ToolCatalog:
    """The catalog for one engine connection."""
    context = ServerContext(adapter=adapter, config=config or Config())
    return ToolCatalog(tools=all_tools(context))
