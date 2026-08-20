"""Handing the catalog to the official MCP SDK (SPEC §13).

The SDK is reached by injected import, exactly as the harness reaches its client
half: agentdb imports and its tests run with ``mcp`` uninstalled, and only
actually serving a client needs the optional extra. Everything with logic in it
lives in :mod:`agentdb.server.app`, so this module stays small enough to read in
one sitting — which is the only real defence for a layer whose counterpart is a
vendor object.

The surface used here is the ``mcp`` 2.x low-level server, which targets the
2026-07-28 revision: handlers are constructor arguments taking a request context
and typed params, not the 1.x decorators. Only stdio is wired. Streamable HTTP
is the remote transport and it drags in the OAuth 2.1 resource-server work of
SPEC §13.3; shipping it without JWKS verification and DNS-rebinding protection
would be worse than not shipping it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any, Final

from agentdb import __version__
from agentdb.server.app import ToolCatalog

Importer = Callable[[str], ModuleType]

SDK: Final = "mcp.server.lowlevel"
STDIO: Final = "mcp.server.stdio"
TYPES: Final = "mcp.types"

SERVER_NAME: Final = "agentdb"


class TransportError(RuntimeError):
    """The SDK is missing, or the transport could not be opened."""


def tool_handlers(catalog: ToolCatalog, types: Any) -> tuple[Any, Any]:
    """The ``on_list_tools`` and ``on_call_tool`` pair, built against ``types``.

    Split out from :func:`build_server` so a test holding the real ``mcp.types``
    can await them and let the vendor's own models validate what we produce.
    That is the part worth checking: the wiring is trivial, and the field names
    on the vendor's models are exactly what a version bump moves.

    A tool that fails comes back as a result with ``is_error`` set, not as a
    raised exception: the failure is information about the engine, and the agent
    on the other end can act on it. ``structured_content`` is left off those
    results on purpose — it is the field a client is promised will match the
    tool's ``outputSchema``, and an error object does not.
    """

    async def on_list_tools(_context: Any, _params: Any) -> Any:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=tool.name,
                    title=tool.title,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    output_schema=tool.output_schema,
                )
                for tool in catalog.tools
            ]
        )

    async def on_call_tool(_context: Any, params: Any) -> Any:
        response = await catalog.call(params.name, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=response.text)],
            structured_content=None if response.is_error else response.structured,
            is_error=response.is_error,
        )

    return on_list_tools, on_call_tool


def build_server(catalog: ToolCatalog, *, importer: Importer = importlib.import_module) -> Any:
    """An SDK ``Server`` serving ``catalog``."""
    lowlevel = _import(importer, SDK)
    on_list_tools, on_call_tool = tool_handlers(catalog, _import(importer, TYPES))
    return lowlevel.Server(
        SERVER_NAME,
        version=__version__,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def serve_stdio(
    catalog: ToolCatalog, *, importer: Importer = importlib.import_module
) -> None:
    """Serve ``catalog`` over stdio until the client disconnects."""
    stdio = _import(importer, STDIO)
    server = build_server(catalog, importer=importer)
    async with stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _import(importer: Importer, module: str) -> Any:
    try:
        return importer(module)
    except ImportError as exc:
        raise TransportError(
            f"the 'mcp' package is not installed, so {module} could not be imported; "
            "install the optional extra with: uv sync --extra mcp"
        ) from exc
