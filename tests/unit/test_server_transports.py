"""Wiring the catalog to the MCP SDK.

Two halves. The stand-in SDK proves the module works with ``mcp`` uninstalled,
which is the property that keeps importing agentdb free of a vendor dependency.
The real SDK then validates what the handlers actually produce, because a fake
can only ever agree with whatever the author believed the vendor's field names
were — and a stand-in that agreed with a belief one major version out of date is
precisely how this module was wrong the first time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from agentdb.server.transports import (
    TransportError,
    build_server,
    serve_stdio,
    tool_handlers,
)
from tests.server_fakes import clickhouse_catalog


@dataclass
class FakeServer:
    """Stands in for the SDK's low-level ``Server``, which takes its handlers up front."""

    name: str
    version: str
    on_list_tools: Any
    on_call_tool: Any
    ran: list[tuple[Any, ...]] = field(default_factory=list)

    def create_initialization_options(self) -> str:
        return "options"

    async def run(self, read: Any, write: Any, options: Any) -> None:
        self.ran.append((read, write, options))


class FakeSdk(ModuleType):
    """Stands in for ``mcp.server.lowlevel``, ``mcp.types`` and ``mcp.server.stdio``."""

    def __init__(self) -> None:
        super().__init__("fake")
        self.server: FakeServer | None = None

    def Server(self, name: str, **fields: Any) -> FakeServer:  # noqa: N802 - the SDK's own name
        self.server = FakeServer(name=name, **fields)
        return self.server

    def ListToolsResult(self, **fields: Any) -> dict[str, Any]:  # noqa: N802
        return fields

    def CallToolResult(self, **fields: Any) -> dict[str, Any]:  # noqa: N802
        return fields

    def Tool(self, **fields: Any) -> dict[str, Any]:  # noqa: N802
        return fields

    def TextContent(self, **fields: Any) -> dict[str, Any]:  # noqa: N802
        return fields

    @asynccontextmanager
    async def stdio_server(self) -> AsyncIterator[tuple[str, str]]:
        yield ("read", "write")


def _sdk() -> tuple[FakeSdk, Callable[[str], ModuleType]]:
    module = FakeSdk()
    return module, lambda _: module


def _params(name: str, arguments: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments)


async def test_every_tool_is_advertised_with_both_of_its_schemas() -> None:
    catalog, _ = clickhouse_catalog()
    module, importer = _sdk()

    server = build_server(catalog, importer=importer)
    listing = await server.on_list_tools(None, None)

    tools = listing["tools"]
    assert [tool["name"] for tool in tools] == list(catalog.names)
    assert all(tool["input_schema"] and tool["output_schema"] for tool in tools)
    assert module.server is not None


async def test_a_successful_call_returns_both_text_and_structured_content() -> None:
    catalog, _ = clickhouse_catalog()
    _, importer = _sdk()

    server = build_server(catalog, importer=importer)
    result = await server.on_call_tool(None, _params("list_namespaces"))

    assert result["is_error"] is False
    assert result["structured_content"]["namespaces"] == ["agentdb"]
    assert result["content"][0]["text"] == '{"engine": "clickhouse", "namespaces": ["agentdb"]}'


async def test_a_failed_call_carries_no_structured_content() -> None:
    """That field is promised to match the outputSchema; an error object does not."""
    catalog, _ = clickhouse_catalog()
    _, importer = _sdk()

    server = build_server(catalog, importer=importer)
    result = await server.on_call_tool(None, _params("describe_relation", {}))

    assert result["is_error"] is True
    assert result["structured_content"] is None
    assert "is required" in result["content"][0]["text"]


async def test_serving_over_stdio_runs_the_server_on_the_transport_streams() -> None:
    catalog, _ = clickhouse_catalog()
    module, importer = _sdk()

    await serve_stdio(catalog, importer=importer)

    assert module.server is not None
    assert module.server.ran == [("read", "write", "options")]


def test_a_missing_sdk_says_which_extra_to_install() -> None:
    catalog, _ = clickhouse_catalog()

    def missing(name: str) -> ModuleType:
        raise ImportError(name)

    with pytest.raises(TransportError, match="uv sync --extra mcp"):
        build_server(catalog, importer=missing)


# --- against the real SDK --------------------------------------------------


async def test_the_real_sdk_accepts_every_tool_we_advertise() -> None:
    """Field-name drift in the vendor's models fails here, not in production."""
    types = pytest.importorskip("mcp.types")
    catalog, _ = clickhouse_catalog()

    on_list_tools, _ = tool_handlers(catalog, types)
    listing = await on_list_tools(None, None)

    assert [tool.name for tool in listing.tools] == list(catalog.names)
    assert listing.tools[0].output_schema is not None


async def test_the_real_sdk_accepts_both_a_result_and_an_error() -> None:
    types = pytest.importorskip("mcp.types")
    catalog, _ = clickhouse_catalog()

    _, on_call_tool = tool_handlers(catalog, types)
    ok = await on_call_tool(None, _params("list_namespaces"))
    failed = await on_call_tool(None, _params("describe_relation", {}))

    assert ok.structured_content == {"engine": "clickhouse", "namespaces": ["agentdb"]}
    assert ok.is_error is False
    assert failed.is_error is True
    assert failed.structured_content is None


async def test_the_real_sdk_server_takes_our_handlers() -> None:
    pytest.importorskip("mcp.server.lowlevel")
    catalog, _ = clickhouse_catalog()

    server = build_server(catalog)

    assert server.name == "agentdb"
