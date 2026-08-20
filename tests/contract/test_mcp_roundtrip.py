"""One real MCP conversation, client to server, over in-memory streams.

Everything else in this suite tests a half: the catalog with no protocol, or the
transport with a stand-in SDK. This test runs the vendor's own client against
the vendor's own server with agentdb's tools inside it, so the handshake, the
tool listing, ``structuredContent`` and the error path are all proved end to end
against the wire format rather than against our belief about it.

No engine is involved — the adapter is the same fake the rest of the suite uses.
What is under test is the protocol layer, and an engine would only make it slow.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentdb.server.transports import build_server
from tests.server_fakes import clickhouse_catalog


async def test_a_client_can_list_and_call_agentdbs_tools() -> None:
    anyio = pytest.importorskip("anyio")
    client_module = pytest.importorskip("mcp")
    memory = pytest.importorskip("mcp.shared.memory")

    catalog, _ = clickhouse_catalog()
    server = build_server(catalog)
    seen: dict[str, Any] = {}

    async def conversation() -> None:
        async with (
            memory.create_client_server_memory_streams() as (client, engine),
            anyio.create_task_group() as tasks,
        ):
            tasks.start_soon(_run, server, engine)
            async with client_module.ClientSession(*client) as session:
                await session.initialize()
                listing = await session.list_tools()
                seen["tools"] = [tool.name for tool in listing.tools]
                seen["schema"] = listing.tools[0].output_schema

                seen["layout"] = await session.call_tool(
                    "physical_layout", {"relation": "agentdb.hits"}
                )
                seen["failed"] = await session.call_tool("describe_relation", {})
            tasks.cancel_scope.cancel()

    await conversation()

    assert seen["tools"] == list(catalog.names)
    assert seen["schema"]["$schema"].endswith("2020-12/schema")

    layout = seen["layout"]
    assert layout.is_error is False
    assert layout.structured_content["order_by"] == ["CounterID", "EventDate", "UserID"]

    failed = seen["failed"]
    assert failed.is_error is True
    assert "'relation' is required" in failed.content[0].text


async def _run(server: Any, streams: tuple[Any, Any]) -> None:
    read_stream, write_stream = streams
    await server.run(
        read_stream,
        write_stream,
        server.create_initialization_options(),
        raise_exceptions=True,
    )
