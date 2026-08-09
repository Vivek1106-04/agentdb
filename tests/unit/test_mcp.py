"""The MCP client, exercised without the SDK and without launching a process."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from agenteval.mcp.base import McpError, ToolResult, ToolSpec
from agenteval.mcp.config import (
    REDACTED,
    McpConfigError,
    McpServerConfig,
    load_servers,
    parse_server,
)
from agenteval.mcp.stdio import StdioSession, connect

VALID = {
    "name": "S1_mcp_clickhouse",
    "version": "0.1.12",
    "command": "uvx",
    "args": ["mcp-clickhouse"],
    "env": {"CLICKHOUSE_HOST": "localhost"},
    "env_passthrough": ["CLICKHOUSE_PASSWORD"],
    "tools": ["run_select_query"],
}


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_a_server_config_parses_the_documented_shape() -> None:
    server = parse_server(VALID)

    assert server.name == "S1_mcp_clickhouse"
    assert server.version == "0.1.12"
    assert server.args == ("mcp-clickhouse",)
    assert server.tools == ("run_select_query",)


@pytest.mark.parametrize("missing", ["name", "version", "command"])
def test_the_identifying_fields_are_required(missing: str) -> None:
    payload = {k: v for k, v in VALID.items() if k != missing}

    with pytest.raises(McpConfigError, match=f"is missing: {missing}"):
        parse_server(payload)


@pytest.mark.parametrize(("field_name", "value"), [("name", ""), ("version", ""), ("command", "")])
def test_a_blank_identifying_field_is_refused(field_name: str, value: str) -> None:
    with pytest.raises(McpConfigError):
        parse_server({**VALID, field_name: value})


def test_unknown_fields_are_an_error_not_a_shrug() -> None:
    with pytest.raises(McpConfigError, match="unknown field\\(s\\): transport"):
        parse_server({**VALID, "transport": "sse"})


def test_a_string_where_a_list_belongs_is_refused() -> None:
    with pytest.raises(McpConfigError, match="expected a list of strings"):
        parse_server({**VALID, "args": "mcp-clickhouse"})


def test_secrets_are_read_from_the_environment_at_launch() -> None:
    server = parse_server(VALID)

    env = server.resolve_env({"CLICKHOUSE_PASSWORD": "hunter2"})

    assert env == {"CLICKHOUSE_HOST": "localhost", "CLICKHOUSE_PASSWORD": "hunter2"}


def test_a_missing_secret_fails_before_the_run_starts() -> None:
    server = parse_server(VALID)

    with pytest.raises(McpConfigError, match="needs environment variable\\(s\\) CLICKHOUSE_PASS"):
        server.resolve_env({})


def test_a_committed_config_names_secrets_but_never_their_values() -> None:
    # Arrange — this record goes into the report; a leaked password would ship
    record = parse_server(VALID).as_record()

    assert record["env_passthrough"] == {"CLICKHOUSE_PASSWORD": REDACTED}
    assert "hunter2" not in str(record)
    assert record["env"] == {"CLICKHOUSE_HOST": "localhost"}


def test_the_fingerprint_changes_with_the_pinned_version() -> None:
    # A benchmark of a moving beta is meaningless without saying which build
    first = parse_server(VALID).fingerprint
    second = parse_server({**VALID, "version": "0.1.13"}).fingerprint

    assert first.startswith("sha256:")
    assert first != second


def _write_servers(path: Path, body: str) -> Path:
    file = path / "servers.yaml"
    file.write_text(body, encoding="utf-8")
    return file


def test_servers_load_from_a_yaml_list(tmp_path: Path) -> None:
    file = _write_servers(
        tmp_path,
        "- name: S1\n  version: '1'\n  command: uvx\n- name: S2\n  version: '2'\n  command: npx\n",
    )

    servers = load_servers(file)

    assert [server.name for server in servers] == ["S1", "S2"]


def test_a_missing_server_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(McpConfigError, match="no MCP server config"):
        load_servers(tmp_path / "nope.yaml")


def test_a_server_file_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    file = _write_servers(tmp_path, "name: S1\n")

    with pytest.raises(McpConfigError, match="must contain a list"):
        load_servers(file)


def test_two_servers_cannot_share_an_arm_name(tmp_path: Path) -> None:
    file = _write_servers(
        tmp_path,
        "- name: S1\n  version: '1'\n  command: uvx\n- name: S1\n  version: '2'\n  command: npx\n",
    )

    with pytest.raises(McpConfigError, match="defines S1 more than once"):
        load_servers(file)


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------


@dataclass
class FakeTool:
    name: str
    description: str | None = "does a thing"
    inputSchema: dict[str, Any] | None = None  # noqa: N815 - mirrors the SDK field


@dataclass
class FakeListing:
    tools: list[FakeTool]


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeImageBlock:
    data: str = "..."


@dataclass
class FakeOutcome:
    content: Any
    isError: bool = False  # noqa: N815 - mirrors the SDK field


@dataclass
class FakeSdkSession:
    listing: FakeListing = field(default_factory=lambda: FakeListing(tools=[]))
    outcome: Any = None
    calls: list[tuple[str, Mapping[str, Any]]] = field(default_factory=list)
    initialized: bool = False

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> FakeListing:
        return self.listing

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _session(sdk: FakeSdkSession) -> StdioSession:
    from contextlib import AsyncExitStack

    return StdioSession(session=sdk, stack=AsyncExitStack())


async def test_advertised_tools_are_reported_in_order() -> None:
    sdk = FakeSdkSession(
        listing=FakeListing(
            tools=[
                FakeTool(name="run_select_query", inputSchema={"type": "object"}),
                FakeTool(name="list_tables", description=None),
            ]
        )
    )

    tools = await _session(sdk).list_tools()

    assert tools == (
        ToolSpec(
            name="run_select_query",
            description="does a thing",
            input_schema={"type": "object"},
        ),
        ToolSpec(name="list_tables", description="", input_schema={}),
    )


async def test_a_tool_result_flattens_its_text_blocks() -> None:
    sdk = FakeSdkSession(
        outcome=FakeOutcome(
            content=[FakeTextBlock("row 1"), FakeImageBlock(), FakeTextBlock("row 2")]
        )
    )

    result = await _session(sdk).call_tool("run_select_query", {"query": "SELECT 1"})

    assert result == ToolResult(content="row 1\nrow 2", is_error=False)
    assert sdk.calls == [("run_select_query", {"query": "SELECT 1"})]


async def test_a_tool_that_reports_failure_is_data_not_an_exception() -> None:
    # The system under test refusing a call is a measurement of that system
    sdk = FakeSdkSession(outcome=FakeOutcome(content=[FakeTextBlock("bad column")], isError=True))

    result = await _session(sdk).call_tool("run_select_query", {})

    assert result.is_error is True
    assert result.content == "bad column"


async def test_non_list_content_is_stringified_rather_than_dropped() -> None:
    sdk = FakeSdkSession(outcome=FakeOutcome(content="plain"))

    assert (await _session(sdk).call_tool("t", {})).content == "plain"


async def test_a_transport_failure_reaches_the_runner() -> None:
    sdk = FakeSdkSession(outcome=BrokenPipeError("server died"))

    with pytest.raises(McpError, match="calling tool 't' failed"):
        await _session(sdk).call_tool("t", {})


# --------------------------------------------------------------------------
# connecting
# --------------------------------------------------------------------------


class FakeSdkModule(ModuleType):
    """Stands in for ``mcp`` and ``mcp.client.stdio``."""

    def __init__(self, session: FakeSdkSession, *, explode: Exception | None = None) -> None:
        super().__init__("mcp")
        self._session = session
        self._explode = explode
        self.parameters: list[dict[str, Any]] = []

    def StdioServerParameters(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802 - SDK name
        self.parameters.append(kwargs)
        return kwargs

    @asynccontextmanager
    async def stdio_client(self, parameters: Any) -> Any:
        if self._explode is not None:
            raise self._explode
        yield ("read", "write")

    @asynccontextmanager
    async def ClientSession(self, read: Any, write: Any) -> Any:  # noqa: N802 - SDK name
        yield self._session


def _config(**overrides: Any) -> McpServerConfig:
    return parse_server({**VALID, **overrides})


async def test_connecting_launches_the_configured_command_and_handshakes() -> None:
    sdk = FakeSdkSession()
    module = FakeSdkModule(sdk)

    session = await connect(
        _config(),
        environ={"CLICKHOUSE_PASSWORD": "hunter2"},
        importer=lambda _: module,
    )

    assert sdk.initialized is True
    assert module.parameters[0]["command"] == "uvx"
    assert module.parameters[0]["args"] == ["mcp-clickhouse"]
    assert module.parameters[0]["env"]["CLICKHOUSE_PASSWORD"] == "hunter2"
    await session.close()


async def test_a_missing_sdk_is_an_actionable_error() -> None:
    def missing(name: str) -> ModuleType:
        raise ImportError(name)

    with pytest.raises(McpError, match="uv sync --extra mcp"):
        await connect(_config(env_passthrough=[]), environ={}, importer=missing)


async def test_a_server_that_will_not_start_names_itself() -> None:
    module = FakeSdkModule(FakeSdkSession(), explode=FileNotFoundError("uvx not found"))

    with pytest.raises(McpError, match="could not start MCP server 'S1_mcp_clickhouse'"):
        await connect(_config(), environ={"CLICKHOUSE_PASSWORD": "x"}, importer=lambda _: module)
