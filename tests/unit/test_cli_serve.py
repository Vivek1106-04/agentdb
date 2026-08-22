"""The launcher: which engine it connects to, and where credentials come from.

Nothing here talks to an engine. What is worth asserting is that the choice of
engine reaches the adapter layer intact and that the connection details are read
from the environment rather than accepted on a command line, where a warehouse
token would land in shell history.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from agentdb import cli
from agentdb.adapters import EngineConnectionError
from agentdb.adapters.clickhouse_client import ClickHouseTarget
from agentdb.adapters.databricks import DatabricksAdapter
from tests.fakes import clickhouse_hits_fixture
from tests.memory_fakes import FakeConnection


async def test_the_default_engine_is_the_one_that_runs_locally() -> None:
    assert cli.parser().parse_args([]).engine == "clickhouse"


async def test_an_unknown_engine_is_refused_before_anything_connects() -> None:
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["--engine", "snowflake"])


async def test_clickhouse_is_reached_through_its_environment_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    async def build_client(target: Any, **_: Any) -> str:
        seen.append(target)
        return "client"

    monkeypatch.setattr(cli, "build_client", build_client)

    adapter = await cli.build_adapter("clickhouse")

    assert adapter.engine == "clickhouse"
    assert seen[0].host == ClickHouseTarget.from_env().host


async def test_databricks_takes_its_default_catalog_from_the_same_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reference arriving without a catalog must resolve against the configured one."""
    monkeypatch.setenv("AGENTDB_DBX_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("AGENTDB_DBX_WAREHOUSE_ID", "abc123")
    monkeypatch.setenv("AGENTDB_DBX_CATALOG", "measured")

    async def build_databricks_client(target: Any, **_: Any) -> str:
        return "client"

    monkeypatch.setattr(cli, "build_databricks_client", build_databricks_client)

    adapter = await cli.build_adapter("databricks")

    assert adapter.engine == "databricks"
    assert cast(DatabricksAdapter, adapter).catalog == "measured"


async def test_a_connection_failure_surfaces_rather_than_starting_a_useless_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def build_client(target: Any, **_: Any) -> str:
        raise EngineConnectionError("clickhouse is not running")

    monkeypatch.setattr(cli, "build_client", build_client)

    with pytest.raises(EngineConnectionError):
        await cli.serve("clickhouse")


def test_main_serves_the_engine_it_was_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[tuple[str, bool]] = []

    async def serve(engine: str, config: object = None, *, memory: bool = False) -> None:
        served.append((engine, memory))

    monkeypatch.setattr(cli, "serve", serve)

    assert cli.main(["--engine", "databricks"]) == 0
    assert served == [("databricks", False)]


def test_the_memory_tools_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent by default: a build with no Postgres serves the rest of the catalog."""
    served: list[tuple[str, bool]] = []

    async def serve(engine: str, config: object = None, *, memory: bool = False) -> None:
        served.append((engine, memory))

    monkeypatch.setattr(cli, "serve", serve)

    assert cli.main(["--memory"]) == 0
    assert served == [("clickhouse", True)]


async def test_serving_with_memory_opens_the_store_and_applies_its_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogs: list[Any] = []
    connection = FakeConnection()

    async def build_adapter(engine: str) -> Any:
        return clickhouse_hits_fixture()

    async def serve_stdio(catalog: Any, **_: Any) -> None:
        catalogs.append(catalog)

    monkeypatch.setattr(cli, "build_adapter", build_adapter)
    monkeypatch.setattr(cli, "serve_stdio", serve_stdio)
    monkeypatch.setattr(cli, "connect", lambda _dsn: connection)

    await cli.serve("clickhouse", memory=True)

    assert "retrieve_exemplars" in catalogs[0].names
    assert "CREATE TABLE IF NOT EXISTS agentdb_exemplar" in connection.statements[0][0]


async def test_serving_hands_the_catalog_for_that_adapter_to_the_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogs: list[Any] = []

    async def build_adapter(engine: str) -> Any:
        return clickhouse_hits_fixture()

    async def serve_stdio(catalog: Any, **_: Any) -> None:
        catalogs.append(catalog)

    monkeypatch.setattr(cli, "build_adapter", build_adapter)
    monkeypatch.setattr(cli, "serve_stdio", serve_stdio)

    await cli.serve("clickhouse")

    assert catalogs[0].names[0] == "list_namespaces"
