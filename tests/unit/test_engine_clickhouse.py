"""The live executor, exercised against a scripted client rather than a server."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import pytest

from agenteval.engines.clickhouse import (
    LOG_COMMENT_PREFIX,
    ClickHouseExecutor,
    ClickHouseLimits,
    SchemaError,
)
from agenteval.engines.connect import (
    DEFAULT_USER,
    ClickHouseTarget,
    EngineConnectionError,
    build_client,
)
from agenteval.engines.errors import clickhouse_error_class
from agenteval.execution import QueryExecutor


@dataclass
class FakeResult:
    column_names: Sequence[str] = ()
    result_rows: Sequence[Sequence[Any]] = ()
    summary: Mapping[str, str] = field(default_factory=dict)


@dataclass
class FakeClient:
    """Replays results in order and records the settings each query carried."""

    results: list[FakeResult | Exception] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    settings: list[Mapping[str, Any]] = field(default_factory=list)

    async def query(self, query: str, *, settings: Mapping[str, Any]) -> FakeResult:
        self.queries.append(query)
        self.settings.append(settings)
        outcome = self.results.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _executor(*results: FakeResult | Exception) -> tuple[ClickHouseExecutor, FakeClient]:
    client = FakeClient(results=list(results))
    return ClickHouseExecutor(client=client, turn_id=lambda: "turn0001"), client


# --------------------------------------------------------------------------
# error classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Code: 62. DB::Exception: Syntax error", "syntax"),
        ("Code: 47. DB::Exception: Unknown identifier", "semantic"),
        ("Code: 159. DB::Exception: Timeout exceeded", "timeout"),
        ("Code: 164. DB::Exception: Cannot execute query in readonly mode", "permission"),
        ("Code: 241. DB::Exception: Memory limit exceeded", "limit_exceeded"),
        ("Code: 9999. DB::Exception: something new", "semantic"),
    ],
)
def test_server_error_codes_map_onto_the_reported_taxonomy(message: str, expected: str) -> None:
    assert clickhouse_error_class(message) == expected


def test_an_error_with_no_code_is_not_a_query_failure() -> None:
    # A dead socket is run-fatal, not a wrong answer
    assert clickhouse_error_class("Connection refused") is None


# --------------------------------------------------------------------------
# executing queries
# --------------------------------------------------------------------------


def test_it_is_a_query_executor() -> None:
    executor, _ = _executor()

    assert isinstance(executor, QueryExecutor)
    assert executor.engine == "clickhouse"


async def test_a_successful_query_reports_rows_and_the_servers_own_counters() -> None:
    # Arrange
    executor, _ = _executor(
        FakeResult(
            column_names=["c"],
            result_rows=[[99997497]],
            summary={"read_rows": "99997497", "read_bytes": "800000000"},
        )
    )

    # Act
    emitted = await executor.run("SELECT count() FROM hits")

    # Assert
    assert emitted.succeeded is True
    assert emitted.columns == ("c",)
    assert emitted.rows == ((99997497,),)
    assert emitted.row_count == 1
    assert emitted.rows_read == 99997497
    assert emitted.bytes_read == 800000000
    assert emitted.duration_ms is not None


async def test_a_rejected_query_is_recorded_not_raised() -> None:
    executor, _ = _executor(RuntimeError("Code: 62. DB::Exception: Syntax error"))

    emitted = await executor.run("SELEC 1")

    assert emitted.succeeded is False
    assert emitted.error_class == "syntax"
    assert emitted.error_text is not None
    assert "Syntax error" in emitted.error_text


async def test_a_connection_failure_reaches_the_runner() -> None:
    # Arrange — grading a dead socket as a wrong answer would corrupt the table
    executor, _ = _executor(OSError("Connection refused"))

    with pytest.raises(OSError, match="Connection refused"):
        await executor.run("SELECT 1")


async def test_every_query_is_tagged_for_query_log_attribution() -> None:
    executor, client = _executor(FakeResult())

    await executor.run("SELECT 1")

    assert client.settings[0]["log_comment"] == f"{LOG_COMMENT_PREFIX}:agenteval:turn0001"


async def test_per_query_ceilings_are_sent() -> None:
    client = FakeClient(results=[FakeResult()])
    executor = ClickHouseExecutor(
        client=client, limits=ClickHouseLimits(max_execution_time=5, max_result_rows=100)
    )

    await executor.run("SELECT 1")

    assert client.settings[0]["max_execution_time"] == 5
    assert client.settings[0]["max_result_rows"] == 100


@pytest.mark.parametrize("summary", [{}, {"read_rows": "not-a-number"}])
async def test_a_missing_or_junk_counter_is_reported_as_unknown(
    summary: dict[str, str],
) -> None:
    # Arrange — an absent counter must read as unknown, never as zero
    executor, _ = _executor(FakeResult(summary=summary))

    assert (await executor.run("SELECT 1")).rows_read is None


# --------------------------------------------------------------------------
# reading the schema
# --------------------------------------------------------------------------


async def test_the_schema_is_the_ddl_of_every_table_in_name_order() -> None:
    executor, client = _executor(
        FakeResult(result_rows=[["visits"], ["hits"]]),
        FakeResult(result_rows=[["CREATE TABLE hits (...)"]]),
        FakeResult(result_rows=[["CREATE TABLE visits (...)"]]),
    )

    schema = await executor.schema_text("agentdb")

    assert schema == "CREATE TABLE hits (...)\n\nCREATE TABLE visits (...)"
    assert client.queries[0] == "SHOW TABLES FROM `agentdb`"
    assert client.queries[1] == "SHOW CREATE TABLE `agentdb`.`hits`"


async def test_an_empty_database_is_fatal() -> None:
    executor, _ = _executor(FakeResult(result_rows=[]))

    with pytest.raises(SchemaError, match="has no tables"):
        await executor.schema_text("agentdb")


async def test_a_malformed_namespace_is_refused() -> None:
    executor, _ = _executor()

    with pytest.raises(SchemaError, match="not a valid ClickHouse identifier"):
        await executor.schema_text("agentdb`; DROP")


# --------------------------------------------------------------------------
# connecting
# --------------------------------------------------------------------------


def test_the_default_target_is_the_read_only_role_on_the_compose_port() -> None:
    target = ClickHouseTarget()

    assert target.username == DEFAULT_USER
    assert target.port == 58123


def test_the_target_reads_the_environment() -> None:
    target = ClickHouseTarget.from_env(
        {"AGENTEVAL_CLICKHOUSE_HOST": "ch.internal", "AGENTEVAL_CLICKHOUSE_PORT": "8123"}
    )

    assert target.host == "ch.internal"
    assert target.port == 8123
    assert target.username == DEFAULT_USER


def test_the_target_falls_back_to_compose_defaults() -> None:
    assert ClickHouseTarget.from_env({}) == ClickHouseTarget()


def test_a_non_numeric_port_is_refused() -> None:
    with pytest.raises(EngineConnectionError, match="must be a number"):
        ClickHouseTarget.from_env({"AGENTEVAL_CLICKHOUSE_PORT": "eight-one-two-three"})


class FakeDriverModule(ModuleType):
    """Stands in for ``clickhouse_connect``."""

    def __init__(self, client: object | Exception) -> None:
        super().__init__("clickhouse_connect")
        self._client = client
        self.kwargs: list[dict[str, object]] = []

    async def get_async_client(self, **kwargs: object) -> object:
        self.kwargs.append(kwargs)
        if isinstance(self._client, Exception):
            raise self._client
        return self._client


async def test_build_client_passes_the_target_to_the_driver() -> None:
    client = FakeClient()
    module = FakeDriverModule(client)

    built = await build_client(ClickHouseTarget(host="db"), importer=lambda _: module)

    assert built is client
    assert module.kwargs[0]["host"] == "db"
    assert module.kwargs[0]["username"] == DEFAULT_USER


async def test_a_missing_driver_is_an_actionable_error() -> None:
    def missing(name: str) -> ModuleType:
        raise ImportError(name)

    with pytest.raises(EngineConnectionError, match="uv sync --extra clickhouse"):
        await build_client(ClickHouseTarget(), importer=missing)


async def test_an_unreachable_server_says_so() -> None:
    module = FakeDriverModule(OSError("Connection refused"))

    with pytest.raises(EngineConnectionError, match="Is `make up` running"):
        await build_client(ClickHouseTarget(), importer=lambda _: module)
