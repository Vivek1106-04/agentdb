"""The Databricks executor, the harness's second engine (SPEC §11, §8.2).

The point of this arm is comparability: same suite, same grader, same trace
shape, a different engine underneath. So the tests check the things that would
quietly break a cross-engine number — schema dumps that teach under-qualified
names, failures graded as wrong answers when the warehouse was never reached,
and statements that cannot be traced back to the run that issued them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from agenteval.engines.clickhouse import SchemaError
from agenteval.engines.connect import (
    DatabricksTarget,
    EngineConnectionError,
    StatementExecutionClient,
    build_databricks_client,
)
from agenteval.engines.databricks import DatabricksExecutor, DatabricksLimits
from agenteval.engines.errors import databricks_error_class


@dataclass(frozen=True, slots=True)
class FakeResult:
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    statement_id: str | None = "01ef-abc"
    rows_read: int | None = None
    bytes_read: int | None = None


@dataclass
class FakeClient:
    responses: dict[str, FakeResult] = field(default_factory=dict)
    failure: Exception | None = None
    calls: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = field(default_factory=list)

    async def statement(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        row_limit: int | None = None,
        timeout_s: int | None = None,
    ) -> FakeResult:
        self.calls.append((sql, parameters, {"row_limit": row_limit, "timeout_s": timeout_s}))
        if self.failure is not None:
            raise self.failure
        for fragment, response in self.responses.items():
            if fragment in sql:
                return response
        return FakeResult()

    def statements(self) -> list[str]:
        return [sql for sql, _, _ in self.calls]


def _client(**overrides: FakeResult) -> FakeClient:
    responses: dict[str, FakeResult] = {
        "information_schema.tables": FakeResult(
            columns=("table_name",), rows=(("orders",), ("lineitem",))
        ),
        "SHOW CREATE TABLE `samples`.`tpch`.`lineitem`": FakeResult(
            columns=("createtab_stmt",),
            rows=(("CREATE TABLE samples.tpch.lineitem (l_orderkey BIGINT) USING delta",),),
        ),
        "SHOW CREATE TABLE `samples`.`tpch`.`orders`": FakeResult(
            columns=("createtab_stmt",),
            rows=(("CREATE TABLE samples.tpch.orders (o_orderkey BIGINT) USING delta",),),
        ),
    }
    responses.update(overrides)
    return FakeClient(responses=responses)


def _executor(client: FakeClient | None = None, **overrides: Any) -> DatabricksExecutor:
    return DatabricksExecutor(client=client or _client(), turn_id=lambda: "turn01", **overrides)


async def test_the_schema_dump_is_ordered_and_fully_qualified() -> None:
    schema = await _executor().schema_text("tpch")

    # alphabetical, so two machines produce the same prompt
    assert schema.index("lineitem") < schema.index("orders")
    # three-part names, because a two-part dump teaches the habit UNQUALIFIED_RELATION catches
    assert "samples.tpch.lineitem" in schema


async def test_a_catalog_qualified_namespace_is_honoured() -> None:
    client = _client()

    await _executor(client, catalog="samples").schema_text("main.sales")

    assert client.calls[0][1] == {"catalog": "main", "schema": "sales"}


async def test_an_empty_schema_is_fatal_rather_than_an_empty_prompt() -> None:
    client = _client(**{"information_schema.tables": FakeResult(columns=("table_name",), rows=())})

    with pytest.raises(SchemaError, match="has no tables"):
        await _executor(client).schema_text("tpch")


@pytest.mark.parametrize("namespace", ["with space", "2fast", "drop`it"])
async def test_a_namespace_that_is_not_an_identifier_never_reaches_the_warehouse(
    namespace: str,
) -> None:
    client = _client()

    with pytest.raises(SchemaError):
        await _executor(client).schema_text(namespace)


async def test_a_successful_query_carries_the_engines_own_counters() -> None:
    client = _client(
        **{
            "SELECT count(*)": FakeResult(
                columns=("c",), rows=((6_001_215,),), rows_read=6_001_215, bytes_read=48_000_000
            )
        }
    )

    emitted = await _executor(client).run("SELECT count(*) FROM samples.tpch.lineitem")

    assert emitted.succeeded is True
    assert emitted.rows == ((6_001_215,),)
    assert emitted.rows_read == 6_001_215
    assert emitted.bytes_read == 48_000_000
    assert emitted.duration_ms is not None


async def test_a_rejected_query_is_graded_not_raised() -> None:
    client = _client()
    client.failure = RuntimeError("[UNRESOLVED_COLUMN] A column with name `l_ship` cannot be found")

    emitted = await _executor(client).run("SELECT l_ship FROM samples.tpch.lineitem")

    assert emitted.succeeded is False
    assert emitted.error_class == "semantic"


async def test_a_failure_that_never_reached_the_warehouse_is_run_fatal() -> None:
    client = _client()
    client.failure = RuntimeError("connection reset by peer")

    # no error class means no SQL layer answered: grading it would record a
    # dead socket as a wrong answer
    with pytest.raises(RuntimeError, match="connection reset"):
        await _executor(client).run("SELECT 1")


async def test_every_statement_carries_its_attribution_prefix() -> None:
    client = _client()

    await _executor(client, context_id="bench").run("SELECT 1")

    assert client.statements()[0].startswith("/* agentdb:bench:turn01 */")


async def test_limits_travel_with_every_statement() -> None:
    client = _client()

    await _executor(client, limits=DatabricksLimits(timeout_s=17, max_result_rows=25)).run(
        "SELECT 1"
    )

    assert client.calls[0][2] == {"row_limit": 25, "timeout_s": 17}


def test_the_executor_declares_the_engine_the_runner_filters_tasks_by() -> None:
    assert _executor().engine == "databricks"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("[PARSE_SYNTAX_ERROR] near 'FROM'. SQLSTATE: 42601", "syntax"),
        ("[TABLE_OR_VIEW_NOT_FOUND] cannot find table", "semantic"),
        ("[UNSUPPORTED_FEATURE] not supported", "plan_rejection"),
        ("[INSUFFICIENT_PERMISSIONS] denied", "permission"),
        ("[OPERATION_CANCELED] cancelled", "timeout"),
        ("[MAX_RECORDS_PER_FETCH_EXCEEDED] too many", "limit_exceeded"),
        ("statement failed. SQLSTATE: 42501", "permission"),
        ("statement failed. SQLSTATE: 22012", "semantic"),
        ("[NEW_CLASS_NOBODY_MAPPED] something", "semantic"),
    ],
)
def test_databricks_failures_map_onto_the_reported_taxonomy(message: str, expected: str) -> None:
    assert databricks_error_class(message) == expected


def test_a_message_with_no_engine_identifier_is_not_a_query_failure() -> None:
    assert databricks_error_class("TLS handshake timed out") is None


# -- connecting -------------------------------------------------------------


ENV = {
    "AGENTEVAL_DBX_HOST": "https://dbc-test.cloud.databricks.com",
    "AGENTEVAL_DBX_WAREHOUSE_ID": "abc123",
    "AGENTEVAL_DBX_TOKEN": "dapi-secret",
}


def test_the_target_reads_the_environment() -> None:
    target = DatabricksTarget.from_env(ENV)

    assert target.host == "https://dbc-test.cloud.databricks.com"
    assert target.warehouse_id == "abc123"
    assert target.catalog == "samples"


@pytest.mark.parametrize("missing", ["AGENTEVAL_DBX_HOST", "AGENTEVAL_DBX_WAREHOUSE_ID"])
def test_an_incomplete_configuration_refuses_to_start(missing: str) -> None:
    env = {key: value for key, value in ENV.items() if key != missing}

    with pytest.raises(EngineConnectionError, match=missing):
        DatabricksTarget.from_env(env)


class _Api:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def execute_statement(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def _response(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "statement_id": "01ef-abc",
        "manifest": SimpleNamespace(
            schema=SimpleNamespace(columns=[SimpleNamespace(name="c")]),
            total_row_count=1,
            total_byte_count=64,
        ),
        "result": SimpleNamespace(data_array=[[1]]),
        "status": SimpleNamespace(error=None),
    }
    return SimpleNamespace(**{**base, **overrides})


async def test_a_statement_returns_rows_and_the_id_that_makes_it_auditable() -> None:
    api = _Api(_response())
    client = StatementExecutionClient(
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch"
    )

    result = await client.statement("SELECT 1", parameters={"catalog": "samples"}, timeout_s=17)

    assert result.rows == ((1,),)
    assert result.columns == ("c",)
    assert result.statement_id == "01ef-abc"
    assert api.calls[0]["parameters"] == [{"name": "catalog", "value": "samples"}]
    assert api.calls[0]["wait_timeout"] == "17s"


@pytest.mark.parametrize(("timeout_s", "expected"), [(1, "5s"), (600, "50s"), (None, "50s")])
async def test_the_synchronous_wait_is_clamped(timeout_s: int | None, expected: str) -> None:
    api = _Api(_response())
    client = StatementExecutionClient(
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch"
    )

    await client.statement("SELECT 1", parameters={}, timeout_s=timeout_s)

    assert api.calls[0]["wait_timeout"] == expected


async def test_a_statement_that_returned_nothing_is_empty_not_a_crash() -> None:
    api = _Api(_response(manifest=None, result=None))
    client = StatementExecutionClient(
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch"
    )

    result = await client.statement("SHOW TBLPROPERTIES t", parameters={})

    assert result.rows == ()
    assert result.rows_read is None


async def test_an_api_reported_error_is_raised_rather_than_read_as_empty() -> None:
    api = _Api(_response(status=SimpleNamespace(error=SimpleNamespace(message="[X] failed"))))
    client = StatementExecutionClient(
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch"
    )

    with pytest.raises(EngineConnectionError, match=r"\[X\] failed"):
        await client.statement("SELECT", parameters={})


async def test_building_a_client_without_the_sdk_says_what_to_install() -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    with pytest.raises(EngineConnectionError, match="databricks-sdk"):
        await build_databricks_client(DatabricksTarget.from_env(ENV), importer=missing)


async def test_building_a_client_wires_the_statement_execution_api() -> None:
    workspace = SimpleNamespace(statement_execution=_Api(_response()))
    module = SimpleNamespace(WorkspaceClient=lambda **kwargs: workspace)

    client = await build_databricks_client(
        DatabricksTarget.from_env(ENV),
        importer=lambda _: cast(ModuleType, module),
    )

    assert isinstance(client, StatementExecutionClient)
    assert client.warehouse_id == "abc123"


async def test_an_unreachable_workspace_says_so() -> None:
    def explode(**kwargs: Any) -> Any:
        raise RuntimeError("host unreachable")

    module = SimpleNamespace(WorkspaceClient=explode)

    with pytest.raises(EngineConnectionError, match="cannot reach the Databricks workspace"):
        await build_databricks_client(
            DatabricksTarget.from_env(ENV),
            importer=lambda _: cast(ModuleType, module),
        )


async def test_a_datetime_parameter_travels_in_iso_form() -> None:
    api = _Api(_response())
    client = StatementExecutionClient(
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch"
    )

    await client.statement("SELECT :start", parameters={"start": datetime(2026, 8, 14, tzinfo=UTC)})

    assert api.calls[0]["parameters"][0]["value"].startswith("2026-08-14T00:00:00")
