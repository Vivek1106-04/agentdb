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
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from agenteval.engines.clickhouse import SchemaError
from agenteval.engines.connect import (
    DBX_PARAMETER_MODULE,
    DatabricksTarget,
    EngineConnectionError,
    StatementExecutionClient,
    build_databricks_client,
    normalize_host,
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
    history: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    history_lookups: list[str] = field(default_factory=list)

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

    async def query_info(self, statement_id: str) -> Mapping[str, Any] | None:
        self.history_lookups.append(statement_id)
        return self.history.get(statement_id)

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


def _parameter(*, name: str, value: str) -> SimpleNamespace:
    """Stands in for the SDK's StatementParameterListItem."""
    return SimpleNamespace(name=name, value=value)


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
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch", parameter=_parameter
    )

    result = await client.statement("SELECT 1", parameters={"catalog": "samples"}, timeout_s=17)

    assert result.rows == ((1,),)
    assert result.columns == ("c",)
    assert result.statement_id == "01ef-abc"
    assert [(item.name, item.value) for item in api.calls[0]["parameters"]] == [
        ("catalog", "samples")
    ]
    assert api.calls[0]["wait_timeout"] == "17s"


@pytest.mark.parametrize(("timeout_s", "expected"), [(1, "5s"), (600, "50s"), (None, "50s")])
async def test_the_synchronous_wait_is_clamped(timeout_s: int | None, expected: str) -> None:
    api = _Api(_response())
    client = StatementExecutionClient(
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch", parameter=_parameter
    )

    await client.statement("SELECT 1", parameters={}, timeout_s=timeout_s)

    assert api.calls[0]["wait_timeout"] == expected


async def test_a_statement_that_returned_nothing_is_empty_not_a_crash() -> None:
    api = _Api(_response(manifest=None, result=None))
    client = StatementExecutionClient(
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch", parameter=_parameter
    )

    result = await client.statement("SHOW TBLPROPERTIES t", parameters={})

    assert result.rows == ()
    assert result.rows_read is None


async def test_an_api_reported_error_is_raised_rather_than_read_as_empty() -> None:
    api = _Api(_response(status=SimpleNamespace(error=SimpleNamespace(message="[X] failed"))))
    client = StatementExecutionClient(
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch", parameter=_parameter
    )

    with pytest.raises(EngineConnectionError, match=r"\[X\] failed"):
        await client.statement("SELECT", parameters={})


async def test_building_a_client_without_the_sdk_says_what_to_install() -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    with pytest.raises(EngineConnectionError, match="databricks-sdk"):
        await build_databricks_client(DatabricksTarget.from_env(ENV), importer=missing)


async def test_building_a_client_wires_the_statement_execution_api() -> None:
    history = _History()
    workspace = SimpleNamespace(statement_execution=_Api(_response()), query_history=history)
    sdk = SimpleNamespace(WorkspaceClient=lambda **kwargs: workspace)
    parameters = SimpleNamespace(StatementParameterListItem=_parameter, QueryFilter=_filter)

    def importer(name: str) -> ModuleType:
        return cast(ModuleType, parameters if name == DBX_PARAMETER_MODULE else sdk)

    client = await build_databricks_client(DatabricksTarget.from_env(ENV), importer=importer)

    assert isinstance(client, StatementExecutionClient)
    assert client.warehouse_id == "abc123"
    # the SDK's typed parameter class, not a dict builder
    assert client.parameter is _parameter
    # and the history API, without which no Databricks pruning is ever measured
    assert client.history is history
    assert client.query_filter is _filter


@pytest.mark.parametrize(
    "raw",
    [
        "https://dbc-test.cloud.databricks.com/?o=1234567890123456",
        "https://dbc-test.cloud.databricks.com/",
        "dbc-test.cloud.databricks.com",
    ],
)
def test_a_pasted_workspace_url_is_reduced_to_scheme_and_host(raw: str) -> None:
    # the SDK appends its API path to whatever it is given; the extra parts made
    # every statement return "NotFound: Not Found"
    assert normalize_host(raw) == "https://dbc-test.cloud.databricks.com"


def test_an_empty_host_stays_empty_so_the_missing_variable_is_reported() -> None:
    assert normalize_host("  ") == ""


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
        api=api, warehouse_id="abc123", catalog="samples", schema="tpch", parameter=_parameter
    )

    await client.statement("SELECT :start", parameters={"start": datetime(2026, 8, 14, tzinfo=UTC)})

    assert api.calls[0]["parameters"][0].value.startswith("2026-08-14T00:00:00")


# -- measured pruning --------------------------------------------------------
#
# Databricks EXPLAIN carries no file counts, so a trace can only record what was
# pruned by asking the warehouse afterwards. Every assertion below is about
# refusing to record a number that is not a measurement.


def _filter(*, statement_ids: list[str]) -> SimpleNamespace:
    """Stand-in for the SDK's typed ``QueryFilter``."""
    return SimpleNamespace(statement_ids=statement_ids)


class _Entry:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def as_dict(self) -> dict[str, Any]:
        return self._payload


class _History:
    def __init__(self, *entries: Any) -> None:
        self.entries = list(entries)
        self.calls: list[Any] = []

    def list(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(res=self.entries)


def _executor_with_history(entry: Mapping[str, Any] | None, **overrides: Any) -> Any:
    client = _client()
    if entry is not None:
        client.history["01ef-abc"] = entry
    return DatabricksExecutor(client=client, collect_pruning=True, **overrides), client


async def test_a_traced_query_records_the_pruning_the_warehouse_measured() -> None:
    executor, client = _executor_with_history(
        {
            "query_id": "01ef-abc",
            "is_final": True,
            "metrics": {"read_files_count": 3, "pruned_files_count": 37},
        }
    )

    emitted = await executor.run("SELECT 1")

    assert emitted.files_read == 3
    assert emitted.files_pruned == 37
    assert emitted.statement_id == "01ef-abc"
    assert client.history_lookups == ["01ef-abc"]


async def test_pruning_is_not_collected_unless_the_run_asked_for_it() -> None:
    client = _client()
    client.history["01ef-abc"] = {
        "query_id": "01ef-abc",
        "metrics": {"read_files_count": 3, "pruned_files_count": 37},
    }

    emitted = await DatabricksExecutor(client=client).run("SELECT 1")

    assert emitted.files_read is None
    assert client.history_lookups == []  # one API call per query is not free


async def test_a_cache_hit_records_no_pruning_rather_than_a_perfect_one() -> None:
    # every counter is zero on a cached answer; recorded naively that reads as
    # "read 0 files of 0", which a ratio turns into flawless data skipping
    executor, _ = _executor_with_history(
        {
            "query_id": "01ef-abc",
            "metrics": {
                "read_files_count": 0,
                "pruned_files_count": 0,
                "result_from_cache": True,
            },
        }
    )

    emitted = await executor.run("SELECT 1")

    assert emitted.files_read is None
    assert emitted.files_pruned is None


async def test_a_metadata_only_answer_records_no_pruning_either() -> None:
    # SELECT count(*) is served from the Delta log; no file is opened and none
    # is pruned, so there is nothing to report
    executor, _ = _executor_with_history(
        {
            "query_id": "01ef-abc",
            "metrics": {"read_files_count": 0, "result_from_cache": False},
        }
    )

    emitted = await executor.run("SELECT count(*) FROM samples.tpch.region")

    assert emitted.files_read is None


async def test_a_statement_the_history_never_recorded_leaves_the_trace_silent() -> None:
    executor, _ = _executor_with_history(None)

    emitted = await executor.run("SELECT 1")

    assert emitted.files_read is None
    assert emitted.files_pruned is None


async def test_an_entry_without_a_metrics_section_leaves_the_trace_silent() -> None:
    executor, _ = _executor_with_history({"query_id": "01ef-abc", "is_final": True})

    assert (await executor.run("SELECT 1")).files_read is None


async def test_an_unreadable_file_count_is_unknown_rather_than_zero() -> None:
    executor, _ = _executor_with_history(
        {
            "query_id": "01ef-abc",
            "metrics": {"read_files_count": "many", "pruned_files_count": "7"},
        }
    )

    emitted = await executor.run("SELECT 1")

    assert emitted.files_read is None
    assert emitted.files_pruned == 7


async def test_a_statement_with_no_id_cannot_be_attributed_so_it_is_not_looked_up() -> None:
    client = _client(**{"SELECT 1": FakeResult(statement_id=None)})
    executor = DatabricksExecutor(client=client, collect_pruning=True)

    emitted = await executor.run("SELECT 1")

    assert emitted.files_read is None
    assert client.history_lookups == []


async def test_the_history_client_looks_up_by_statement_id() -> None:
    history = _History(_Entry({"query_id": "sid-1", "metrics": {"read_files_count": 2}}))
    client = StatementExecutionClient(
        api=_Api(_response()),
        warehouse_id="abc123",
        catalog="samples",
        schema="tpch",
        parameter=_parameter,
        history=history,
        query_filter=_filter,
    )

    found = await client.query_info("sid-1")

    assert found is not None
    assert found["metrics"]["read_files_count"] == 2
    assert history.calls[0]["filter_by"].statement_ids == ["sid-1"]
    assert history.calls[0]["include_metrics"] is True


async def test_a_history_client_without_the_api_reports_nothing() -> None:
    client = StatementExecutionClient(
        api=_Api(_response()),
        warehouse_id="abc123",
        catalog="samples",
        schema="tpch",
        parameter=_parameter,
    )

    assert await client.query_info("sid-1") is None
    assert await client.query_info("") is None


async def test_a_history_entry_the_sdk_cannot_render_as_a_mapping_is_skipped() -> None:
    client = StatementExecutionClient(
        api=_Api(_response()),
        warehouse_id="abc123",
        catalog="samples",
        schema="tpch",
        parameter=_parameter,
        history=_History(object()),
        query_filter=_filter,
    )

    assert await client.query_info("sid-1") is None


# --------------------------------------------------------------------------
# cells arrive as text, and the grader compares numbers
# --------------------------------------------------------------------------


def _typed_response(types: list[Any], row: list[Any]) -> SimpleNamespace:
    """A response whose manifest declares a type per column, as the API's does."""
    return _response(
        manifest=SimpleNamespace(
            schema=SimpleNamespace(
                columns=[
                    SimpleNamespace(name=f"c{index}", type_name=type_name)
                    for index, type_name in enumerate(types)
                ]
            ),
            total_row_count=1,
            total_byte_count=64,
        ),
        result=SimpleNamespace(data_array=[row]),
    )


async def _cells(types: list[Any], row: list[Any]) -> tuple[Any, ...]:
    client = StatementExecutionClient(
        api=_Api(_typed_response(types, row)),
        warehouse_id="abc123",
        catalog="samples",
        schema="tpch",
        parameter=_parameter,
    )
    result = await client.statement("SELECT 1", parameters={})
    return result.rows[0]


async def test_numbers_are_typed_from_the_manifest_not_left_as_text() -> None:
    # The API returns every cell as a string; the grader normalizes numbers to
    # float but leaves strings alone, so uncoerced cells compare by spelling.
    assert await _cells(["LONG", "DOUBLE", "DECIMAL"], ["42", "1.5", "33199131663.4780"]) == (
        42,
        1.5,
        33199131663.478,
    )


async def test_two_spellings_of_one_number_compare_equal_once_typed() -> None:
    trailing_zero = await _cells(["DECIMAL"], ["33199131663.4780"])
    plain = await _cells(["DECIMAL"], ["33199131663.478"])

    assert trailing_zero == plain


async def test_text_columns_holding_digits_stay_text() -> None:
    # An order id is not a number just because it is spelled with digits.
    assert await _cells(["STRING", "DATE"], ["007", "1995-01-01"]) == ("007", "1995-01-01")


async def test_booleans_are_read_from_the_word() -> None:
    assert await _cells(["BOOLEAN", "BOOLEAN"], ["true", "false"]) == (True, False)


async def test_a_cell_its_declared_type_cannot_parse_is_passed_through() -> None:
    # Losing a whole run to one odd value would be worse than one odd value.
    assert await _cells(["LONG"], ["not-a-number"]) == ("not-a-number",)


async def test_nulls_and_untyped_columns_survive_untouched() -> None:
    assert await _cells(["LONG", ""], [None, "left alone"]) == (None, "left alone")


async def test_the_databricks_executor_holds_nothing_to_release() -> None:
    executor = DatabricksExecutor(client=FakeClient())

    await executor.aclose()  # the SDK owns its session; closing must still be safe


class _TypeName(Enum):
    """Shaped like the SDK's ColumnInfoTypeName, whose str() is not the bare name."""

    LONG = "LONG"


async def test_the_sdk_type_enum_is_read_by_value_not_by_str() -> None:
    # str(ColumnInfoTypeName.LONG) is 'ColumnInfoTypeName.LONG'. Matching on that
    # silently coerces nothing, which is how this fix first shipped doing nothing.
    assert await _cells([_TypeName.LONG], ["42"]) == (42,)
