"""Connecting to a warehouse, on a machine with no SDK and no warehouse.

The factory is tested through an injected importer for the same reason the
ClickHouse one is: importing agentdb must not require a database driver, and a
missing credential must fail at startup with a message naming what is missing —
not halfway through a benchmark run.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from agentdb.adapters.base import EngineConnectionError
from agentdb.adapters.databricks_client import (
    PARAMETER_MODULE,
    ApiStatementResult,
    DatabricksTarget,
    StatementExecutionClient,
    build_client,
    normalize_host,
)

ENV = {
    "AGENTDB_DBX_HOST": "https://dbc-test.cloud.databricks.com",
    "AGENTDB_DBX_WAREHOUSE_ID": "abc123",
    "AGENTDB_DBX_TOKEN": "dapi-secret",
}


def test_the_target_is_read_from_the_environment_only() -> None:
    target = DatabricksTarget.from_env(ENV)

    assert target.host == "https://dbc-test.cloud.databricks.com"
    assert target.warehouse_id == "abc123"
    assert target.catalog == "samples"
    assert target.schema == "tpch"


@pytest.mark.parametrize("missing", ["AGENTDB_DBX_HOST", "AGENTDB_DBX_WAREHOUSE_ID"])
def test_an_incomplete_configuration_fails_at_startup_naming_what_is_missing(
    missing: str,
) -> None:
    env = {key: value for key, value in ENV.items() if key != missing}

    with pytest.raises(EngineConnectionError, match=missing):
        DatabricksTarget.from_env(env)


class _Api:
    """A Statement Execution API that records what it was asked to run."""

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
            schema=SimpleNamespace(columns=[SimpleNamespace(name="version")]),
            truncated=False,
            total_row_count=1,
            total_byte_count=64,
        ),
        "result": SimpleNamespace(data_array=[["2026.30"]]),
        "status": SimpleNamespace(error=None),
    }
    return SimpleNamespace(**{**base, **overrides})


def _parameter(*, name: str, value: str) -> SimpleNamespace:
    """Stands in for the SDK's StatementParameterListItem."""
    return SimpleNamespace(name=name, value=value)


def _client(api: _Api) -> StatementExecutionClient:
    return StatementExecutionClient(
        api=api,
        warehouse_id="abc123",
        catalog="samples",
        schema="tpch",
        parameter=_parameter,
    )


async def test_a_statement_returns_columns_rows_and_the_id_that_makes_it_auditable() -> None:
    api = _Api(_response())

    result = await _client(api).statement("SELECT current_version()", parameters={})

    assert result.columns == ("version",)
    assert result.rows == (("2026.30",),)
    assert result.statement_id == "01ef-abc"
    assert result.rows_read == 1
    assert result.bytes_read == 64


async def test_parameters_are_passed_as_markers_rather_than_interpolated() -> None:
    api = _Api(_response())

    await _client(api).statement("SELECT :catalog", parameters={"catalog": "samples"})

    # typed objects, not dicts: the SDK calls .as_dict() on each one
    assert [(item.name, item.value) for item in api.calls[0]["parameters"]] == [
        ("catalog", "samples")
    ]


async def test_a_datetime_parameter_travels_in_iso_form() -> None:
    from datetime import UTC, datetime

    api = _Api(_response())

    await _client(api).statement(
        "SELECT :start", parameters={"start": datetime(2026, 8, 14, tzinfo=UTC)}
    )

    assert api.calls[0]["parameters"][0].value.startswith("2026-08-14T00:00:00")


async def test_limits_and_timeout_reach_the_api() -> None:
    api = _Api(_response())

    await _client(api).statement(
        "SELECT 1", parameters={}, row_limit=10, byte_limit=1_000, timeout_s=17
    )

    assert api.calls[0]["row_limit"] == 10
    assert api.calls[0]["byte_limit"] == 1_000
    assert api.calls[0]["wait_timeout"] == "17s"


@pytest.mark.parametrize(("timeout_s", "expected"), [(1, "5s"), (600, "50s"), (None, "50s")])
async def test_the_synchronous_wait_is_clamped_to_the_documented_range(
    timeout_s: int | None, expected: str
) -> None:
    api = _Api(_response())

    await _client(api).statement("SELECT 1", parameters={}, timeout_s=timeout_s)

    assert api.calls[0]["wait_timeout"] == expected


async def test_a_statement_that_returned_nothing_is_empty_rather_than_a_crash() -> None:
    api = _Api(_response(manifest=None, result=None))

    result = await _client(api).statement("SHOW TBLPROPERTIES t", parameters={})

    assert result.columns == ()
    assert result.rows == ()
    assert result.rows_read is None
    assert result.truncated is False


async def test_an_api_reported_error_is_raised_rather_than_read_as_an_empty_result() -> None:
    api = _Api(
        _response(status=SimpleNamespace(error=SimpleNamespace(message="[PARSE_SYNTAX_ERROR] no")))
    )

    with pytest.raises(EngineConnectionError, match="PARSE_SYNTAX_ERROR"):
        await _client(api).statement("SELECT", parameters={})


def _importer(module: ModuleType | Exception) -> Any:
    def _import(name: str) -> ModuleType:
        if isinstance(module, Exception):
            raise module
        return module

    return _import


async def test_building_a_client_without_the_sdk_says_what_to_install() -> None:
    with pytest.raises(EngineConnectionError, match="databricks-sdk"):
        await build_client(
            DatabricksTarget.from_env(ENV), importer=_importer(ImportError("no module"))
        )


async def test_building_a_client_wires_the_statement_execution_api() -> None:
    workspace = SimpleNamespace(statement_execution=_Api(_response()))
    sdk = SimpleNamespace(WorkspaceClient=lambda **kwargs: workspace)
    parameters = SimpleNamespace(StatementParameterListItem=_parameter)

    def importer(name: str) -> ModuleType:
        return cast(ModuleType, parameters if name == PARAMETER_MODULE else sdk)

    client = await build_client(DatabricksTarget.from_env(ENV), importer=importer)

    assert isinstance(client, StatementExecutionClient)
    assert client.warehouse_id == "abc123"


async def test_an_unreachable_workspace_is_reported_as_a_connection_failure() -> None:
    def _explode(**kwargs: Any) -> Any:
        raise RuntimeError("host unreachable")

    module = SimpleNamespace(WorkspaceClient=_explode)

    with pytest.raises(EngineConnectionError, match="cannot reach the Databricks workspace"):
        await build_client(DatabricksTarget.from_env(ENV), importer=_importer(module))  # type: ignore[arg-type]


def test_the_api_result_satisfies_the_shape_the_adapter_reads() -> None:
    result = ApiStatementResult(columns=("a",), rows=(("b",),))

    assert result.duration_ms is None
    assert result.truncated is False


# -- what the first live run taught -----------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "https://dbc-test.cloud.databricks.com/?o=1234567890123456",
        "https://dbc-test.cloud.databricks.com/",
        "https://dbc-test.cloud.databricks.com",
        "dbc-test.cloud.databricks.com",
        "  https://dbc-test.cloud.databricks.com/sql/warehouses  ",
    ],
)
def test_a_pasted_workspace_url_is_reduced_to_scheme_and_host(raw: str) -> None:
    # the SDK appends its API path to whatever it is given, so a trailing "?o="
    # turned every statement into "NotFound: Not Found" — a message that reads
    # like a missing table rather than a malformed host
    assert normalize_host(raw) == "https://dbc-test.cloud.databricks.com"


def test_an_empty_host_stays_empty_so_the_missing_variable_is_what_gets_reported() -> None:
    assert normalize_host("   ") == ""


def test_the_target_normalizes_the_host_it_was_given() -> None:
    target = DatabricksTarget.from_env(
        {**ENV, "AGENTDB_DBX_HOST": "https://dbc-test.cloud.databricks.com/?o=123"}
    )

    assert target.host == "https://dbc-test.cloud.databricks.com"


async def test_parameters_are_built_with_the_api_type_not_plain_dicts() -> None:
    # a dict fails inside the SDK with "'dict' object has no attribute 'as_dict'"
    api = _Api(_response())

    await _client(api).statement("SELECT :catalog", parameters={"catalog": "samples"})

    item = api.calls[0]["parameters"][0]
    assert not isinstance(item, dict)
    assert (item.name, item.value) == ("catalog", "samples")


async def test_the_built_client_uses_the_sdks_parameter_type() -> None:
    workspace = SimpleNamespace(statement_execution=_Api(_response()))
    sdk = SimpleNamespace(WorkspaceClient=lambda **kwargs: workspace)
    parameter_module = SimpleNamespace(StatementParameterListItem=_parameter)

    def importer(name: str) -> ModuleType:
        module = parameter_module if name == PARAMETER_MODULE else sdk
        return cast(ModuleType, module)

    client = await build_client(DatabricksTarget.from_env(ENV), importer=importer)

    assert isinstance(client, StatementExecutionClient)
    assert client.parameter is _parameter
