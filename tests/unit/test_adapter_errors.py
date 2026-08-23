"""Engine exceptions become typed adapter errors (SPEC §6, §11.1).

The error class is a published column of the benchmark report, so the mapping is
tested code, not a guess made at the call site.
"""

from __future__ import annotations

import pytest

from agentdb.adapters.base import (
    AdapterError,
    EngineConnectionError,
    LimitExceededError,
    PlanRejectionError,
    QueryPermissionError,
    QuerySemanticError,
    QuerySyntaxError,
    QueryTimeoutError,
)
from agentdb.adapters.errors import clickhouse_error, databricks_error
from agentdb.adapters.models import ErrorClass


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Code: 62. DB::Exception: Syntax error", QuerySyntaxError),
        ("Code: 47. DB::Exception: Unknown identifier", QuerySemanticError),
        ("Code: 48. DB::Exception: Not implemented", PlanRejectionError),
        ("Code: 159. DB::Exception: Timeout exceeded", QueryTimeoutError),
        ("Code: 164. DB::Exception: Cannot execute query in readonly mode", QueryPermissionError),
        ("Code: 241. DB::Exception: Memory limit exceeded", LimitExceededError),
    ],
)
def test_each_known_code_maps_to_its_error_type(message: str, expected: type[AdapterError]) -> None:
    assert isinstance(clickhouse_error(message), expected)


def test_an_unlisted_code_is_semantic_because_the_server_parsed_and_refused_it() -> None:
    error = clickhouse_error("Code: 9999. DB::Exception: something new")

    assert isinstance(error, QuerySemanticError)
    assert error.error_class is ErrorClass.SEMANTIC


def test_a_message_without_a_code_is_the_connection_breaking_not_a_bad_query() -> None:
    error = clickhouse_error("Connection refused to localhost:58123")

    assert isinstance(error, EngineConnectionError)
    assert error.error_class is ErrorClass.CONNECTION


def test_every_error_carries_an_actionable_suggestion() -> None:
    error = clickhouse_error("Code: 62. DB::Exception: Syntax error")

    assert error.suggestion is not None
    assert error.as_dict() == {
        "error_class": "syntax",
        "message": "Code: 62. DB::Exception: Syntax error",
        "suggestion": error.suggestion,
    }


# -- Databricks -------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("[PARSE_SYNTAX_ERROR] Syntax error at or near 'FROM'. SQLSTATE: 42601", QuerySyntaxError),
        (
            "[TABLE_OR_VIEW_NOT_FOUND] The table or view `tpch`.`nope` cannot be found",
            QuerySemanticError,
        ),
        ("[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column with name `l_ship`", QuerySemanticError),
        ("[UNSUPPORTED_FEATURE] The feature is not supported. SQLSTATE: 0A000", PlanRejectionError),
        ("[INSUFFICIENT_PERMISSIONS] User does not have SELECT on table", QueryPermissionError),
        ("[OPERATION_CANCELED] The operation has been canceled", QueryTimeoutError),
        ("[MAX_RECORDS_PER_FETCH_EXCEEDED] too many records", LimitExceededError),
    ],
)
def test_a_named_databricks_error_class_decides_the_type(
    message: str, expected: type[AdapterError]
) -> None:
    assert isinstance(databricks_error(message), expected)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Statement failed. SQLSTATE: 42P01", QuerySemanticError),
        ("Statement failed. SQLSTATE: 42501", QueryPermissionError),
        ("Statement failed. SQLSTATE: 0A000", PlanRejectionError),
        ("Statement failed. SQLSTATE: 57014", QueryTimeoutError),
        ("Statement failed. SQLSTATE: 54000", LimitExceededError),
    ],
)
def test_sqlstate_carries_the_cases_the_named_classes_do_not(
    message: str, expected: type[AdapterError]
) -> None:
    assert isinstance(databricks_error(message), expected)


def test_an_unknown_sqlstate_is_semantic_because_the_warehouse_parsed_the_query() -> None:
    assert isinstance(databricks_error("failed. SQLSTATE: 22012"), QuerySemanticError)


def test_an_unknown_named_class_is_still_a_rejected_query_not_a_broken_connection() -> None:
    # the warehouse named its own error, so it was reached
    assert isinstance(
        databricks_error("[BRAND_NEW_ERROR_CLASS] something happened"), QuerySemanticError
    )


def test_a_databricks_message_with_no_identifier_is_the_connection_breaking() -> None:
    error = databricks_error("Connection reset by peer")

    assert isinstance(error, EngineConnectionError)
    assert error.error_class is ErrorClass.CONNECTION


def test_databricks_advice_names_the_three_level_namespace() -> None:
    error = databricks_error("[TABLE_OR_VIEW_NOT_FOUND] cannot find `tpch`.`lineitem`")

    assert "catalog.schema.table" in (error.suggestion or "")


def test_databricks_falls_back_to_the_shared_advice_where_it_does_not_differ() -> None:
    error = databricks_error("[PARSE_SYNTAX_ERROR] Syntax error. SQLSTATE: 42601")

    assert error.suggestion == "check SQL syntax; call dialect_rules for quoting and reserved words"


def test_a_refused_setting_is_a_permission_problem_not_a_semantic_one() -> None:
    """Observed live: the read-only role caps max_execution_time, and code 452 says so.

    Classified as semantic it would have told an agent to go and check its table
    names, which are fine — the server refused a ceiling the caller tried to raise.
    """
    error = clickhouse_error(
        "Code: 452. DB::Exception: Setting max_execution_time shouldn't be greater "
        "than 30. (SETTING_CONSTRAINT_VIOLATION)"
    )

    assert isinstance(error, QueryPermissionError)
    assert error.suggestion is not None
    assert "profile" in error.suggestion
