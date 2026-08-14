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
from agentdb.adapters.errors import clickhouse_error
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
