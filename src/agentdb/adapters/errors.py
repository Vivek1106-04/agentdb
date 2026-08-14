"""Turning an engine's exception into the adapter's own failure type (SPEC §6).

Core never sees a driver exception. It sees an :class:`~agentdb.adapters.base.AdapterError`
subclass carrying an :class:`~agentdb.adapters.models.ErrorClass`, because a tool
response has to tell an agent *what kind* of wrong it was — a syntax slip it can
repair, a plan the engine refused, or a ceiling it must respect.

ClickHouse puts a stable numeric code in every server exception, so the code is
the classification. A message with no code never reached the server: that is the
connection breaking, and it is reported as such rather than as a bad query.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

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

_CODE = re.compile(r"\bCode:\s*(\d+)\b")

CLICKHOUSE_ERROR_TYPES: Mapping[int, type[AdapterError]] = {
    10: QuerySemanticError,  # NOT_FOUND_COLUMN_IN_BLOCK
    42: QuerySemanticError,  # NUMBER_OF_ARGUMENTS_DOESNT_MATCH
    43: QuerySemanticError,  # ILLEGAL_TYPE_OF_ARGUMENT
    46: QuerySemanticError,  # UNKNOWN_FUNCTION
    47: QuerySemanticError,  # UNKNOWN_IDENTIFIER
    48: PlanRejectionError,  # NOT_IMPLEMENTED — the shape, not the syntax
    53: QuerySemanticError,  # TYPE_MISMATCH
    60: QuerySemanticError,  # UNKNOWN_TABLE
    62: QuerySyntaxError,  # SYNTAX_ERROR
    81: QuerySemanticError,  # UNKNOWN_DATABASE
    158: LimitExceededError,  # TOO_MANY_ROWS
    159: QueryTimeoutError,  # TIMEOUT_EXCEEDED
    160: QueryTimeoutError,  # TOO_SLOW
    164: QueryPermissionError,  # READONLY
    241: LimitExceededError,  # MEMORY_LIMIT_EXCEEDED
    307: LimitExceededError,  # TOO_MANY_BYTES
    386: QuerySemanticError,  # NO_COMMON_TYPE
    396: LimitExceededError,  # TOO_MANY_ROWS_OR_BYTES
    497: QueryPermissionError,  # ACCESS_DENIED
}
"""Codes seen from agent-written SQL. An unlisted code is semantic: the server
parsed the query and refused it, which is exactly what semantic means here."""

_SUGGESTIONS: Mapping[type[AdapterError], str] = {
    QuerySyntaxError: (
        "check ClickHouse SQL syntax; call dialect_rules for quoting and reserved words"
    ),
    QuerySemanticError: "call describe_relation to confirm the table and column names exist",
    PlanRejectionError: (
        "the engine refused this query shape; try rewriting the aggregate as an "
        "explicit GROUP BY, or call explain to see the plan"
    ),
    QueryTimeoutError: "add a filter on the sort key or reduce the scanned range, then retry",
    QueryPermissionError: "this connection is read-only; only SELECT statements are allowed",
    LimitExceededError: "narrow the query with a filter or a LIMIT; a scan ceiling was hit",
    EngineConnectionError: "check the server is reachable and the credentials are correct",
}


def clickhouse_error(message: str) -> AdapterError:
    """Build the adapter error for a ClickHouse exception ``message``.

    A message with no ``Code:`` is a connection failure, not a rejected query,
    and is reported as :class:`~agentdb.adapters.base.EngineConnectionError` so a
    dead socket can never be recorded as a wrong answer.
    """
    match = _CODE.search(message)
    error_type: type[AdapterError]
    if match is None:
        error_type = EngineConnectionError
    else:
        error_type = CLICKHOUSE_ERROR_TYPES.get(int(match.group(1)), QuerySemanticError)
    return error_type(message, suggestion=_SUGGESTIONS[error_type])
