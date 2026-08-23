"""Turning an engine's exception into the adapter's own failure type (SPEC §6).

Core never sees a driver exception. It sees an :class:`~agentdb.adapters.base.AdapterError`
subclass carrying an :class:`~agentdb.adapters.models.ErrorClass`, because a tool
response has to tell an agent *what kind* of wrong it was — a syntax slip it can
repair, a plan the engine refused, or a ceiling it must respect.

ClickHouse puts a stable numeric code in every server exception, so the code is
the classification. A message with no code never reached the server: that is the
connection breaking, and it is reported as such rather than as a bad query.

Databricks does the same job with two identifiers instead of one: a named error
class in brackets (``[TABLE_OR_VIEW_NOT_FOUND]``) and a five-character SQLSTATE.
The named class is preferred where it is known, the SQLSTATE class is the
fallback, and a message carrying neither never reached the warehouse.
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
    452: QueryPermissionError,  # SETTING_CONSTRAINT_VIOLATION — the role capped it
    497: QueryPermissionError,  # ACCESS_DENIED
}
"""Codes seen from agent-written SQL. An unlisted code is semantic: the server
parsed the query and refused it, which is exactly what semantic means here."""

_SUGGESTIONS: Mapping[type[AdapterError], str] = {
    QuerySyntaxError: ("check SQL syntax; call dialect_rules for quoting and reserved words"),
    QuerySemanticError: "call describe_relation to confirm the table and column names exist",
    PlanRejectionError: (
        "the engine refused this query shape; try rewriting the aggregate as an "
        "explicit GROUP BY, or call explain to see the plan"
    ),
    QueryTimeoutError: "add a filter on the sort key or reduce the scanned range, then retry",
    QueryPermissionError: (
        "this connection is read-only and its ceilings are capped by the server's "
        "own profile; only SELECT statements are allowed, and a timeout or row "
        "limit above the profile's maximum is refused rather than granted"
    ),
    LimitExceededError: "narrow the query with a filter or a LIMIT; a scan ceiling was hit",
    EngineConnectionError: "check the server is reachable and the credentials are correct",
}


_DBX_ERROR_CLASS = re.compile(r"\[([A-Z][A-Z0-9_.]+)\]")
_SQLSTATE = re.compile(r"\bSQLSTATE:?\s*([0-9A-Z]{5})\b")

DATABRICKS_ERROR_TYPES: Mapping[str, type[AdapterError]] = {
    "PARSE_SYNTAX_ERROR": QuerySyntaxError,
    "PARSE_EMPTY_STATEMENT": QuerySyntaxError,
    "TABLE_OR_VIEW_NOT_FOUND": QuerySemanticError,
    "UNRESOLVED_COLUMN": QuerySemanticError,
    "UNRESOLVED_ROUTINE": QuerySemanticError,
    "AMBIGUOUS_REFERENCE": QuerySemanticError,
    "SCHEMA_NOT_FOUND": QuerySemanticError,
    "CATALOG_NOT_FOUND": QuerySemanticError,
    "DATATYPE_MISMATCH": QuerySemanticError,
    "UNSUPPORTED_FEATURE": PlanRejectionError,
    "UNSUPPORTED_EXPR_FOR_OPERATOR": PlanRejectionError,
    "NOT_SUPPORTED_IN_JDBC_CATALOG": PlanRejectionError,
    "OPERATION_CANCELED": QueryTimeoutError,
    "STATEMENT_TIMEOUT": QueryTimeoutError,
    "INSUFFICIENT_PERMISSIONS": QueryPermissionError,
    "UC_NOT_ENABLED": QueryPermissionError,
    "DELTA_EXCEED_CHAR_VARCHAR_LIMIT": LimitExceededError,
    "MAX_RECORDS_PER_FETCH_EXCEEDED": LimitExceededError,
}
"""Named error classes seen from agent-written SQL. ``VERIFY:`` against the
workspace runtime before citing any of these in documentation — Databricks adds
error classes between releases, which is why an unknown class falls through to
the SQLSTATE rule rather than to a guess."""

DATABRICKS_SQLSTATE_TYPES: Mapping[str, type[AdapterError]] = {
    "42601": QuerySyntaxError,  # syntax error
    "42P01": QuerySemanticError,  # undefined table
    "42703": QuerySemanticError,  # undefined column
    "42883": QuerySemanticError,  # undefined function
    "42000": QuerySemanticError,  # syntax or access rule violation
    "42501": QueryPermissionError,  # insufficient privilege
    "0A000": PlanRejectionError,  # feature not supported
    "HY008": QueryTimeoutError,  # operation cancelled
    "57014": QueryTimeoutError,  # query cancelled
    "54000": LimitExceededError,  # program limit exceeded
}
"""The fallback classification. SQLSTATE is standardized and moves far less than
the named classes, so it carries the unknown cases."""


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


_DATABRICKS_SUGGESTIONS: Mapping[type[AdapterError], str] = {
    QuerySemanticError: (
        "call describe_relation to confirm the name exists, and qualify it fully as "
        "catalog.schema.table — a two-part name resolves against session state this "
        "connection does not have"
    ),
    PlanRejectionError: (
        "the warehouse refused this query shape; call explain to see the physical plan, "
        "and check whether the expression forced a fallback off Photon"
    ),
    QueryTimeoutError: (
        "add a predicate on the clustering or partition columns so the scan skips files, then retry"
    ),
}
"""Where the Databricks advice differs from the shared advice. Anything not named
here falls back to :data:`_SUGGESTIONS`, which is engine-neutral."""


def databricks_error(message: str) -> AdapterError:
    """Build the adapter error for a Databricks failure ``message``.

    Classification prefers the named error class, falls back to SQLSTATE, and
    reports anything carrying neither as
    :class:`~agentdb.adapters.base.EngineConnectionError` — a message with no
    server identifier in it did not come from the warehouse, and a dead socket
    must never be recorded as a wrong answer.
    """
    error_type = _databricks_error_type(message)
    suggestion = _DATABRICKS_SUGGESTIONS.get(error_type) or _SUGGESTIONS[error_type]
    return error_type(message, suggestion=suggestion)


def _databricks_error_type(message: str) -> type[AdapterError]:
    named = _DBX_ERROR_CLASS.search(message)
    if named is not None:
        known = DATABRICKS_ERROR_TYPES.get(named.group(1))
        if known is not None:
            return known

    sqlstate = _SQLSTATE.search(message)
    if sqlstate is not None:
        return DATABRICKS_SQLSTATE_TYPES.get(sqlstate.group(1), QuerySemanticError)

    # A named class the mapping does not know is still a rejected query, not a
    # broken connection: the warehouse parsed enough to name its own error.
    return QuerySemanticError if named is not None else EngineConnectionError
