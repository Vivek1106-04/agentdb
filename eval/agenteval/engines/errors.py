"""Mapping engine errors onto the reported taxonomy (SPEC §11.1).

The error distribution is a published column, so classification cannot be a
guess. ClickHouse puts a stable numeric code in every server exception; that
code is the classification, and anything without one is not a query failure at
all — it is the connection breaking, which must reach the runner rather than be
recorded as a wrong answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from agenteval.systems.base import ErrorClass

_CODE = re.compile(r"\bCode:\s*(\d+)\b")

CLICKHOUSE_ERROR_CLASSES: Mapping[int, ErrorClass] = {
    10: "semantic",  # NOT_FOUND_COLUMN_IN_BLOCK
    42: "semantic",  # NUMBER_OF_ARGUMENTS_DOESNT_MATCH
    43: "semantic",  # ILLEGAL_TYPE_OF_ARGUMENT
    46: "semantic",  # UNKNOWN_FUNCTION
    47: "semantic",  # UNKNOWN_IDENTIFIER
    53: "semantic",  # TYPE_MISMATCH
    60: "semantic",  # UNKNOWN_TABLE
    62: "syntax",  # SYNTAX_ERROR
    81: "semantic",  # UNKNOWN_DATABASE
    158: "limit_exceeded",  # TOO_MANY_ROWS
    159: "timeout",  # TIMEOUT_EXCEEDED
    160: "timeout",  # TOO_SLOW
    164: "permission",  # READONLY
    241: "limit_exceeded",  # MEMORY_LIMIT_EXCEEDED
    307: "limit_exceeded",  # TOO_MANY_BYTES
    386: "semantic",  # NO_COMMON_TYPE
    396: "limit_exceeded",  # TOO_MANY_ROWS_OR_BYTES
    497: "permission",  # ACCESS_DENIED
}
"""Codes seen from agent-written SQL. Unlisted codes fall back to ``semantic``:
the server parsed the query and refused it, which is what semantic means here."""


def clickhouse_error_class(message: str) -> ErrorClass | None:
    """Classify a ClickHouse exception, or ``None`` if it is not a server error.

    ``None`` means the query never reached the server — a dead socket, a DNS
    failure, an auth handshake. Those are run-fatal and must not be graded.
    """
    match = _CODE.search(message)
    if match is None:
        return None
    return CLICKHOUSE_ERROR_CLASSES.get(int(match.group(1)), "semantic")


_DBX_ERROR_CLASS = re.compile(r"\[([A-Z][A-Z0-9_.]+)\]")
_SQLSTATE = re.compile(r"\bSQLSTATE:?\s*([0-9A-Z]{5})\b")

DATABRICKS_ERROR_CLASSES: Mapping[str, ErrorClass] = {
    "PARSE_SYNTAX_ERROR": "syntax",
    "PARSE_EMPTY_STATEMENT": "syntax",
    "TABLE_OR_VIEW_NOT_FOUND": "semantic",
    "UNRESOLVED_COLUMN": "semantic",
    "UNRESOLVED_ROUTINE": "semantic",
    "AMBIGUOUS_REFERENCE": "semantic",
    "SCHEMA_NOT_FOUND": "semantic",
    "CATALOG_NOT_FOUND": "semantic",
    "DATATYPE_MISMATCH": "semantic",
    "UNSUPPORTED_FEATURE": "plan_rejection",
    "UNSUPPORTED_EXPR_FOR_OPERATOR": "plan_rejection",
    "OPERATION_CANCELED": "timeout",
    "STATEMENT_TIMEOUT": "timeout",
    "INSUFFICIENT_PERMISSIONS": "permission",
    "MAX_RECORDS_PER_FETCH_EXCEEDED": "limit_exceeded",
}
"""Databricks names its own error classes. ``VERIFY:`` against the workspace
runtime before citing these anywhere a reader will check them."""

DATABRICKS_SQLSTATE_CLASSES: Mapping[str, ErrorClass] = {
    "42601": "syntax",
    "42P01": "semantic",
    "42703": "semantic",
    "42883": "semantic",
    "42000": "semantic",
    "42501": "permission",
    "0A000": "plan_rejection",
    "HY008": "timeout",
    "57014": "timeout",
    "54000": "limit_exceeded",
}
"""SQLSTATE carries the cases the named classes do not: it is standardized and
moves far less between runtimes."""


def databricks_error_class(message: str) -> ErrorClass | None:
    """Classify a Databricks failure, or ``None`` if it never reached the warehouse.

    A message with neither a named error class nor a SQLSTATE did not come from
    the SQL layer — it is a dead connection or an auth failure, which is
    run-fatal and must not be graded as a wrong answer.
    """
    named = _DBX_ERROR_CLASS.search(message)
    if named is not None:
        known = DATABRICKS_ERROR_CLASSES.get(named.group(1))
        if known is not None:
            return known

    sqlstate = _SQLSTATE.search(message)
    if sqlstate is not None:
        return DATABRICKS_SQLSTATE_CLASSES.get(sqlstate.group(1), "semantic")
    return "semantic" if named is not None else None
