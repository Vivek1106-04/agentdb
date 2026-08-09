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
