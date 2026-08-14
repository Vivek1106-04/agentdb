"""The exact ClickHouse statements the adapter issues, and the parsing they need.

Kept apart from the adapter for one reason: these strings are the spec (SPEC §8.1).
A reviewer checking that ``EXPLAIN`` really disables ``use_query_condition_cache``
should be able to read the statement without reading an adapter, and a test can
assert on the text rather than on a mock.

Everything here is pure. No I/O, no client, no state.
"""

from __future__ import annotations

import re
from typing import Final

from agentdb.adapters.models import ExplainMode, RelationKind

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SYSTEM_DATABASES: Final = ("system", "INFORMATION_SCHEMA", "information_schema")
"""Excluded from listings: an agent asking "what tables are there" means the data."""

EXPLAIN_SETTINGS: Final = "use_query_condition_cache = 0, use_skip_indexes_on_data_read = 0"
"""The 25.9 footgun. Without both disabled the ``indexes`` output is meaningless,
so the pruning evidence the plan IR is built on would silently be wrong."""

QUERY_LOG_TYPES: Final = (
    "'QueryFinish'",
    "'ExceptionBeforeStart'",
    "'ExceptionWhileProcessing'",
)
"""Failed queries are workload too — they are the shapes an agent got wrong."""


class IdentifierError(ValueError):
    """A database, table or column name that is not a ClickHouse identifier.

    Names reach the adapter from core and from the engine's own system tables,
    never from a model, so this is a corruption check rather than an injection
    defence — but a malformed name must not reach the server regardless.
    """


def quote_identifier(name: str) -> str:
    """Backtick ``name``, refusing anything that is not a bare identifier."""
    if not _IDENTIFIER.match(name):
        raise IdentifierError(f"{name!r} is not a valid ClickHouse identifier")
    return f"`{name}`"


def qualified(namespace: str, name: str) -> str:
    """``\\`db\\`.\\`table\\``` — both halves validated."""
    return f"{quote_identifier(namespace)}.{quote_identifier(name)}"


LIST_RELATIONS: Final = """
SELECT database, name, engine, total_rows, total_bytes, comment
FROM system.tables
WHERE database = {database:String}
ORDER BY name
""".strip()

LIST_RELATIONS_ALL: Final = f"""
SELECT database, name, engine, total_rows, total_bytes, comment
FROM system.tables
WHERE database NOT IN ({", ".join(f"'{db}'" for db in SYSTEM_DATABASES)})
ORDER BY database, name
""".strip()

DESCRIBE_COLUMNS: Final = """
SELECT name, type, default_expression, comment,
       data_compressed_bytes, data_uncompressed_bytes
FROM system.columns
WHERE database = {database:String} AND table = {table:String}
ORDER BY position
""".strip()

TABLE_ROW: Final = """
SELECT engine, sorting_key, partition_key, primary_key, sampling_key,
       create_table_query, total_rows, total_bytes
FROM system.tables
WHERE database = {database:String} AND name = {table:String}
""".strip()

SKIP_INDEXES: Final = """
SELECT name, type, type_full, expr, granularity, data_compressed_bytes
FROM system.data_skipping_indices
WHERE database = {database:String} AND table = {table:String}
ORDER BY name
""".strip()

PROJECTIONS: Final = """
SELECT name, query
FROM system.projections
WHERE database = {database:String} AND table = {table:String}
ORDER BY name
""".strip()

COLUMN_FOOTPRINT: Final = """
SELECT sum(data_compressed_bytes), sum(data_uncompressed_bytes)
FROM system.columns
WHERE database = {database:String} AND table = {table:String}
""".strip()

WORKLOAD: Final = f"""
SELECT normalizeQuery(query) AS normalized,
       count() AS calls,
       sum(query_duration_ms) AS total_ms,
       avg(query_duration_ms) AS mean_ms,
       sum(read_rows) AS rows_read,
       sum(read_bytes) AS bytes_read,
       any(query) AS sample_sql,
       arrayDistinct(arrayFlatten(groupArray(tables))) AS relations
FROM system.query_log
WHERE event_time >= {{start:DateTime}}
  AND event_time < {{end:DateTime}}
  AND type IN ({", ".join(QUERY_LOG_TYPES)})
  AND is_initial_query
GROUP BY normalized
ORDER BY bytes_read DESC
LIMIT {{top_n:UInt32}}
""".strip()

VERSION: Final = "SELECT version()"


def explain_statement(sql: str, mode: ExplainMode) -> str:
    """The ``EXPLAIN`` text for ``mode``, verbatim per SPEC §8.1.

    The estimate plan is requested as JSON. The indented text tree is written for
    humans and its layout is not a contract, while the JSON carries the same
    index evidence under keys a parser can be tested against — and a misread
    pruning number is worse than a crash, because it looks like a measurement.

    :attr:`~agentdb.adapters.models.ExplainMode.COST` is absent by design:
    ClickHouse reports no cost annotations, and the adapter refuses the call
    rather than returning estimates dressed up as measurements. The caller checks
    :attr:`~agentdb.adapters.base.Capability.COST_ANNOTATED_PLAN` first.
    """
    if mode is ExplainMode.ESTIMATE:
        return f"EXPLAIN indexes = 1, projections = 1, json = 1\nSETTINGS {EXPLAIN_SETTINGS}\n{sql}"
    if mode is ExplainMode.PIPELINE:
        return f"EXPLAIN PIPELINE {sql}"
    return f"EXPLAIN SYNTAX {sql}"


def profile_statement(
    *, source: str, column: str, top_k: int, sample_fraction: float | None, max_rows: int
) -> str:
    """The distribution probe for one column (SPEC §8.1).

    ``sample_fraction`` is used only where the table declares a sampling key.
    Without one the probe reads a bounded prefix instead — never a full scan of a
    hundred-million-row table to build a profile nobody asked to wait for.

    ``approx_top_k`` is used rather than the spec sketch's ``topK``: the profile
    carries a count per value, and ``topK`` returns values alone.
    """
    quoted = quote_identifier(column)
    if sample_fraction is not None:
        scan = f"{source} SAMPLE {sample_fraction}"
    else:
        scan = f"(SELECT {quoted} FROM {source} LIMIT {max_rows})"
    return (
        f"SELECT uniqCombined64({quoted}) AS approx_distinct,\n"
        f"       countIf({quoted} IS NULL) / greatest(count(), 1) AS null_ratio,\n"
        f"       toString(min({quoted})) AS min_value,\n"
        f"       toString(max({quoted})) AS max_value,\n"
        f"       approx_top_k({top_k})({quoted}) AS top_values,\n"
        f"       count() AS sampled_rows\n"
        f"FROM {scan}"
    )


def split_key_expression(expression: str) -> tuple[str, ...] | None:
    """Split a sort/partition/primary key into its top-level terms.

    ``system.tables`` reports keys as one expression string, and the terms can be
    function calls: ``toDate(EventTime), CounterID`` is two columns, not three.
    Splitting only on depth-zero commas keeps ``toStartOfHour(t)`` intact.

    Returns ``None`` for a table with no such key, which is a different fact from
    a key with no columns.
    """
    text = expression.strip()
    if not text:
        return None

    terms: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "," and depth == 0:
            terms.append("".join(current).strip())
            current = []
            continue
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        current.append(char)
    terms.append("".join(current).strip())
    return tuple(term for term in terms if term)


def is_nullable(data_type: str) -> bool:
    """Whether a declared ClickHouse type admits NULL.

    ``LowCardinality(Nullable(String))`` is nullable; the wrapper is an encoding
    choice, not a nullability one.
    """
    return "Nullable(" in data_type


def relation_kind(engine: str) -> RelationKind:
    """Map a ClickHouse table engine onto the adapter's relation kinds."""
    if engine == "View":
        return "view"
    if engine == "MaterializedView":
        return "materialized_view"
    return "table"
