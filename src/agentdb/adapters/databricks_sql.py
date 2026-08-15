"""The exact Databricks statements the adapter issues, and the parsing they need.

Kept apart from the adapter for the same reason as its ClickHouse counterpart:
these strings are the spec (SPEC §8.2). A reviewer checking that the adapter
really reads ``delta.dataSkippingNumIndexedCols`` — the property that decides
whether a filter can skip a single file — should be able to read the statement
without reading an adapter.

Everything here is pure. No I/O, no client, no state.

Two Databricks facts shape this module:

* **Three-level names.** Every reference is ``catalog.schema.table``. A two-part
  name resolves against session ``USE`` state that a stateless server does not
  have, so this module refuses to build one (SPEC §8.2 footgun 3).
* **``DESCRIBE`` output is not a contract.** ``DESCRIBE DETAIL`` has gained and
  renamed columns across runtimes, so results are read **by column name**, never
  by position, and a missing column is reported as unknown rather than guessed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from agentdb.adapters.models import ExplainMode, RelationKind, RelationRef

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SYSTEM_SCHEMAS: Final = ("information_schema",)
"""Excluded from listings: an agent asking "what tables are there" means the data."""

AGENTDB_TAG_PREFIX: Final = "agentdb"
"""Databricks has no ``log_comment`` setting, so attribution rides in a comment
prefix — ``/* agentdb:{context}:{turn} */`` — which ``system.query.history`` keeps
verbatim in ``statement_text`` (SPEC §8.2)."""

DELTA_STATS_COLUMNS_PROPERTY: Final = "delta.dataSkippingNumIndexedCols"
DELTA_STATS_COLUMN_LIST_PROPERTY: Final = "delta.dataSkippingStatsColumns"
DELTA_DELETION_VECTORS_PROPERTY: Final = "delta.enableDeletionVectors"

MANAGED_TABLE_TYPES: Final = frozenset({"MANAGED", "MANAGED_SHALLOW_CLONE"})
"""Predictive optimization runs on managed tables only, which changes what advice
is worth giving about compaction and clustering."""


class IdentifierError(ValueError):
    """A catalog, schema, table or column name that is not a bare identifier.

    Names reach the adapter from core and from Unity Catalog's own system tables,
    never from a model, so this is a corruption check rather than an injection
    defence — but ``DESCRIBE DETAIL`` and friends take no parameter markers, so a
    malformed name must not be interpolated into one regardless.
    """


def quote_identifier(name: str) -> str:
    """Backtick ``name``, refusing anything that is not a bare identifier."""
    if not _IDENTIFIER.match(name):
        raise IdentifierError(f"{name!r} is not a valid Databricks identifier")
    return f"`{name}`"


def qualified(ref: RelationRef) -> str:
    """``\\`catalog\\`.\\`schema\\`.\\`table\\``` — every part validated.

    A reference without a catalog is refused rather than defaulted: guessing the
    catalog is how a benchmark silently measures the wrong table.
    """
    if ref.catalog is None:
        raise IdentifierError(
            f"{ref} has no catalog; Unity Catalog references need all three name parts"
        )
    return ".".join(quote_identifier(part) for part in ref.parts)


def tag(sql: str, *, context_id: str, turn_id: str) -> str:
    """Prefix ``sql`` with the attribution comment (SPEC §8.2)."""
    return f"/* {AGENTDB_TAG_PREFIX}:{context_id}:{turn_id} */\n{sql}"


LIST_RELATIONS: Final = """
SELECT table_catalog, table_schema, table_name, table_type, data_source_format, comment
FROM system.information_schema.tables
WHERE table_catalog = :catalog AND table_schema = :schema
ORDER BY table_name
""".strip()

LIST_RELATIONS_ALL: Final = f"""
SELECT table_catalog, table_schema, table_name, table_type, data_source_format, comment
FROM system.information_schema.tables
WHERE table_catalog = :catalog
  AND table_schema NOT IN ({", ".join(f"'{schema}'" for schema in SYSTEM_SCHEMAS)})
ORDER BY table_schema, table_name
""".strip()

TABLE_ROW: Final = """
SELECT table_type, data_source_format, comment
FROM system.information_schema.tables
WHERE table_catalog = :catalog AND table_schema = :schema AND table_name = :table
""".strip()

DESCRIBE_COLUMNS: Final = """
SELECT column_name, ordinal_position, full_data_type, is_nullable, comment
FROM system.information_schema.columns
WHERE table_catalog = :catalog AND table_schema = :schema AND table_name = :table
ORDER BY ordinal_position
""".strip()

WORKLOAD: Final = """
SELECT statement_text, statement_type, execution_status,
       total_duration_ms, read_bytes, read_rows, produced_rows, statement_id
FROM system.query.history
WHERE start_time >= :start AND start_time < :end
ORDER BY read_bytes DESC
LIMIT :top_n
""".strip()
"""The ``system.query_log`` counterpart (SPEC §8.2).

Failed statements are workload too — they are the shapes an agent got wrong — so
``execution_status`` is selected rather than filtered on.
"""

VERSION: Final = "SELECT current_version().dbsql_version AS version"
"""``VERIFY:`` against the workspace runtime; the struct's field names have moved."""


def describe_detail(ref: RelationRef) -> str:
    """Format, location, partition and clustering keys, file count and size."""
    return f"DESCRIBE DETAIL {qualified(ref)}"


def show_tblproperties(ref: RelationRef) -> str:
    """The Delta properties that decide whether data skipping can fire at all."""
    return f"SHOW TBLPROPERTIES {qualified(ref)}"


def show_create_table(ref: RelationRef) -> str:
    return f"SHOW CREATE TABLE {qualified(ref)}"


def describe_history(ref: RelationRef, *, limit: int) -> str:
    """Table history, from which legacy Z-ORDER columns are mined.

    Z-ORDER is not a table property — it is a historical ``OPTIMIZE`` operation,
    so the only record of it is the log (SPEC §8.2).
    """
    return f"DESCRIBE HISTORY {qualified(ref)} LIMIT {limit}"


def explain_statement(sql: str, mode: ExplainMode) -> str:
    """The ``EXPLAIN`` text for ``mode``, verbatim per SPEC §8.2.

    ``FORMATTED`` is the estimate plan: it is where ``PartitionFilters``,
    ``PushedFilters``, the file counts and the ``Photon`` node-name prefixes live,
    which is the whole of the pruning evidence. ``COST`` is gated on
    :attr:`~agentdb.adapters.base.Capability.COST_ANNOTATED_PLAN` by the caller
    because its numbers are meaningless without ``ANALYZE``.
    """
    if mode is ExplainMode.ESTIMATE:
        return f"EXPLAIN FORMATTED {sql}"
    if mode is ExplainMode.COST:
        return f"EXPLAIN COST {sql}"
    if mode is ExplainMode.PIPELINE:
        return f"EXPLAIN FORMATTED {sql}"
    return f"EXPLAIN EXTENDED {sql}"


PHYSICAL_PLAN_MARKER: Final = "== Physical Plan =="
"""Present in every ``EXPLAIN`` mode's output. Its absence means no plan."""

PLANNING_ERROR_MARKERS: Final = ("Error occurred during query planning", "SQLSTATE:")


def explain_failure(payload: str) -> str | None:
    """The error inside an ``EXPLAIN`` result, if the statement did not plan.

    Databricks answers ``EXPLAIN`` over an invalid query with **success**, and
    puts the analysis error in the result rows where a plan would be — observed
    live against ``samples.tpch``. Left alone, the plan layer hands an agent an
    unparseable payload, and a query the warehouse could name the error for
    surfaces as a parser crash instead of a semantic failure it could repair.
    """
    if PHYSICAL_PLAN_MARKER in payload:
        return None
    if any(marker in payload for marker in PLANNING_ERROR_MARKERS):
        return payload.strip()
    return None


def profile_statement(*, source: str, column: str, sample_percent: float) -> str:
    """The distribution probe for one column (SPEC §8.2, path 2).

    ``TABLESAMPLE`` rather than a full scan: profiling is work the agent did not
    ask to wait for, and ``samples.tpch.lineitem`` is not small.

    Path 1 of the spec — ``ANALYZE TABLE … COMPUTE STATISTICS`` then
    ``DESCRIBE EXTENDED`` — is deliberately not used here: ``ANALYZE`` writes to
    the table's statistics, and the measurement connection is read-only (SPEC
    §13.3). The result is labelled ``sample``, which is what it is.
    """
    quoted = quote_identifier(column)
    return (
        f"SELECT approx_count_distinct({quoted}) AS approx_distinct,\n"
        f"       count_if({quoted} IS NULL) / greatest(count(*), 1) AS null_ratio,\n"
        f"       cast(min({quoted}) AS STRING) AS min_value,\n"
        f"       cast(max({quoted}) AS STRING) AS max_value,\n"
        f"       count(*) AS sampled_rows\n"
        f"FROM {source} TABLESAMPLE ({sample_percent} PERCENT)"
    )


def top_values_statement(*, source: str, column: str, top_k: int, sample_percent: float) -> str:
    """Top-k for one column.

    Unavoidably a second query: Databricks has no ``topK()``/``approx_top_k()``
    aggregate that returns value-count pairs in the same pass, so profiling costs
    two statements here against ClickHouse's one. That asymmetry is engine-
    intrinsic and belongs in the report, not hidden in the agent's latency
    (SPEC §8.2).
    """
    quoted = quote_identifier(column)
    return (
        f"SELECT cast({quoted} AS STRING) AS value, count(*) AS occurrences\n"
        f"FROM {source} TABLESAMPLE ({sample_percent} PERCENT)\n"
        f"GROUP BY 1 ORDER BY 2 DESC LIMIT {top_k}"
    )


def row_mapping(columns: Sequence[str], row: Sequence[Any]) -> dict[str, Any]:
    """Zip a result row against its column names.

    Everything that reads ``DESCRIBE`` output goes through this: those result
    shapes change between runtimes, and positional access turns a renamed column
    into a wrong number rather than a missing one.
    """
    return dict(zip(columns, row, strict=False))


def relation_kind(table_type: str) -> RelationKind:
    """Map a Unity Catalog ``table_type`` onto the adapter's relation kinds."""
    normalized = table_type.upper()
    if normalized == "VIEW":
        return "view"
    if normalized in {"MATERIALIZED_VIEW", "STREAMING_TABLE"}:
        return "materialized_view"
    if normalized == "FOREIGN":
        return "foreign_table"
    return "table"


def is_nullable(value: object) -> bool:
    """``information_schema.columns.is_nullable`` is the string ``YES``/``NO``."""
    return str(value).strip().upper() == "YES"


def string_tuple(value: object) -> tuple[str, ...] | None:
    """Read a list-shaped value from ``DESCRIBE DETAIL`` into a tuple.

    The column arrives as a real array from the Statement Execution API and as a
    JSON string through some connectors. Both are accepted; anything else is
    ``None``, because an unparsed clustering key must not read as "no clustering
    key" — that is the difference between "unknown" and "measured to be absent".
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return string_tuple(parsed)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return None


def properties(rows: Sequence[Sequence[Any]]) -> dict[str, str]:
    """``SHOW TBLPROPERTIES`` as a mapping, ignoring rows it cannot read."""
    return {str(row[0]): str(row[1]) for row in rows if len(row) >= 2}


def optional_int(value: object) -> int | None:
    """``None`` stays ``None``: an unknown count must not become a zero."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def optional_bool(value: object) -> bool | None:
    """A Delta boolean property, which arrives as the string ``true``/``false``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def zorder_columns(history: Sequence[Mapping[str, Any]]) -> tuple[str, ...] | None:
    """The Z-ORDER columns of the most recent ``OPTIMIZE``, if there was one.

    Only the latest matters: a later ``OPTIMIZE`` with different columns
    reorganized the same files, so an older entry describes a layout that no
    longer exists.
    """
    for entry in history:
        if str(entry.get("operation", "")).upper() != "OPTIMIZE":
            continue
        parameters = operation_parameters(entry.get("operationParameters"))
        columns = string_tuple(parameters.get("zOrderBy"))
        if columns:
            return columns
    return None


def operation_parameters(value: object) -> Mapping[str, Any]:
    """``DESCRIBE HISTORY.operationParameters`` as a mapping.

    The Statement Execution API returns this column as a **JSON string**, not as
    a map — observed on a live workspace, where treating it as a mapping made
    Z-ORDER mining silently find nothing on every table. A connector may hand
    back a real map, so both are accepted and anything else reads as empty.
    """
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}
