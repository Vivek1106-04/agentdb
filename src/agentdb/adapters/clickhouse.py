"""The ClickHouse adapter (SPEC §6, §8.1).

Everything an agent cannot see in a schema dump is read here: the sort key that
decides whether a filter prunes granules, the skip indexes and projections, the
per-column footprint, the sampled distributions, and the engine's own plan with
index evidence turned on.

Three properties are deliberate and load-bearing:

* **Estimates are labelled at the source.** A profile built from a ``SAMPLE`` says
  ``sample_method="sample"`` and reports how many rows it actually read.
* **Read-only is the connection's property, not this module's.** The adapter
  inspects no SQL strings; it connects as an account whose profile carries
  ``readonly`` and the ceilings (SPEC §13.3). String filtering is not a security
  boundary and pretending otherwise would publish a false safety claim.
* **Every statement is attributable.** Executions carry
  ``log_comment = agentdb:{context}:{turn}``, so any figure the server reports
  can be checked against ``system.query_log`` by someone who does not trust us.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from agentdb.adapters import clickhouse_sql as ch
from agentdb.adapters.base import BaseAdapter, Capability, QuerySemanticError
from agentdb.adapters.errors import clickhouse_error
from agentdb.adapters.models import (
    MAX_TOP_VALUES,
    ColumnDef,
    ColumnProfile,
    DialectRules,
    Engine,
    ExplainMode,
    Limits,
    PhysicalLayout,
    Projection,
    RawPlan,
    Relation,
    RelationDetail,
    RelationRef,
    ResultSet,
    SampleMethod,
    SamplePolicy,
    SkipIndex,
    TimeWindow,
    WorkloadEntry,
)

LOG_COMMENT_PREFIX = "agentdb"
"""``WHERE log_comment LIKE 'agentdb:%'`` attributes rows back to an agent turn."""

CLICKHOUSE_QUIRKS: tuple[str, ...] = (
    "EXPLAIN is estimate-only; there is no ANALYZE. Measured rows come from system.query_log.",
    "Filters prune granules only through the ORDER BY key, left to right; a filter on a "
    "later key column without the leading one prunes nothing.",
    "Identifiers are quoted with backticks; double quotes work but backticks are canonical.",
    "GROUP BY on a high-cardinality column builds aggregate state in memory before it spills.",
    "JOIN loads the right-hand table into memory: put the smaller relation on the right.",
)
"""Facts a model trained mostly on ANSI SQL and Postgres does not have. Cheap to
state, and each one corresponds to a failure mode observed in the wild (SPEC §2.2)."""

CLICKHOUSE_RESERVED_WORDS: frozenset[str] = frozenset(
    {
        "ALL", "ALTER", "AND", "ANY", "ARRAY", "AS", "ASOF", "BETWEEN", "BY", "CASE", "CAST",
        "CHECK", "CLUSTER", "COLLATE", "COLUMN", "CREATE", "CROSS", "CUBE", "DATABASE", "DEFAULT",
        "DELETE", "DESC", "DESCRIBE", "DISTINCT", "DROP", "ELSE", "END", "EXISTS", "EXPLAIN",
        "FINAL", "FIRST", "FORMAT", "FROM", "FULL", "GLOBAL", "GROUP", "HAVING", "IF", "IN",
        "INDEX", "INNER", "INSERT", "INTERVAL", "INTO", "IS", "JOIN", "KEY", "LEFT", "LIKE",
        "LIMIT", "NOT", "NULL", "OFFSET", "ON", "OR", "ORDER", "OUTER", "PREWHERE", "PRIMARY",
        "RIGHT", "SAMPLE", "SELECT", "SEMI", "SET", "SETTINGS", "TABLE", "THEN", "TOTALS", "UNION",
        "USING", "VALUES", "VIEW", "WHEN", "WHERE", "WITH",
    }
)  # fmt: skip


class QueryResult(Protocol):
    """The slice of a ``clickhouse-connect`` result this adapter reads."""

    column_names: Sequence[str]
    result_rows: Sequence[Sequence[Any]]
    summary: Mapping[str, str]


class ClickHouseClient(Protocol):
    """The one client method the adapter uses, so the driver stays swappable."""

    async def query(
        self,
        query: str,
        *,
        parameters: Mapping[str, Any],
        settings: Mapping[str, Any],
    ) -> QueryResult: ...


@dataclass(frozen=True, slots=True)
class ClickHouseAdapter(BaseAdapter):
    """A live ClickHouse server, behind the adapter contract."""

    client: ClickHouseClient
    context_id: str = "agentdb"
    turn_id: Callable[[], str] = lambda: uuid4().hex[:12]
    top_k: int = MAX_TOP_VALUES
    engine: Engine = "clickhouse"
    capabilities: frozenset[Capability] = field(
        default_factory=lambda: frozenset(
            {
                Capability.ESTIMATE_ONLY_PLAN,
                Capability.SKIP_INDEX,
                Capability.PROJECTION,
                Capability.SORT_KEY,
                Capability.WORKLOAD_LOG,
                Capability.COLUMN_STATS,
                Capability.SAMPLING,
            }
        )
    )

    # -- discovery ---------------------------------------------------------

    async def list_relations(self, namespace: str | None = None) -> list[Relation]:
        """Tables and views, with the size facts ``system.tables`` already holds."""
        if namespace is None:
            result = await self._query(ch.LIST_RELATIONS_ALL, {})
        else:
            result = await self._query(ch.LIST_RELATIONS, {"database": namespace})
        return [
            Relation(
                ref=RelationRef(namespace=str(row[0]), name=str(row[1])),
                kind=ch.relation_kind(str(row[2])),
                engine_type=str(row[2]),
                approx_rows=_optional_int(row[3]),
                on_disk_bytes=_optional_int(row[4]),
                comment=str(row[5]) or None,
            )
            for row in result.result_rows
        ]

    async def describe_relation(self, ref: RelationRef) -> RelationDetail:
        """Columns in declaration order, plus the engine's own ``CREATE`` statement."""
        table_row = await self._table_row(ref)
        columns = await self._query(ch.DESCRIBE_COLUMNS, _ref_params(ref))
        return RelationDetail(
            ref=ref,
            columns=tuple(
                ColumnDef(
                    name=str(row[0]),
                    data_type=str(row[1]),
                    is_nullable=ch.is_nullable(str(row[1])),
                    default_expression=str(row[2]) or None,
                    comment=str(row[3]) or None,
                    compressed_bytes=_optional_int(row[4]),
                    uncompressed_bytes=_optional_int(row[5]),
                )
                for row in columns.result_rows
            ),
            create_statement=str(table_row[5]),
        )

    async def physical_layout(self, ref: RelationRef) -> PhysicalLayout:
        """The sort key, partitioning, skip indexes and projections, in one payload."""
        table_row = await self._table_row(ref)
        indexes = await self._query(ch.SKIP_INDEXES, _ref_params(ref))
        projections = await self._query(ch.PROJECTIONS, _ref_params(ref))
        footprint = await self._query(ch.COLUMN_FOOTPRINT, _ref_params(ref))

        compressed, uncompressed = (
            (_optional_int(footprint.result_rows[0][0]), _optional_int(footprint.result_rows[0][1]))
            if footprint.result_rows
            else (None, None)
        )
        return PhysicalLayout(
            engine=self.engine,
            ref=ref,
            create_statement=str(table_row[5]),
            table_engine=str(table_row[0]),
            order_by=ch.split_key_expression(str(table_row[1])),
            partition_by=ch.split_key_expression(str(table_row[2])),
            primary_key=ch.split_key_expression(str(table_row[3])),
            sampling_key=str(table_row[4]) or None,
            skip_indexes=tuple(
                SkipIndex(
                    name=str(row[0]),
                    index_type=str(row[1]),
                    expression=str(row[3]),
                    granularity=int(row[4]),
                    compressed_bytes=_optional_int(row[5]),
                )
                for row in indexes.result_rows
            ),
            projections=tuple(
                Projection(name=str(row[0]), query=str(row[1])) for row in projections.result_rows
            ),
            approx_rows=_optional_int(table_row[6]),
            on_disk_bytes=_optional_int(table_row[7]),
            compression_ratio=_ratio(uncompressed, compressed),
        )

    async def dialect_rules(self) -> DialectRules:
        """Quoting, reserved words and quirks for the connected server version."""
        result = await self._query(ch.VERSION, {})
        return DialectRules(
            engine=self.engine,
            version=str(result.result_rows[0][0]),
            identifier_quote="`",
            supports_ilike=True,
            reserved_words=CLICKHOUSE_RESERVED_WORDS,
            quirks=CLICKHOUSE_QUIRKS,
        )

    # -- profiling ---------------------------------------------------------

    async def column_profile(
        self, ref: RelationRef, columns: list[str], sample: SamplePolicy
    ) -> list[ColumnProfile]:
        """Sampled distributions, one probe per column, each labelled with its method.

        The probe reads a declared fraction where the table has a sampling key and
        a bounded prefix where it does not. Either way the figures are estimates
        and say so — a profile that claimed exactness would eventually make an
        agent confidently wrong about a column it never fully read.
        """
        self.require(Capability.COLUMN_STATS)
        detail = await self.describe_relation(ref)
        types = {column.name: column.data_type for column in detail.columns}
        layout = await self.physical_layout(ref)

        fraction = sample.fraction if layout.is_sampleable else None
        method: SampleMethod = "full" if fraction == 1.0 else "sample"
        source = ch.qualified(ref.namespace, ref.name)

        profiles: list[ColumnProfile] = []
        for column in columns:
            if column not in types:
                raise QuerySemanticError(
                    f"{ref} has no column {column!r}",
                    suggestion="call describe_relation to list the columns that exist",
                )
            statement = ch.profile_statement(
                source=source,
                column=column,
                top_k=self.top_k,
                sample_fraction=fraction,
                max_rows=sample.max_rows,
            )
            row = (
                await self._query(
                    statement,
                    {},
                    settings={
                        "max_execution_time": sample.timeout_s,
                        "max_rows_to_read": sample.max_rows,
                    },
                )
            ).result_rows[0]
            profiles.append(
                ColumnProfile(
                    name=column,
                    data_type=types[column],
                    sample_method=method,
                    sampled_rows=int(row[5]),
                    approx_distinct=_optional_int(row[0]),
                    null_ratio=float(row[1]),
                    min_value=str(row[2]),
                    max_value=str(row[3]),
                    top_values=_top_values(row[4]),
                )
            )
        return profiles

    # -- plans and execution ----------------------------------------------

    async def explain(self, sql: str, mode: ExplainMode) -> RawPlan:
        """The engine's plan, verbatim, with the statement that produced it.

        ``ANALYZE`` is refused rather than approximated: ClickHouse cannot measure
        a plan without running the query, and an estimate returned under the name
        of a measurement is the one thing the plan layer must never do.
        """
        if mode is ExplainMode.ANALYZE:
            self.require(Capability.ANALYZE_PLAN)
        statement = ch.explain_statement(sql, mode)
        result = await self._query(statement, {})
        return RawPlan(
            engine=self.engine,
            mode=mode,
            sql=sql,
            payload="\n".join(str(row[0]) for row in result.result_rows),
            statements=(statement,),
        )

    async def execute(self, sql: str, limits: Limits) -> ResultSet:
        """Run ``sql`` under ``limits``, truncating rather than streaming forever."""
        settings: dict[str, Any] = {
            "max_execution_time": limits.timeout_s,
            "max_result_rows": limits.max_result_rows,
        }
        if limits.max_rows_to_read is not None:
            settings["max_rows_to_read"] = limits.max_rows_to_read
        if limits.max_bytes_to_read is not None:
            settings["max_bytes_to_read"] = limits.max_bytes_to_read

        result = await self._query(sql, {}, settings=settings)
        rows = tuple(tuple(row) for row in result.result_rows)
        truncated = len(rows) > limits.max_result_rows
        kept = rows[: limits.max_result_rows] if truncated else rows
        return ResultSet(
            columns=tuple(result.column_names),
            rows=kept,
            row_count=len(kept),
            truncated=truncated,
            rows_read=_summary_int(result.summary, "read_rows"),
            bytes_read=_summary_int(result.summary, "read_bytes"),
        )

    async def workload(self, window: TimeWindow, top_n: int) -> list[WorkloadEntry]:
        """The costliest normalized query shapes in ``window``, from the server's own log."""
        self.require(Capability.WORKLOAD_LOG)
        result = await self._query(
            ch.WORKLOAD,
            {"start": window.start, "end": window.end, "top_n": top_n},
        )
        return [
            WorkloadEntry(
                normalized_sql=str(row[0]),
                calls=int(row[1]),
                total_duration_ms=float(row[2]),
                mean_duration_ms=float(row[3]),
                rows_read=_optional_int(row[4]),
                bytes_read=_optional_int(row[5]),
                sample_sql=str(row[6]),
                relations=tuple(str(name) for name in row[7]),
            )
            for row in result.result_rows
        ]

    # -- plumbing ----------------------------------------------------------

    async def _table_row(self, ref: RelationRef) -> Sequence[Any]:
        """The ``system.tables`` row for ``ref``, or a semantic error if there is none."""
        result = await self._query(ch.TABLE_ROW, _ref_params(ref))
        if not result.result_rows:
            raise QuerySemanticError(
                f"relation {ref} does not exist",
                suggestion="call list_relations to see what this database holds",
            )
        return result.result_rows[0]

    async def _query(
        self,
        statement: str,
        parameters: Mapping[str, Any],
        *,
        settings: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        """Issue one statement, translating any driver exception into an adapter error."""
        merged: dict[str, Any] = {
            "log_comment": f"{LOG_COMMENT_PREFIX}:{self.context_id}:{self.turn_id()}"
        }
        merged.update(settings or {})
        try:
            return await self.client.query(statement, parameters=parameters, settings=merged)
        except Exception as exc:
            raise clickhouse_error(str(exc)) from exc


def _ref_params(ref: RelationRef) -> dict[str, Any]:
    return {"database": ref.namespace, "table": ref.name}


def _optional_int(value: Any) -> int | None:
    """``None`` stays ``None``: an unknown row count must not become a zero."""
    return None if value is None else int(value)


def _ratio(uncompressed: int | None, compressed: int | None) -> float | None:
    if not compressed or uncompressed is None:
        return None
    return uncompressed / compressed


def _top_values(values: Sequence[Sequence[Any]]) -> tuple[tuple[str, int], ...]:
    """``approx_top_k`` returns ``(value, count, error)`` triples; keep the first two."""
    return tuple((str(entry[0]), int(entry[1])) for entry in values)


def _summary_int(summary: Mapping[str, str], key: str) -> int | None:
    """Read one counter from the server's own summary, tolerating its absence."""
    raw = summary.get(key)
    if raw is None:
        return None
    return int(raw)
