"""The Databricks adapter (SPEC §6, §8.2).

The Databricks half of the same story ClickHouse tells: everything an agent
cannot see in a schema dump, read from the engine itself. Where ClickHouse has a
sort key deciding granule pruning, Delta has a clustering key and a *bounded set
of columns carrying per-file statistics* deciding file pruning — and the second
of those is the fact this adapter exists to surface. Delta collects min/max
statistics for the first ``delta.dataSkippingNumIndexedCols`` columns in schema
order (32 by default), so a filter on column 40 of a wide table skips nothing at
all, no matter how selective it is. No ``CREATE TABLE`` output says this.

Three properties are deliberate and load-bearing:

* **Names are three-part or they are refused.** Unity Catalog resolves a
  two-part name against session ``USE`` state, which a stateless server does not
  have; guessing the catalog is how a benchmark measures the wrong table.
* **Read-only is the credential's property, not this module's.** The adapter
  inspects no SQL strings. It connects as a principal with ``SELECT`` and nothing
  else, and never runs ``ANALYZE`` — which writes — during measurement.
* **Every statement is attributable.** Statements carry
  ``/* agentdb:{context}:{turn} */`` and the client returns the API's own
  ``statement_id``, so any figure reported here can be joined to
  ``system.query.history`` by someone who does not trust us.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from agentdb.adapters import databricks_sql as dbx
from agentdb.adapters.base import BaseAdapter, Capability, QuerySemanticError
from agentdb.adapters.errors import databricks_error
from agentdb.adapters.models import (
    MAX_TOP_VALUES,
    ColumnDef,
    ColumnProfile,
    DialectRules,
    Engine,
    ExplainMode,
    Limits,
    PhysicalLayout,
    RawPlan,
    Relation,
    RelationDetail,
    RelationRef,
    ResultSet,
    SamplePolicy,
    TimeWindow,
    WorkloadEntry,
)

HISTORY_DEPTH = 50
"""``DESCRIBE HISTORY`` entries scanned for the latest Z-ORDER ``OPTIMIZE``.

Deep enough to see past a run of routine writes, shallow enough that the probe
costs nothing on a table with years of history."""

DEFAULT_SAMPLE_PERCENT = 1.0
"""``TABLESAMPLE`` percent used when a policy's fraction is not usable."""

DATABRICKS_QUIRKS: tuple[str, ...] = (
    "Every table is catalog.schema.table. A two-part name resolves against session "
    "USE state and may silently hit a different table.",
    "EXPLAIN is estimate-only; there is no ANALYZE. Measured rows come from the query "
    "profile and system.query.history after the statement runs.",
    "Delta collects per-file statistics only for the first 32 columns in schema order "
    "unless the table overrides it; a filter on a later column cannot skip any file.",
    "Filters prune files through the liquid clustering key (CLUSTER BY); a filter on a "
    "non-clustered column reads every file the partition filter left behind.",
    "Wrapping a partition or clustering column in a function — year(ts) = 2026 — defeats "
    "pushdown; write a range on the column itself.",
    "Identifiers are quoted with backticks, as in Spark SQL.",
    "The planner broadcasts the smaller join side when its statistics say it fits; without "
    "ANALYZE those statistics may not exist, and it falls back to a sort-merge join.",
)
"""Facts a model trained mostly on ANSI SQL and Postgres does not have. Each one
corresponds to a failure mode this project expects to measure (SPEC §2.2)."""

DATABRICKS_RESERVED_WORDS: frozenset[str] = frozenset(
    {
        "ALL", "ALTER", "AND", "ANTI", "ANY", "AS", "AUTHORIZATION", "BETWEEN", "BOTH", "BY",
        "CASE", "CAST", "CHECK", "CLUSTER", "COLLATE", "COLUMN", "CONSTRAINT", "CREATE", "CROSS",
        "CUBE", "CURRENT", "DATABASE", "DELETE", "DESC", "DESCRIBE", "DISTINCT", "DROP", "ELSE",
        "END", "ESCAPE", "EXCEPT", "EXISTS", "EXPLAIN", "EXTRACT", "FALSE", "FETCH", "FILTER",
        "FOR", "FOREIGN", "FROM", "FULL", "FUNCTION", "GRANT", "GROUP", "HAVING", "IN", "INNER",
        "INSERT", "INTERSECT", "INTERVAL", "INTO", "IS", "JOIN", "LATERAL", "LEADING", "LEFT",
        "LIKE", "LIMIT", "NATURAL", "NOT", "NULL", "ON", "ONLY", "OR", "ORDER", "OUTER", "OVER",
        "OVERLAPS", "PARTITION", "PRIMARY", "REFERENCES", "RIGHT", "ROLLUP", "SELECT", "SEMI",
        "SESSION_USER", "SOME", "TABLE", "TABLESAMPLE", "THEN", "TO", "TRAILING", "TRUE", "UNION",
        "UNIQUE", "UNKNOWN", "USER", "USING", "VALUES", "WHEN", "WHERE", "WINDOW", "WITH",
    }
)  # fmt: skip


class StatementResult(Protocol):
    """The slice of a statement response this adapter reads.

    Read-only members, so an implementation can be a frozen dataclass: a result
    that could be edited after the fact is a result nobody can audit.
    """

    @property
    def columns(self) -> Sequence[str]: ...

    @property
    def rows(self) -> Sequence[Sequence[Any]]: ...

    @property
    def statement_id(self) -> str | None: ...

    @property
    def truncated(self) -> bool: ...

    @property
    def rows_read(self) -> int | None: ...

    @property
    def bytes_read(self) -> int | None: ...

    @property
    def duration_ms(self) -> int | None: ...


class DatabricksClient(Protocol):
    """The one client method the adapter uses, so the transport stays swappable.

    Named parameters rather than interpolation: the Statement Execution API takes
    ``:name`` markers, and the system-table filters this adapter issues are the
    only place a value from outside ever reaches a statement.
    """

    async def statement(
        self,
        sql: str,
        *,
        parameters: Mapping[str, Any],
        row_limit: int | None = None,
        byte_limit: int | None = None,
        timeout_s: int | None = None,
    ) -> StatementResult: ...


@dataclass(frozen=True, slots=True)
class DatabricksAdapter(BaseAdapter):
    """A live Databricks SQL warehouse, behind the adapter contract."""

    client: DatabricksClient
    catalog: str
    """The default catalog for references that arrive without one."""

    context_id: str = "agentdb"
    turn_id: Callable[[], str] = lambda: uuid4().hex[:12]
    top_k: int = MAX_TOP_VALUES
    sample_percent: float = DEFAULT_SAMPLE_PERCENT
    engine: Engine = "databricks"
    capabilities: frozenset[Capability] = field(
        default_factory=lambda: frozenset(
            {
                Capability.ESTIMATE_ONLY_PLAN,
                Capability.COST_ANNOTATED_PLAN,
                Capability.POST_HOC_PLAN_METRICS,
                Capability.CLUSTERING_KEY,
                Capability.ZORDER,
                Capability.FILE_PRUNING,
                Capability.DATA_SKIPPING_STATS,
                Capability.DELETION_VECTORS,
                Capability.VECTORIZED_ENGINE,
                Capability.THREE_LEVEL_NAMESPACE,
                Capability.PARTITION_PRUNING,
                Capability.WORKLOAD_LOG,
                Capability.COLUMN_STATS,
                Capability.SAMPLING,
            }
        )
    )

    # -- discovery ---------------------------------------------------------

    async def list_relations(self, namespace: str | None = None) -> list[Relation]:
        """Tables and views from Unity Catalog's own information schema.

        ``namespace`` is a schema, optionally catalog-qualified. Omitting it lists
        the adapter's catalog rather than every catalog in the metastore: a
        workspace-wide listing is a bill, not a context payload.
        """
        catalog, schema = self._split_namespace(namespace)
        if schema is None:
            result = await self._query(dbx.LIST_RELATIONS_ALL, {"catalog": catalog})
        else:
            result = await self._query(dbx.LIST_RELATIONS, {"catalog": catalog, "schema": schema})
        return [
            Relation(
                ref=RelationRef(catalog=str(row[0]), namespace=str(row[1]), name=str(row[2])),
                kind=dbx.relation_kind(str(row[3])),
                engine_type=_optional_str(row[4]),
                approx_rows=None,
                on_disk_bytes=None,
                comment=_optional_str(row[5]),
            )
            for row in result.rows
        ]

    async def describe_relation(self, ref: RelationRef) -> RelationDetail:
        """Columns in ordinal order, plus the engine's own ``CREATE`` statement.

        Ordinal order is not cosmetic here: it is the order Delta's statistics
        cut off at, so a column list in any other order would make
        ``STATS_NOT_COLLECTED`` uncomputable.
        """
        resolved = self._resolve(ref)
        await self._assert_exists(resolved)
        columns = await self._query(dbx.DESCRIBE_COLUMNS, _ref_params(resolved))
        return RelationDetail(
            ref=resolved,
            columns=tuple(
                ColumnDef(
                    name=str(row[0]),
                    data_type=str(row[2]),
                    is_nullable=dbx.is_nullable(row[3]),
                    comment=_optional_str(row[4]),
                )
                for row in columns.rows
            ),
            create_statement=await self._create_statement(resolved),
        )

    async def physical_layout(self, ref: RelationRef) -> PhysicalLayout:
        """Clustering key, file layout and the statistics set, in one payload."""
        resolved = self._resolve(ref)
        detail = await self._query(dbx.describe_detail(resolved), {})
        if not detail.rows:
            raise QuerySemanticError(
                f"relation {resolved} reported no detail row",
                suggestion="call list_relations to see what this schema holds",
            )
        row = dbx.row_mapping(detail.columns, detail.rows[0])
        table_properties = dbx.properties(
            (await self._query(dbx.show_tblproperties(resolved), {})).rows
        )
        history = await self._query(dbx.describe_history(resolved, limit=HISTORY_DEPTH), {})

        num_files = dbx.optional_int(row.get("numFiles"))
        size_bytes = dbx.optional_int(row.get("sizeInBytes"))
        return PhysicalLayout(
            engine=self.engine,
            ref=resolved,
            create_statement=await self._create_statement(resolved),
            partition_by=dbx.string_tuple(row.get("partitionColumns")),
            table_format=_optional_str(row.get("format")),
            clustering_columns=dbx.string_tuple(row.get("clusteringColumns")),
            zorder_columns=dbx.zorder_columns(
                [dbx.row_mapping(history.columns, entry) for entry in history.rows]
            ),
            is_managed=_is_managed(row.get("tableType") or row.get("type")),
            deletion_vectors_enabled=dbx.optional_bool(
                table_properties.get(dbx.DELTA_DELETION_VECTORS_PROPERTY)
            ),
            num_files=num_files,
            avg_file_bytes=_average(size_bytes, num_files),
            stats_indexed_columns=dbx.optional_int(
                table_properties.get(dbx.DELTA_STATS_COLUMNS_PROPERTY)
            ),
            stats_columns=_stats_columns(
                table_properties.get(dbx.DELTA_STATS_COLUMN_LIST_PROPERTY)
            ),
            approx_rows=_row_count(row),
            on_disk_bytes=size_bytes,
        )

    async def dialect_rules(self) -> DialectRules:
        """Quoting, reserved words and quirks for the connected warehouse."""
        result = await self._query(dbx.VERSION, {})
        version = str(result.rows[0][0]) if result.rows else "unknown"
        return DialectRules(
            engine=self.engine,
            version=version,
            identifier_quote="`",
            supports_ilike=True,
            reserved_words=DATABRICKS_RESERVED_WORDS,
            quirks=DATABRICKS_QUIRKS,
        )

    # -- profiling ---------------------------------------------------------

    async def column_profile(
        self, ref: RelationRef, columns: list[str], sample: SamplePolicy
    ) -> list[ColumnProfile]:
        """Sampled distributions, two probes per column, each labelled honestly.

        Two probes rather than ClickHouse's one: Databricks has no aggregate that
        returns value-count pairs, so top-k is its own grouped query. The
        asymmetry is engine-intrinsic and belongs in the report (SPEC §8.2).
        """
        self.require(Capability.COLUMN_STATS)
        resolved = self._resolve(ref)
        detail = await self.describe_relation(resolved)
        types = {column.name: column.data_type for column in detail.columns}
        source = dbx.qualified(resolved)
        percent = _sample_percent(sample.fraction, self.sample_percent)

        profiles: list[ColumnProfile] = []
        for column in columns:
            if column not in types:
                raise QuerySemanticError(
                    f"{resolved} has no column {column!r}",
                    suggestion="call describe_relation to list the columns that exist",
                )
            statistics = await self._query(
                dbx.profile_statement(source=source, column=column, sample_percent=percent),
                {},
                timeout_s=sample.timeout_s,
                row_limit=sample.max_rows,
            )
            top = await self._query(
                dbx.top_values_statement(
                    source=source, column=column, top_k=self.top_k, sample_percent=percent
                ),
                {},
                timeout_s=sample.timeout_s,
                row_limit=self.top_k,
            )
            row = statistics.rows[0]
            profiles.append(
                ColumnProfile(
                    name=column,
                    data_type=types[column],
                    sample_method="sample",
                    sampled_rows=dbx.optional_int(row[4]) or 0,
                    approx_distinct=dbx.optional_int(row[0]),
                    null_ratio=_optional_float(row[1]),
                    min_value=_optional_str(row[2]),
                    max_value=_optional_str(row[3]),
                    top_values=tuple(
                        (str(entry[0]), dbx.optional_int(entry[1]) or 0) for entry in top.rows
                    ),
                )
            )
        return profiles

    # -- plans and execution ----------------------------------------------

    async def explain(self, sql: str, mode: ExplainMode) -> RawPlan:
        """The warehouse's plan, verbatim, with the statement that produced it.

        ``EXPLAIN COST`` is gated on its capability *and* is worth little without
        ``ANALYZE`` having been run for the columns involved — it prints plans
        with absent or default statistics that look authoritative (SPEC §8.2
        footgun 1). The caller sees the raw text and can judge.
        """
        if mode is ExplainMode.COST:
            self.require(Capability.COST_ANNOTATED_PLAN)
        statement = dbx.explain_statement(sql, mode)
        result = await self._query(statement, {})
        payload = "\n".join(str(row[0]) for row in result.rows)

        # EXPLAIN over an invalid query *succeeds* and returns the analysis error
        # where the plan should be. Raising here keeps a repairable semantic
        # error from reaching the agent as an unparseable plan.
        failure = dbx.explain_failure(payload)
        if failure is not None:
            raise databricks_error(failure)

        return RawPlan(
            engine=self.engine,
            mode=mode,
            sql=sql,
            payload=payload,
            statements=(statement,),
        )

    async def execute(self, sql: str, limits: Limits) -> ResultSet:
        """Run ``sql`` under ``limits``, truncating rather than streaming forever."""
        result = await self._query(
            sql,
            {},
            row_limit=limits.max_result_rows,
            byte_limit=limits.max_bytes_to_read,
            timeout_s=limits.timeout_s,
        )
        rows = tuple(tuple(row) for row in result.rows)
        over_limit = len(rows) > limits.max_result_rows
        kept = rows[: limits.max_result_rows] if over_limit else rows
        return ResultSet(
            columns=tuple(result.columns),
            rows=kept,
            row_count=len(kept),
            truncated=over_limit or result.truncated,
            duration_ms=result.duration_ms,
            rows_read=result.rows_read,
            bytes_read=result.bytes_read,
            query_id=result.statement_id,
        )

    async def workload(self, window: TimeWindow, top_n: int) -> list[WorkloadEntry]:
        """The costliest statements in ``window``, from the warehouse's own history.

        ``system.query.history`` stores statements one row per execution, with no
        ``normalizeQuery`` equivalent, so each entry is one statement rather than
        a shape with a call count. Saying ``calls=1`` is the honest reading;
        grouping shapes is the workload miner's job, not the adapter's.
        """
        self.require(Capability.WORKLOAD_LOG)
        result = await self._query(
            dbx.WORKLOAD,
            {"start": window.start, "end": window.end, "top_n": top_n},
        )
        return [
            WorkloadEntry(
                normalized_sql=str(row[0]),
                calls=1,
                total_duration_ms=_optional_float(row[3]),
                mean_duration_ms=_optional_float(row[3]),
                rows_read=dbx.optional_int(row[5]),
                bytes_read=dbx.optional_int(row[4]),
                query_id=_optional_str(row[7]),
                sample_sql=str(row[0]),
            )
            for row in result.rows
        ]

    # -- plumbing ----------------------------------------------------------

    def _resolve(self, ref: RelationRef) -> RelationRef:
        """Fill in the adapter's catalog for a two-part reference.

        Filled in once, here, rather than left to the engine: the default belongs
        to the adapter's configuration, where it is auditable, not to whatever
        ``USE`` statement last ran on a shared session.
        """
        if ref.catalog is not None:
            return ref
        return RelationRef(catalog=self.catalog, namespace=ref.namespace, name=ref.name)

    def _split_namespace(self, namespace: str | None) -> tuple[str, str | None]:
        """``schema`` or ``catalog.schema`` into its parts."""
        if namespace is None:
            return self.catalog, None
        catalog, separator, schema = namespace.partition(".")
        if not separator:
            return self.catalog, namespace
        return catalog, schema

    async def _assert_exists(self, ref: RelationRef) -> None:
        result = await self._query(dbx.TABLE_ROW, _ref_params(ref))
        if not result.rows:
            raise QuerySemanticError(
                f"relation {ref} does not exist",
                suggestion="call list_relations to see what this schema holds",
            )

    async def _create_statement(self, ref: RelationRef) -> str:
        result = await self._query(dbx.show_create_table(ref), {})
        return str(result.rows[0][0]) if result.rows else ""

    async def _query(
        self,
        statement: str,
        parameters: Mapping[str, Any],
        *,
        row_limit: int | None = None,
        byte_limit: int | None = None,
        timeout_s: int | None = None,
    ) -> StatementResult:
        """Issue one tagged statement, translating any failure into an adapter error."""
        tagged = dbx.tag(statement, context_id=self.context_id, turn_id=self.turn_id())
        try:
            return await self.client.statement(
                tagged,
                parameters=parameters,
                row_limit=row_limit,
                byte_limit=byte_limit,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            raise databricks_error(str(exc)) from exc


def _ref_params(ref: RelationRef) -> dict[str, Any]:
    return {"catalog": ref.catalog, "schema": ref.namespace, "table": ref.name}


def _row_count(row: Mapping[str, Any]) -> int | None:
    """The relation's row count, if ``DESCRIBE DETAIL`` reported one.

    Older runtimes print ``numRows``; current ones carry it inside a
    ``statistics`` struct, which arrives empty on a table nobody has analyzed —
    observed on ``samples.tpch``, where it is ``{}``. The honest answer is then
    ``None``: a rule that needs a row count stays silent rather than firing on a
    zero that means "not measured".
    """
    direct = dbx.optional_int(row.get("numRows"))
    if direct is not None:
        return direct
    statistics = dbx.operation_parameters(row.get("statistics"))
    return dbx.optional_int(statistics.get("numRows"))


def _is_managed(table_type: object) -> bool | None:
    if table_type is None:
        return None
    return str(table_type).upper() in dbx.MANAGED_TABLE_TYPES


def _stats_columns(raw: str | None) -> tuple[str, ...] | None:
    """``delta.dataSkippingStatsColumns`` is a comma-separated property string."""
    if raw is None:
        return None
    columns = tuple(part.strip() for part in raw.split(",") if part.strip())
    return columns or None


def _average(total: int | None, count: int | None) -> float | None:
    if total is None or not count:
        return None
    return total / count


def _sample_percent(fraction: float, fallback: float) -> float:
    """A sample policy's fraction as a ``TABLESAMPLE`` percent.

    A fraction of 1.0 would ask for the whole table, which profiling must never
    do (SPEC §8.1), so it falls back to the adapter's configured percent.
    """
    if 0.0 < fraction < 1.0:
        return fraction * 100.0
    return fallback


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None
