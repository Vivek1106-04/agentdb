"""Engine-neutral value objects exchanged between adapters and core (SPEC §6).

Every type here is frozen. Adapters build them; core reads them and builds new
ones. Nothing crossing this boundary is ever mutated in place.

Two design rules carry the weight:

* **Estimates are labelled.** Anything derived from a sample or a system table
  says so — see :attr:`ColumnProfile.sample_method`. An agent that cannot tell
  an estimate from an exact count will eventually be confidently wrong.
* **Unsupported is explicit.** A field an engine cannot provide is ``None``,
  never a plausible-looking default, and the corresponding
  :class:`~agentdb.adapters.base.Capability` is absent from the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

Engine = Literal["clickhouse", "databricks"]
"""The two engines in scope. SPEC §17: two engines, hard stop.

Spec 1.2 replaced Postgres with Databricks as the second *measured* engine.
Postgres survives in the stack only as the pgvector exemplar store of SPEC §10,
which is infrastructure and never a system under test.
"""

SampleMethod = Literal["full", "sample", "system_table", "unavailable"]
"""How a profile figure was obtained. Never present an estimate as exact."""

RelationKind = Literal["table", "view", "materialized_view", "foreign_table"]
"""What a listed relation is. Views carry no physical layout of their own."""


class ExplainMode(StrEnum):
    """Which plan an adapter should produce.

    Both engines are estimate-only: ClickHouse has no ``ANALYZE``, and Databricks
    ``EXPLAIN`` does not execute either (SPEC §8.2). Neither can populate
    :attr:`PlanNode.actual_rows`, so there is no mode that claims to — measured
    rows come from the workload log after execution, and are labelled as such.
    """

    ESTIMATE = "estimate"
    """Plan without executing the query."""

    COST = "cost"
    """Plan annotated with cost and statistics (Databricks ``EXPLAIN COST``).

    Available only where
    :attr:`~agentdb.adapters.base.Capability.COST_ANNOTATED_PLAN` is declared, and
    meaningless without ``ANALYZE`` having been run for the columns involved
    (SPEC §8.2 footgun 1).
    """

    PIPELINE = "pipeline"
    """Physical execution pipeline (ClickHouse ``EXPLAIN PIPELINE``)."""

    SYNTAX = "syntax"
    """The query after the engine's own rewrites; useful to show the agent."""


class ErrorClass(StrEnum):
    """Taxonomy every failure is bucketed into (SPEC §11.1).

    The benchmark reports a distribution over these, which is how the report can
    say *which* kind of failure a given kind of grounding fixes.
    """

    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    PLAN_REJECTION = "plan_rejection"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    LIMIT_EXCEEDED = "limit_exceeded"
    CONNECTION = "connection"


@dataclass(frozen=True, slots=True)
class RelationRef:
    """A fully qualified table or view.

    ``catalog`` exists for Unity Catalog's three-level ``catalog.schema.table``
    namespace (SPEC §8.2 footgun 3). It is ``None`` on ClickHouse, whose names
    have two parts. A Databricks reference that reaches the engine with fewer
    than three parts resolves against session ``USE`` state the stateless core
    does not have, which is a correctness hazard, not a style one.
    """

    namespace: str
    name: str
    catalog: str | None = None

    def __str__(self) -> str:
        if self.catalog is None:
            return f"{self.namespace}.{self.name}"
        return f"{self.catalog}.{self.namespace}.{self.name}"

    @property
    def parts(self) -> tuple[str, ...]:
        """The name components, in qualification order."""
        if self.catalog is None:
            return (self.namespace, self.name)
        return (self.catalog, self.namespace, self.name)


@dataclass(frozen=True, slots=True)
class Relation:
    """A relation as it appears in a listing: cheap facts only."""

    ref: RelationRef
    kind: RelationKind
    engine_type: str | None
    """Storage engine or data source format, e.g. ``MergeTree``, ``DELTA``."""

    approx_rows: int | None
    on_disk_bytes: int | None
    comment: str | None = None


@dataclass(frozen=True, slots=True)
class ColumnDef:
    """A column's declared shape, as the engine reports it."""

    name: str
    data_type: str
    is_nullable: bool
    default_expression: str | None = None
    comment: str | None = None
    compressed_bytes: int | None = None
    uncompressed_bytes: int | None = None

    @property
    def compression_ratio(self) -> float | None:
        """Uncompressed / compressed, or ``None`` when either side is unknown."""
        if not self.compressed_bytes or self.uncompressed_bytes is None:
            return None
        return self.uncompressed_bytes / self.compressed_bytes


@dataclass(frozen=True, slots=True)
class RelationDetail:
    """Everything ``describe_relation`` returns."""

    ref: RelationRef
    columns: tuple[ColumnDef, ...]
    create_statement: str
    comment: str | None = None

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)


@dataclass(frozen=True, slots=True)
class SkipIndex:
    """A ClickHouse data-skipping index (SPEC §9.1.B)."""

    name: str
    index_type: str
    """Normalized type, e.g. ``bloom_filter``, ``minmax``, ``set``, ``tokenbf_v1``."""

    expression: str
    granularity: int
    compressed_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class Projection:
    """A ClickHouse projection: a second physical ordering of the same data."""

    name: str
    query: str


@dataclass(frozen=True, slots=True)
class PhysicalLayout:
    """The physical design an agent cannot see in a schema dump (SPEC §6).

    This is the differentiating payload on both engines, and the mechanism is the
    same story twice: on ClickHouse whether a filter prunes *granules* depends on
    :attr:`order_by`, and on Databricks whether it prunes *files* depends on
    :attr:`clustering_columns` and on which columns Delta collects statistics for.
    No ``CREATE TABLE`` listing of column names conveys either.
    """

    engine: Engine
    ref: RelationRef
    create_statement: str

    # ClickHouse
    table_engine: str | None = None
    order_by: tuple[str, ...] | None = None
    partition_by: tuple[str, ...] | None = None
    primary_key: tuple[str, ...] | None = None
    sampling_key: str | None = None
    skip_indexes: tuple[SkipIndex, ...] = ()
    projections: tuple[Projection, ...] = ()
    ttl: str | None = None

    # Databricks / Delta
    table_format: str | None = None
    """``delta``, ``parquet``, ``view``… from ``DESCRIBE DETAIL.format``."""

    clustering_columns: tuple[str, ...] | None = None
    """Liquid clustering key (``CLUSTER BY``) — the analogue of ``order_by``."""

    zorder_columns: tuple[str, ...] | None = None
    """Legacy Z-ORDER, mined from ``DESCRIBE HISTORY``; not a table property."""

    is_managed: bool | None = None
    """Managed Unity Catalog table, so predictive optimization may apply."""

    deletion_vectors_enabled: bool | None = None
    num_files: int | None = None
    avg_file_bytes: float | None = None
    stats_indexed_columns: int | None = None
    """``delta.dataSkippingNumIndexedCols``. ``None`` means the table did not set
    it, and the Delta default applies (see :data:`agentdb.config.DELTA_DEFAULT_STATS_COLUMNS`)."""

    stats_columns: tuple[str, ...] | None = None
    """``delta.dataSkippingStatsColumns``. When set, it — not the ordinal — decides."""

    # both  (partition_by above is shared: MergeTree PARTITION BY / Delta PARTITIONED BY)
    approx_rows: int | None = None
    on_disk_bytes: int | None = None
    compression_ratio: float | None = None

    def has_file_statistics(self, column: str, ordinal: int, *, default_indexed: int) -> bool:
        """Whether Delta collects per-file statistics for ``column`` (SPEC §8.2).

        ``ordinal`` is the 1-based position from ``information_schema.columns``.
        Delta indexes only the first ``delta.dataSkippingNumIndexedCols`` columns
        in schema order, so a filter on column 40 of a wide table cannot skip a
        single file — a silent 100x that no schema dump reveals. An explicit
        ``delta.dataSkippingStatsColumns`` overrides the ordinal rule entirely.
        """
        if self.stats_columns is not None:
            return column in self.stats_columns
        if self.stats_indexed_columns is not None:
            return ordinal <= self.stats_indexed_columns
        return ordinal <= default_indexed

    @property
    def is_sampleable(self) -> bool:
        """Whether ``SAMPLE`` is available, i.e. the table declares a sampling key."""
        return self.sampling_key is not None

    @property
    def leading_sort_column(self) -> str | None:
        """First column of the sort key — the one that decides granule pruning."""
        if not self.order_by:
            return None
        return self.order_by[0]


MAX_TOP_VALUES = 10
"""Cap on :attr:`ColumnProfile.top_values`; matches the ``topK(10)`` probe in SPEC §8.1."""


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Sampled distribution facts for one column (SPEC §6).

    ``approx_distinct`` and friends are estimates from a sample or from engine
    statistics; :attr:`sample_method` and :attr:`sampled_rows` say which, so the
    agent can weigh them.
    """

    name: str
    data_type: str
    sample_method: SampleMethod
    sampled_rows: int
    approx_distinct: int | None = None
    null_ratio: float | None = None
    min_value: str | None = None
    max_value: str | None = None
    top_values: tuple[tuple[str, int], ...] = ()
    """Up to ten ``(value, approx_count)`` pairs, most frequent first."""

    avg_bytes: float | None = None

    def __post_init__(self) -> None:
        if self.sampled_rows < 0:
            raise ValueError(f"sampled_rows must be >= 0, got {self.sampled_rows}")
        if self.null_ratio is not None and not 0.0 <= self.null_ratio <= 1.0:
            raise ValueError(f"null_ratio must be in [0, 1], got {self.null_ratio}")
        if len(self.top_values) > MAX_TOP_VALUES:
            raise ValueError(
                f"top_values holds at most {MAX_TOP_VALUES} entries, got {len(self.top_values)}"
            )

    def is_low_cardinality(self, threshold: int) -> bool:
        """Whether the column is low cardinality at ``threshold`` (see config).

        A method rather than a stored flag: the threshold is configuration, and
        baking it into the value object would make the profile un-cacheable
        across configs.
        """
        return self.approx_distinct is not None and self.approx_distinct <= threshold

    @property
    def is_estimate(self) -> bool:
        """True unless every figure came from reading the whole relation."""
        return self.sample_method != "full"


@dataclass(frozen=True, slots=True)
class RawPlan:
    """An engine's plan output, before normalization into the plan IR.

    Kept verbatim so a reader auditing a benchmark trace sees exactly what the
    engine said, not only agentdb's reading of it.
    """

    engine: Engine
    mode: ExplainMode
    sql: str
    payload: str
    """The raw plan text or JSON document, exactly as returned."""

    statements: tuple[str, ...] = ()
    """The EXPLAIN statements issued, including SETTINGS — reproducibility matters."""


@dataclass(frozen=True, slots=True)
class ResultSet:
    """The outcome of an executed query.

    ``bytes_read`` and ``rows_read`` are engine-intrinsic efficiency measures and
    generalize across hardware in a way wall-clock time does not (SPEC §17).
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    row_count: int
    truncated: bool
    """Whether the result was cut short by ``Limits.max_result_rows``."""

    duration_ms: int | None = None
    rows_read: int | None = None
    bytes_read: int | None = None
    query_id: str | None = None

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(
                    f"row has {len(row)} values but result declares {len(self.columns)} columns"
                )


MetricsSource = Literal["query_history_api", "workload_log"]
"""Where a :class:`QueryMetrics` came from.

Both exist on Databricks and they are not interchangeable. The Query History API
answers by ``statement_id`` immediately; ``system.query.history`` was measured
lagging between **1,514 and 23,290 seconds** behind the warehouse clock across
two runs on a Free Edition workspace — fine for mining yesterday's workload,
useless for attributing the statement a benchmark just ran.
"""


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """What an engine measured while running one statement (SPEC §6, §8.2).

    The counterpart to :class:`RawPlan`, and the more truthful of the two on
    Databricks: ``EXPLAIN`` there is estimate-only and its Photon plans carry no
    file counts at all, so **this is the only place file-pruning evidence exists**.

    Every field is ``None`` when the engine did not report it. That matters more
    here than anywhere else in this module, because three separate situations
    make a warehouse report ``read_files_count = 0`` while the query succeeded:

    * the **result cache** answered it, so no data was read at all;
    * the answer came from Delta **metadata** — ``count(*)`` needs no file;
    * the warehouse simply did not report metrics for that statement.

    Read naively, each of those says "read no files", which a pruning ratio turns
    into "pruned everything, perfectly". :attr:`measured` exists to stop that.
    """

    statement_id: str
    engine: Engine
    source: MetricsSource

    from_result_cache: bool | None = None
    """Whether the result cache answered. When true, nothing below measures data access."""

    files_read: int | None = None
    files_pruned: int | None = None
    """Files data skipping eliminated before reading. The pruning numerator's partner."""

    partitions_read: int | None = None
    rows_read: int | None = None
    rows_produced: int | None = None
    bytes_read: int | None = None
    bytes_in_files_read: int | None = None
    """Total size of the files opened, against which :attr:`bytes_read` is the part
    actually fetched. Their ratio is column projection and in-file skipping
    together — a different mechanism from file pruning, so never merged with it."""

    bytes_pruned: int | None = None
    photon_time_ms: float | None = None
    execution_time_ms: float | None = None
    compilation_time_ms: float | None = None
    total_time_ms: float | None = None
    spill_bytes: int | None = None

    @property
    def measured(self) -> bool:
        """Whether these numbers describe this statement reading data.

        False for a cache hit and for a statement that reported no file access,
        because a ratio computed from either is an artefact rather than a result.
        """
        if self.from_result_cache:
            return False
        return bool(self.files_read) or bool(self.files_pruned)

    @property
    def files_considered(self) -> int | None:
        """Files the scan started from: read plus pruned."""
        if not self.measured or self.files_read is None:
            return None
        return self.files_read + (self.files_pruned or 0)

    @property
    def pruning_ratio(self) -> float | None:
        """Files read over files considered, or ``None`` when unmeasured.

        Low is good. ``1.0`` means data skipping eliminated nothing — the common
        case on an unclustered table, and a true statement about it rather than a
        missing measurement.
        """
        considered = self.files_considered
        if not considered or self.files_read is None:
            return None
        return self.files_read / considered

    @property
    def bytes_ratio(self) -> float | None:
        """Bytes fetched over bytes in the files opened.

        Measured at 0.07 on a TPC-H aggregate that read all ten files: columnar
        projection and row-group skipping threw away 93% of what file pruning
        could not. Reported beside :attr:`pruning_ratio`, never instead of it.
        """
        if not self.measured or not self.bytes_in_files_read or self.bytes_read is None:
            return None
        return self.bytes_read / self.bytes_in_files_read

    def render(self) -> str:
        """One short block for an agent's context, or for a trace."""
        if self.from_result_cache:
            return "Measured: the result cache answered; no data was read."
        if not self.measured:
            return "Measured: the engine reported no file access for this statement."
        lines = [f"Measured (engine metrics, statement {self.statement_id}):"]
        ratio = self.pruning_ratio
        if ratio is not None:
            considered = self.files_considered
            lines.append(
                f"- files read after pruning: {ratio:.1%} "
                f"({self.files_read} of {considered} considered)"
            )
        byte_ratio = self.bytes_ratio
        if byte_ratio is not None:
            lines.append(
                f"- bytes fetched from those files: {byte_ratio:.1%} "
                "(column projection and in-file skipping, not file pruning)"
            )
        if self.rows_read is not None:
            lines.append(f"- rows read: {self.rows_read:,}")
        if self.photon_time_ms is not None:
            lines.append(f"- Photon time: {self.photon_time_ms:,.0f} ms")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class WorkloadEntry:
    """One normalized query shape mined from the engine's own log (SPEC §8)."""

    normalized_sql: str
    calls: int
    relations: tuple[str, ...] = ()
    total_duration_ms: float | None = None
    mean_duration_ms: float | None = None
    rows_read: int | None = None
    bytes_read: int | None = None
    query_id: str | None = None
    sample_sql: str | None = None
    """One concrete instance of the shape, for the advisor to parse."""


@dataclass(frozen=True, slots=True)
class DialectRules:
    """Engine-specific syntax facts an agent trained mostly on ANSI SQL lacks.

    Reserved-word and quoting drift is a documented failure mode (SPEC §2.2), and
    it is cheap to fix by telling the agent the rule instead of letting it
    discover the rule by failing.
    """

    engine: Engine
    version: str
    identifier_quote: str
    string_quote: str = "'"
    supports_ilike: bool = True
    reserved_words: frozenset[str] = field(default_factory=frozenset)
    quirks: tuple[str, ...] = ()
    """Short, actionable notes, e.g. "EXPLAIN is estimate-only; there is no ANALYZE"."""

    def quote_identifier(self, identifier: str) -> str:
        """Quote ``identifier`` for this engine, escaping any embedded quote."""
        q = self.identifier_quote
        return f"{q}{identifier.replace(q, q * 2)}{q}"

    def needs_quoting(self, identifier: str) -> bool:
        """Whether ``identifier`` must be quoted to survive this dialect."""
        if not identifier:
            return True
        if identifier.upper() in self.reserved_words:
            return True
        if not (identifier[0].isalpha() or identifier[0] == "_"):
            return True
        return not all(char.isalnum() or char == "_" for char in identifier[1:])


@dataclass(frozen=True, slots=True)
class Limits:
    """Per-query bounds. Every execution carries one; unbounded egress is a bug."""

    timeout_s: int
    max_result_rows: int
    max_rows_to_read: int | None = None
    max_bytes_to_read: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}")
        if self.max_result_rows <= 0:
            raise ValueError(f"max_result_rows must be > 0, got {self.max_result_rows}")


@dataclass(frozen=True, slots=True)
class SamplePolicy:
    """How much of a relation a profiling probe may read.

    ``fraction`` is used where the engine supports sampling; ``max_rows`` is the
    fallback ceiling where it does not. Profiling a hundred-million-row table
    must never turn into a full scan (SPEC §8.1).
    """

    fraction: float
    max_rows: int
    timeout_s: int

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.max_rows <= 0:
            raise ValueError(f"max_rows must be > 0, got {self.max_rows}")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}")


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A closed-open interval over which to mine the workload log."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"window end {self.end} must be after start {self.start}")

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def minutes(self) -> float:
        """Window length in minutes, as ClickHouse's ``INTERVAL n MINUTE`` wants it."""
        return self.duration.total_seconds() / 60.0
