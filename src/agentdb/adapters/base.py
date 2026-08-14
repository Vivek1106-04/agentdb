"""The adapter contract (SPEC §6).

Core code may call nothing on an engine except the methods declared here. Every
engine-specific fact an adapter can supply is announced through a
:class:`Capability` flag, so core asks *"can you?"* rather than *"which engine
are you?"* — that is what keeps engine knowledge out of core and makes a third
adapter a day's work rather than a refactor.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from agentdb.adapters.models import (
    ColumnProfile,
    DialectRules,
    Engine,
    ErrorClass,
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


class Capability(StrEnum):
    """What an adapter can do. Absence is a fact, not a gap to paper over."""

    # planning
    ESTIMATE_ONLY_PLAN = "estimate_only_plan"
    """Plans are estimates only; neither engine has an executing EXPLAIN."""

    COST_ANNOTATED_PLAN = "cost_annotated_plan"
    """Databricks ``EXPLAIN COST``: cost and statistics annotations, and only
    meaningful once ``ANALYZE`` has run for the columns involved (SPEC §8.2)."""

    POST_HOC_PLAN_METRICS = "post_hoc_plan_metrics"
    """Measured plan metrics exist, but only *after* execution — Databricks query
    profile and ``system.query.history``."""

    # physical design — ClickHouse
    SORT_KEY = "sort_key"
    """The relation has an ``ORDER BY`` key that governs granule pruning."""

    SKIP_INDEX = "skip_index"
    """ClickHouse data-skipping indexes."""

    PROJECTION = "projection"
    """ClickHouse projections."""

    GRANULE_PRUNING = "granule_pruning"
    """Pruning is reported in marks/granules (ClickHouse)."""

    # physical design — Databricks
    CLUSTERING_KEY = "clustering_key"
    """Databricks liquid clustering (``CLUSTER BY``)."""

    ZORDER = "zorder"
    """Databricks legacy Z-ORDER, mined from ``DESCRIBE HISTORY``."""

    FILE_PRUNING = "file_pruning"
    """Pruning is reported in Delta files, not granules."""

    DATA_SKIPPING_STATS = "data_skipping_stats"
    """Per-file min/max/nullCount statistics, collected for a bounded column set."""

    DELETION_VECTORS = "deletion_vectors"
    """Deleted rows are masked rather than rewritten."""

    VECTORIZED_ENGINE = "vectorized_engine"
    """Databricks Photon; a query shape can fall off it silently."""

    THREE_LEVEL_NAMESPACE = "three_level_namespace"
    """Names are ``catalog.schema.table`` (Unity Catalog)."""

    # shared
    PARTITION_PRUNING = "partition_pruning"
    """Partition predicates remove data before the scan."""

    SHADOW_VALIDATION = "shadow_validation"
    """A sampled copy of a relation can be built to measure a design change
    (SPEC §9.2.F). One mechanism, both engines."""

    WORKLOAD_LOG = "workload_log"
    """``system.query_log`` or ``system.query.history`` is readable."""

    COLUMN_STATS = "column_stats"
    """Column distribution facts are obtainable, by probe or from system tables."""

    SAMPLING = "sampling"
    """The engine can read a declared fraction of a relation cheaply.

    Beyond the SPEC §6 list, and kept because profiling depends on it: ClickHouse
    ``SAMPLE`` needs a sampling key and Databricks ``TABLESAMPLE`` does not, and a
    profile that silently full-scanned instead is a cost the caller must be able
    to predict."""


class AdapterError(RuntimeError):
    """Base class for every failure an adapter reports.

    Carries an :class:`~agentdb.adapters.models.ErrorClass` so the benchmark can
    bucket failures without pattern-matching on message text, and a
    ``suggestion`` so a tool response can be actionable rather than a traceback
    (SPEC §12).
    """

    error_class: ErrorClass = ErrorClass.SEMANTIC

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def as_dict(self) -> dict[str, str | None]:
        """The structured shape returned to an MCP client. Never a raw traceback."""
        return {
            "error_class": self.error_class.value,
            "message": self.message,
            "suggestion": self.suggestion,
        }


class QuerySyntaxError(AdapterError):
    """The engine rejected the query before planning it."""

    error_class = ErrorClass.SYNTAX


class QuerySemanticError(AdapterError):
    """The query parsed but references something that does not exist or does not fit."""

    error_class = ErrorClass.SEMANTIC


class PlanRejectionError(AdapterError):
    """The engine refused the query's *shape* at plan time.

    The failure mode that motivates the whole plan layer: legal SQL that a
    columnar engine will not execute as written (SPEC §2.2).
    """

    error_class = ErrorClass.PLAN_REJECTION


class QueryTimeoutError(AdapterError):
    """The query exceeded its :class:`~agentdb.adapters.models.Limits` timeout."""

    error_class = ErrorClass.TIMEOUT


class QueryPermissionError(AdapterError):
    """The connection is not allowed to do this. Read-only is enforced here, not in prose."""

    error_class = ErrorClass.PERMISSION


class LimitExceededError(AdapterError):
    """A scan or result-size ceiling was hit. Bounded egress working as designed."""

    error_class = ErrorClass.LIMIT_EXCEEDED


class EngineConnectionError(AdapterError):
    """The engine was unreachable."""

    error_class = ErrorClass.CONNECTION


class UnsupportedCapabilityError(AdapterError):
    """Core asked for something this engine cannot do.

    Raised instead of returning a plausible default, so a missing capability can
    never be silently mistaken for a measurement.
    """

    error_class = ErrorClass.SEMANTIC

    def __init__(self, engine: str, capability: Capability) -> None:
        super().__init__(
            f"{engine} adapter does not support {capability.value}",
            suggestion=f"check adapter.supports({capability.value!r}) before calling",
        )
        self.capability = capability


@runtime_checkable
class Adapter(Protocol):
    """What every engine adapter provides. Core may call nothing else.

    All methods are async: an adapter is I/O, and the server serves several
    agent turns concurrently under a permit pool.
    """

    @property
    def engine(self) -> Engine:
        """Which engine this adapter speaks to.

        Declared read-only so an implementation can be a frozen dataclass: an
        adapter's identity must not drift halfway through a benchmark run.
        """
        ...

    @property
    def capabilities(self) -> frozenset[Capability]:
        """What this adapter can do. Absence is a fact, not a gap to paper over."""
        ...

    def supports(self, capability: Capability) -> bool:
        """Whether ``capability`` is available on this adapter."""
        ...

    async def list_relations(self, namespace: str | None = None) -> list[Relation]:
        """Tables and views, with cheap size facts only."""
        ...

    async def describe_relation(self, ref: RelationRef) -> RelationDetail:
        """Columns, types, comments and the engine's own ``CREATE`` statement."""
        ...

    async def physical_layout(self, ref: RelationRef) -> PhysicalLayout:
        """Sort key, partitioning, skip indexes, projections, indexes, footprint."""
        ...

    async def column_profile(
        self, ref: RelationRef, columns: list[str], sample: SamplePolicy
    ) -> list[ColumnProfile]:
        """Sampled distribution facts, each labelled with how it was obtained."""
        ...

    async def explain(self, sql: str, mode: ExplainMode) -> RawPlan:
        """The engine's plan output, verbatim, plus the statements used to get it."""
        ...

    async def execute(self, sql: str, limits: Limits) -> ResultSet:
        """Run a read-only query under ``limits``."""
        ...

    async def workload(self, window: TimeWindow, top_n: int) -> list[WorkloadEntry]:
        """Top-``top_n`` costliest normalized query shapes in ``window``."""
        ...

    async def dialect_rules(self) -> DialectRules:
        """Quoting, reserved words and engine quirks for the connected version."""
        ...


class BaseAdapter:
    """Shared, engine-agnostic adapter behaviour.

    Concrete adapters inherit this for capability bookkeeping and implement the
    :class:`Adapter` protocol methods themselves.
    """

    engine: Engine
    capabilities: frozenset[Capability] = frozenset()

    def supports(self, capability: Capability) -> bool:
        """Whether ``capability`` is available on this adapter."""
        return capability in self.capabilities

    def require(self, capability: Capability) -> None:
        """Raise :class:`UnsupportedCapabilityError` unless ``capability`` is available."""
        if not self.supports(capability):
            raise UnsupportedCapabilityError(self.engine, capability)
