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

    HYPOTHETICAL_INDEX = "hypothetical_index"
    """Postgres via hypopg: cost a candidate index without building it."""

    ANALYZE_PLAN = "analyze_plan"
    """Plans carry measured row counts (Postgres ``EXPLAIN ANALYZE``)."""

    ESTIMATE_ONLY_PLAN = "estimate_only_plan"
    """Plans are estimates only; there is no ANALYZE (ClickHouse)."""

    SKIP_INDEX = "skip_index"
    """ClickHouse data-skipping indexes."""

    PROJECTION = "projection"
    """ClickHouse projections."""

    SORT_KEY = "sort_key"
    """The relation has an ``ORDER BY`` key that governs granule pruning."""

    WORKLOAD_LOG = "workload_log"
    """``pg_stat_statements`` or ``system.query_log`` is readable."""

    COLUMN_STATS = "column_stats"
    """Column distribution facts are obtainable, by probe or from system tables."""

    SAMPLING = "sampling"
    """The engine can read a declared fraction of a relation cheaply."""


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

    engine: Engine
    capabilities: frozenset[Capability]

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
