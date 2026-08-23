"""What a recommendation is, and the demand signal both advisors reason from (SPEC §9).

Two rules run through this package and are worth stating before any algorithm:

**The advisor is not competing with the engine.** ClickHouse ships engine-side
query optimization; Databricks ships Photon, cost-based optimization and
predictive optimization that picks clustering keys and runs ``OPTIMIZE`` unasked.
Engine-side optimization makes *the query you wrote* faster and automatic
physical tuning makes *the table you have* better laid out. Neither helps when an
agent wrote a query whose shape defeats the key that exists, filtered on a column
past the data-skipping statistics limit, or joined the wrong side. That is the
layer this package occupies, and every rationale it emits should read that way.

**Deterministic analysis first.** Everything here is computed from plan evidence,
column profiles and the logged workload. No model is called. That is what makes
arm ``A6_full`` an ablation of *facts* rather than of prompting, and it is why a
recommendation carries its evidence and the method behind its estimate rather
than a number a reader has to trust.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from agentdb.adapters import RelationRef, WorkloadEntry
from agentdb.core.plan_ir import PruningUnit
from agentdb.core.query_shape import QueryShape, analyze


class Kind(StrEnum):
    """What a recommendation asks for (SPEC §9.1)."""

    ORDER_BY = "order_by"
    SKIP_INDEX = "skip_index"
    PROJECTION = "projection"
    CLUSTER_BY = "cluster_by"
    STATS_COLUMNS = "stats_columns"
    COMPACTION = "compaction"
    BROADCAST_HINT = "broadcast_hint"
    REWRITE = "rewrite"
    PARTITION = "partition"
    TYPE_CHANGE = "type_change"
    MATERIALIZED_VIEW = "materialized_view"


class Confidence(StrEnum):
    """How the expected effect was arrived at.

    The three levels are not politeness. ``MEASURED`` means a shadow table was
    built and the plan compared; ``ESTIMATED`` means the numbers came from the
    engine's own statistics; ``HEURISTIC`` means a rule fired on shape alone. A
    recommendation that claimed the first while doing the third is the failure
    this project would least survive being caught at.
    """

    MEASURED = "measured"
    ESTIMATED = "estimated"
    HEURISTIC = "heuristic"


_CONFIDENCE_RANK: Mapping[Confidence, int] = {
    Confidence.MEASURED: 2,
    Confidence.ESTIMATED: 1,
    Confidence.HEURISTIC: 0,
}


@dataclass(frozen=True, slots=True)
class Evidence:
    """The facts a recommendation was derived from, carried with it."""

    source: str
    """Where the facts came from: ``plan``, ``profile``, ``workload``, ``shadow``
    or a combination, written as it will be read."""

    pruning_ratio: float | None = None
    pruning_unit: PruningUnit | None = None
    """Always beside the ratio: a granule ratio and a file ratio are not the same
    quantity and must never be compared without the unit visible (SPEC §7)."""

    relation_rows: int | None = None
    bytes_read: int | None = None
    distinct_counts: tuple[tuple[str, int], ...] = ()
    """``(column, approx_distinct)`` for every column the rule weighed."""

    workload_queries: int | None = None
    """How many logged queries the demand signal was computed over. ``None`` when
    the advice came from the single query in front of the advisor, which is a
    materially weaker basis and should read that way."""


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    """What the recommendation is expected to change, and how that was worked out."""

    metric: str
    """``granules_read``, ``files_read`` or ``bytes_read`` — never latency, which
    this project does not measure on a shared warehouse (SPEC §9.2.C)."""

    before: float | None
    after: float | None
    method: str
    """The derivation, in one sentence. An estimate whose method is not stated is
    a claim, and this project publishes no claims it cannot show the working for."""

    @property
    def reduction(self) -> float | None:
        """Fraction removed, or ``None`` when either end is unknown."""
        if self.before is None or self.after is None or self.before <= 0:
            return None
        return max(0.0, (self.before - self.after) / self.before)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One physical-design or rewrite proposal (SPEC §9.1)."""

    kind: Kind
    relation: RelationRef
    rationale: str
    """Cites the specific evidence. "The sort key leads with CounterID and this
    query filters only on EventDate" — never "consider optimizing your schema"."""

    evidence: Evidence
    expected_effect: EffectEstimate
    confidence: Confidence
    ddl: str | None = None
    """Exact, runnable DDL. Never executed by agentdb without elicited
    confirmation (SPEC §13.3): what an advisor returns is text."""

    rewritten_sql: str | None = None
    risk_notes: tuple[str, ...] = ()
    """What this costs. A recommendation with a real cost and no risk note is
    worse than no recommendation: it spends someone's disk or write throughput
    without telling them."""

    @property
    def priority(self) -> tuple[int, float, str]:
        """Ranking key: confidence, then expected reduction, then a stable name.

        Confidence outranks size of effect deliberately. A measured 20% beats an
        estimated 60%, because the second number is the one that turns out to be
        wrong in front of a database engineer.
        """
        return (
            _CONFIDENCE_RANK[self.confidence],
            self.expected_effect.reduction or 0.0,
            f"{self.kind}:{self.relation}",
        )


def rank(recommendations: Iterable[Recommendation]) -> tuple[Recommendation, ...]:
    """Best first, deterministically — two runs must order identically."""
    return tuple(
        sorted(
            recommendations,
            key=lambda item: (-item.priority[0], -item.priority[1], item.priority[2]),
        )
    )


@dataclass(frozen=True, slots=True)
class ColumnDemand:
    """How much a workload asks of one column, and in what shape.

    This is the input both sort-key rules consume — and the point at which they
    deliberately diverge (SPEC §9.2.B): ClickHouse orders by cardinality because
    its sparse index prunes on long runs, Databricks by frequency because liquid
    clustering does not work that way.
    """

    column: str
    filters: int = 0
    equality: int = 0
    ranges: int = 0
    groups: int = 0
    text_searches: int = 0
    wrapped: int = 0
    """Times the column was reached only through a function call — the shape that
    prunes nothing and that the rewrite rules exist to fix."""

    total_queries: int = 1

    @property
    def share(self) -> float:
        """Fraction of the queries considered that filter on this column."""
        return self.filters / self.total_queries if self.total_queries else 0.0

    def merged(self, other: ColumnDemand) -> ColumnDemand:
        """Two observations of one column, added. Demand accumulates; nothing mutates."""
        return ColumnDemand(
            column=self.column,
            filters=self.filters + other.filters,
            equality=self.equality + other.equality,
            ranges=self.ranges + other.ranges,
            groups=self.groups + other.groups,
            text_searches=self.text_searches + other.text_searches,
            wrapped=self.wrapped + other.wrapped,
            total_queries=self.total_queries + other.total_queries,
        )


@dataclass(frozen=True, slots=True)
class Demand:
    """What a relation's queries ask of it, column by column."""

    relation: str
    columns: Mapping[str, ColumnDemand] = field(default_factory=dict)
    queries: int = 0
    group_by_shapes: tuple[tuple[str, ...], ...] = ()
    """Distinct ``GROUP BY`` column tuples seen, most recent last. A projection is
    proposed for a *recurring* shape, so the shapes are kept rather than counted."""

    def filtered(self) -> tuple[ColumnDemand, ...]:
        """Columns any query filtered on, most demanded first."""
        return tuple(
            sorted(
                (demand for demand in self.columns.values() if demand.filters),
                key=lambda demand: (-demand.filters, demand.column),
            )
        )

    def of(self, column: str) -> ColumnDemand:
        return self.columns.get(column, ColumnDemand(column=column, total_queries=self.queries))


def demand_from_queries(
    relation: str, shapes: Sequence[QueryShape], engine_calls: Sequence[int] | None = None
) -> Demand:
    """Aggregate what a set of parsed queries asks of ``relation``.

    ``engine_calls`` weights each shape by how many times the workload log saw
    it, so one query run ten thousand times outweighs nine run once. Without it
    every shape counts once, which is the right default for the single query an
    agent is holding.
    """
    weights = tuple(engine_calls) if engine_calls is not None else (1,) * len(shapes)
    columns: dict[str, ColumnDemand] = {}
    group_shapes: list[tuple[str, ...]] = []
    counted = 0

    for shape, weight in zip(shapes, weights, strict=True):
        if relation not in shape.tables:
            continue
        counted += weight
        if shape.group_by_columns:
            group_shapes.append(shape.group_by_columns)
        for column in _mentioned(shape):
            observed = ColumnDemand(
                column=column,
                filters=weight if column in shape.filter_columns else 0,
                equality=weight if column in shape.equality_columns else 0,
                ranges=weight if column in shape.range_columns else 0,
                groups=weight if column in shape.group_by_columns else 0,
                text_searches=weight if column in shape.text_search_columns else 0,
                wrapped=weight if column in shape.wrapped_filter_columns else 0,
                total_queries=weight,
            )
            existing = columns.get(column)
            columns[column] = observed if existing is None else existing.merged(observed)

    return Demand(
        relation=relation,
        columns={
            name: ColumnDemand(
                column=demand.column,
                filters=demand.filters,
                equality=demand.equality,
                ranges=demand.ranges,
                groups=demand.groups,
                text_searches=demand.text_searches,
                wrapped=demand.wrapped,
                total_queries=counted,
            )
            for name, demand in columns.items()
        },
        queries=counted,
        group_by_shapes=tuple(group_shapes),
    )


def workload_shapes(
    entries: Sequence[WorkloadEntry], engine: str
) -> tuple[tuple[QueryShape, ...], tuple[int, ...]]:
    """Parse a mined workload into shapes and their call counts.

    Entries whose SQL will not parse are dropped rather than guessed at: a
    misparsed workload would move a sort-key recommendation toward a column
    nobody actually filters on, which is worse than advising nothing.
    """
    shapes: list[QueryShape] = []
    calls: list[int] = []
    for entry in entries:
        shape = analyze(entry.sample_sql or entry.normalized_sql, engine)
        if not shape.parsed:
            continue
        shapes.append(shape)
        calls.append(max(entry.calls, 1))
    return tuple(shapes), tuple(calls)


def _mentioned(shape: QueryShape) -> frozenset[str]:
    return frozenset(shape.filter_columns) | frozenset(shape.group_by_columns)
