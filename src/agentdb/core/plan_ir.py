"""The engine-neutral plan representation (SPEC §7).

Both engines' plans normalize into one tree so the core reasons once. The tree
exists for one purpose above all others: to carry **pruning evidence**. How many
granules and parts the engine started with, how many survived, and which index
did the surviving — those numbers are what turn "your query is slow" into "your
filter does not touch the leading sort-key column, so nothing was pruned".

:class:`PlanWarning` is the product. A warning names the relation and columns,
says what is wrong in a sentence a model can act on, and carries a rewrite when
one is mechanically derivable. Everything else here exists to make those
warnings computable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PlanOp(StrEnum):
    """What a plan node does, in terms both engines share."""

    SCAN = "scan"
    """Reads a relation. The node that carries the pruning evidence."""

    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    SORT = "sort"
    LIMIT = "limit"
    EXCHANGE = "exchange"
    """Data movement between shards, replicas or pipeline stages."""

    PROJECTION_READ = "projection_read"
    """A ClickHouse projection served the query instead of the base table."""

    OTHER = "other"
    """A node whose type this IR does not model.

    Kept rather than dropped: a plan with an unnamed node is still a plan, and
    silently deleting nodes would make the tree a lie about what ran.
    """


class WarningCode(StrEnum):
    """The actionable findings. This list is the product (SPEC §7)."""

    FULL_SCAN = "FULL_SCAN"
    SORT_KEY_UNUSED = "SORT_KEY_UNUSED"
    SORT_KEY_PREFIX_SKIPPED = "SORT_KEY_PREFIX_SKIPPED"
    PROJECTION_AVAILABLE_UNUSED = "PROJECTION_AVAILABLE_UNUSED"
    HIGH_CARD_GROUP_BY = "HIGH_CARD_GROUP_BY"
    JOIN_ORDER_SUSPECT = "JOIN_ORDER_SUSPECT"
    MISSING_PARTITION_PREDICATE = "MISSING_PARTITION_PREDICATE"
    SELECT_STAR_WIDE = "SELECT_STAR_WIDE"
    NO_LIMIT_UNBOUNDED = "NO_LIMIT_UNBOUNDED"
    NULLABLE_IN_KEY = "NULLABLE_IN_KEY"


class Severity(StrEnum):
    """How loudly a warning asks to be acted on.

    Three levels, not five: a scale an agent cannot act on differently is a
    scale that costs tokens and buys nothing.
    """

    INFO = "info"
    """Worth knowing; the query is correct and probably fine."""

    WARNING = "warning"
    """The query will run but reads far more than it needs to."""

    CRITICAL = "critical"
    """The query is likely to time out or exhaust memory at this data size."""


@dataclass(frozen=True, slots=True)
class PlanWarning:
    """One actionable finding about a plan."""

    code: WarningCode
    severity: Severity
    human_message: str
    """One sentence, addressed to whoever writes the next query."""

    relation: str | None = None
    columns: tuple[str, ...] = ()
    suggested_rewrite: str | None = None
    """A concrete rewrite where one follows mechanically, otherwise ``None``.
    Never a guess: a wrong rewrite costs more trust than a missing one."""


@dataclass(frozen=True, slots=True)
class PlanNode:
    """One node of a normalized plan tree."""

    op: PlanOp
    node_type: str
    """The engine's own name for this node, kept verbatim for auditability."""

    relation: str | None = None
    estimated_rows: int | None = None
    actual_rows: int | None = None
    """Measured rows. ``None`` on ClickHouse, whose EXPLAIN cannot execute."""

    estimated_cost: float | None = None
    filters: tuple[str, ...] = ()

    # pruning evidence — the whole point of the IR
    granules_total: int | None = None
    granules_selected: int | None = None
    parts_total: int | None = None
    parts_selected: int | None = None
    index_used: tuple[str, ...] = ()
    """Names of the primary and skip indexes that actually fired."""

    projection_used: str | None = None
    children: tuple[PlanNode, ...] = ()

    def walk(self) -> tuple[PlanNode, ...]:
        """This node and every descendant, depth-first."""
        return (self, *tuple(node for child in self.children for node in child.walk()))

    @property
    def pruning_ratio(self) -> float | None:
        """Granules kept over granules considered, or ``None`` when unmeasured.

        Low is good: 0.01 means the index threw away 99% of the data before
        reading it. A ratio of 1.0 means the sort key did nothing.
        """
        if not self.granules_total:
            return None
        if self.granules_selected is None:
            return None
        return self.granules_selected / self.granules_total


@dataclass(frozen=True, slots=True)
class PlanSummary:
    """A whole plan, plus what is worth saying about it."""

    root: PlanNode
    engine: str
    sql: str
    pruning_ratio: float | None = None
    """Aggregated over every scan: total selected granules over total considered."""

    full_scan_relations: tuple[str, ...] = ()
    estimated_bytes_read: int | None = None
    warnings: tuple[PlanWarning, ...] = ()

    @property
    def scans(self) -> tuple[PlanNode, ...]:
        """Every relation-reading node, in plan order."""
        return tuple(node for node in self.root.walk() if node.op is PlanOp.SCAN)

    def with_warnings(self, warnings: tuple[PlanWarning, ...]) -> PlanSummary:
        """A copy carrying ``warnings``. Summaries are built, never mutated."""
        return PlanSummary(
            root=self.root,
            engine=self.engine,
            sql=self.sql,
            pruning_ratio=self.pruning_ratio,
            full_scan_relations=self.full_scan_relations,
            estimated_bytes_read=self.estimated_bytes_read,
            warnings=warnings,
        )

    def render(self) -> str:
        """The summary as text for an agent's context.

        Deliberately short. A plan dump is not context, it is noise; what the
        agent can act on is the pruning number and the warnings.
        """
        lines = [f"Plan summary ({self.engine}):"]
        if self.pruning_ratio is not None:
            kept = f"{self.pruning_ratio:.1%}"
            lines.append(f"- granules read after pruning: {kept} of those considered")
        for relation in self.full_scan_relations:
            lines.append(f"- full scan: {relation}")
        if self.estimated_bytes_read is not None:
            lines.append(f"- estimated bytes read: {self.estimated_bytes_read:,}")
        for warning in self.warnings:
            lines.append(_render_warning(warning))
        if len(lines) == 1:
            lines.append("- no pruning evidence reported by the engine")
        return "\n".join(lines)


def _render_warning(warning: PlanWarning) -> str:
    parts = [f"- [{warning.severity.value}] {warning.code.value}: {warning.human_message}"]
    if warning.suggested_rewrite is not None:
        parts.append(f"  try: {warning.suggested_rewrite}")
    return "\n".join(parts)
