"""The rules that turn a plan into advice (SPEC §7).

Each rule is deterministic, cites the facts it fired on, and stays silent when
those facts are missing. That last property is the important one: an agent that
receives a confident warning derived from an absent statistic learns to ignore
warnings, and the whole layer stops paying for its tokens.

None of this is an LLM asking an LLM. A rule is a comparison between what the
query asked for, what the physical design supports, and what the engine's own
plan says happened.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agentdb.adapters import ColumnProfile, PhysicalLayout
from agentdb.config import Config
from agentdb.core import plan_rules_databricks as databricks
from agentdb.core.plan_ir import PlanSummary, PlanWarning, Severity, WarningCode
from agentdb.core.query_shape import QueryShape, mentions


@dataclass(frozen=True, slots=True)
class RelationFacts:
    """What the rules know about one relation."""

    layout: PhysicalLayout
    column_count: int = 0
    profiles: Mapping[str, ColumnProfile] = field(default_factory=dict)
    column_ordinals: Mapping[str, int] = field(default_factory=dict)
    """1-based schema positions. Delta's statistics stop at an ordinal, so this is
    what makes ``STATS_NOT_COLLECTED`` computable; ClickHouse ignores it."""

    @property
    def approx_rows(self) -> int | None:
        return self.layout.approx_rows


def evaluate(
    summary: PlanSummary,
    shape: QueryShape,
    facts: Mapping[str, RelationFacts],
    config: Config,
) -> PlanSummary:
    """Return ``summary`` carrying every warning its evidence supports.

    ``facts`` is keyed by relation name; a qualified ``db.table`` in a plan is
    matched on its last component, because the plan and the catalogue do not
    always agree on qualification.
    """
    warnings: list[PlanWarning] = []
    for relation, relation_facts in _relations_in_play(summary, shape, facts):
        warnings.extend(_relation_warnings(summary, shape, relation, relation_facts, config))
    if summary.engine == "databricks":
        warnings.extend(databricks.query_warnings(shape))
        warnings.extend(databricks.plan_warnings(summary, config))
    else:
        warnings.extend(_join_warnings(shape, facts))
    warnings.extend(_result_size_warnings(summary, shape, facts, config))
    return summary.with_warnings(tuple(warnings))


def _relations_in_play(
    summary: PlanSummary, shape: QueryShape, facts: Mapping[str, RelationFacts]
) -> list[tuple[str, RelationFacts]]:
    """Relations the plan scanned, falling back to the ones the query named."""
    scanned = [node.relation for node in summary.scans if node.relation is not None]
    names = scanned or list(shape.tables)
    resolved: list[tuple[str, RelationFacts]] = []
    for name in names:
        known = facts.get(name) or facts.get(name.rpartition(".")[2])
        if known is not None:
            resolved.append((name, known))
    return resolved


def _relation_warnings(
    summary: PlanSummary,
    shape: QueryShape,
    relation: str,
    facts: RelationFacts,
    config: Config,
) -> list[PlanWarning]:
    warnings: list[PlanWarning] = []
    layout = facts.layout
    rows = facts.approx_rows or 0

    if (
        summary.pruning_ratio is not None
        and summary.pruning_ratio > config.pruning_ratio_threshold
        and rows > config.full_scan_row_threshold
    ):
        warnings.append(
            PlanWarning(
                code=WarningCode.FULL_SCAN,
                severity=Severity.WARNING,
                relation=relation,
                human_message=(
                    f"the scan of {relation} pruned almost nothing: "
                    f"{summary.pruning_ratio:.0%} of {summary.pruning_unit or 'unit'}s were "
                    f"read, over ~{rows:,} rows"
                ),
                suggested_rewrite=_pruning_key_hint(layout),
            )
        )

    if summary.engine == "databricks":
        warnings.extend(
            databricks.relation_warnings(
                summary, shape, relation, layout, facts.column_ordinals, config
            )
        )
    else:
        warnings.extend(_sort_key_warnings(shape, relation, layout))
        warnings.extend(_partition_warning(shape, relation, layout))
        warnings.extend(_projection_warning(summary, shape, relation, layout))
    warnings.extend(_group_by_warnings(shape, relation, facts, config))

    wide = facts.column_count > config.wide_table_column_threshold
    if shape.parsed and shape.selects_star and wide:
        warnings.append(
            PlanWarning(
                code=WarningCode.SELECT_STAR_WIDE,
                severity=Severity.WARNING,
                relation=relation,
                human_message=(
                    f"SELECT * reads all {facts.column_count} columns of {relation}; a columnar "
                    "engine only pays for the columns named"
                ),
                suggested_rewrite="name the columns the answer needs instead of *",
            )
        )
    return warnings


def _pruning_key_hint(layout: PhysicalLayout) -> str | None:
    """The one rewrite that follows mechanically from a layout: filter the key.

    Which key depends on the engine, and so does the unit it prunes: ClickHouse
    prunes granules through the leading sort-key column, Delta prunes files
    through the clustering key.
    """
    if layout.order_by:
        return f"filter on {layout.order_by[0]}, the leading sort-key column, to prune granules"
    if layout.clustering_columns:
        return f"filter on {layout.clustering_columns[0]}, a clustering key column, to skip files"
    return None


def _sort_key_warnings(
    shape: QueryShape, relation: str, layout: PhysicalLayout
) -> list[PlanWarning]:
    """Whether the filters can reach the sort key at all.

    ClickHouse prunes granules through the ``ORDER BY`` key left to right. A
    filter on the second key column with nothing on the first prunes nothing —
    the single most consequential fact a schema dump cannot convey.
    """
    if not shape.parsed or not layout.order_by or not shape.filter_columns:
        return []

    leading = layout.order_by[0]
    if mentions(leading, shape.filter_columns):
        return []

    later = [column for column in layout.order_by[1:] if mentions(column, shape.filter_columns)]
    if later:
        return [
            PlanWarning(
                code=WarningCode.SORT_KEY_PREFIX_SKIPPED,
                severity=Severity.CRITICAL,
                relation=relation,
                columns=tuple(later),
                human_message=(
                    f"the filter reaches {', '.join(later)} but not the leading sort-key column "
                    f"{leading}, so no granules can be pruned"
                ),
                suggested_rewrite=(
                    f"add a predicate on {leading} if the question allows one, even a wide range"
                ),
            )
        ]

    return [
        PlanWarning(
            code=WarningCode.SORT_KEY_UNUSED,
            severity=Severity.WARNING,
            relation=relation,
            columns=tuple(sorted(shape.filter_columns)),
            human_message=(
                f"none of the filtered columns are in the sort key of {relation} "
                f"({', '.join(layout.order_by)}), so the scan reads every granule"
            ),
        )
    ]


def _partition_warning(
    shape: QueryShape, relation: str, layout: PhysicalLayout
) -> list[PlanWarning]:
    if not shape.parsed or not layout.partition_by or not shape.filter_columns:
        return []
    if any(mentions(term, shape.filter_columns) for term in layout.partition_by):
        return []
    return [
        PlanWarning(
            code=WarningCode.MISSING_PARTITION_PREDICATE,
            severity=Severity.WARNING,
            relation=relation,
            columns=layout.partition_by,
            human_message=(
                f"{relation} is partitioned by {', '.join(layout.partition_by)} and the query "
                "constrains none of it, so every partition is opened"
            ),
        )
    ]


def _projection_warning(
    summary: PlanSummary, shape: QueryShape, relation: str, layout: PhysicalLayout
) -> list[PlanWarning]:
    """A projection that matches this query's shape but did not serve it."""
    if not shape.parsed or not layout.projections or not shape.group_by_columns:
        return []
    if any(node.projection_used for node in summary.scans):
        return []

    matching = [
        projection.name
        for projection in layout.projections
        if all(column in projection.query for column in shape.group_by_columns)
    ]
    if not matching:
        return []
    return [
        PlanWarning(
            code=WarningCode.PROJECTION_AVAILABLE_UNUSED,
            severity=Severity.INFO,
            relation=relation,
            columns=shape.group_by_columns,
            human_message=(
                f"{relation} has projection(s) {', '.join(matching)} covering "
                f"{', '.join(shape.group_by_columns)}, but the plan reads the base table"
            ),
        )
    ]


def _group_by_warnings(
    shape: QueryShape, relation: str, facts: RelationFacts, config: Config
) -> list[PlanWarning]:
    warnings: list[PlanWarning] = []
    for column in shape.group_by_columns:
        profile = facts.profiles.get(column)
        if profile is None:
            continue
        if (
            profile.approx_distinct is not None
            and profile.approx_distinct > config.high_card_threshold
        ):
            warnings.append(
                PlanWarning(
                    code=WarningCode.HIGH_CARD_GROUP_BY,
                    severity=Severity.WARNING,
                    relation=relation,
                    columns=(column,),
                    human_message=(
                        f"GROUP BY {column} builds roughly {profile.approx_distinct:,} groups in "
                        "memory before anything is returned"
                    ),
                    suggested_rewrite="add a LIMIT, or group by a coarser expression",
                )
            )
        if profile.null_ratio:
            warnings.append(
                PlanWarning(
                    code=WarningCode.NULLABLE_IN_KEY,
                    severity=Severity.INFO,
                    relation=relation,
                    columns=(column,),
                    human_message=(
                        f"{column} is null in {profile.null_ratio:.0%} of sampled rows and is "
                        "used as a grouping key; those rows collapse into one NULL group"
                    ),
                )
            )
    return warnings


def _join_warnings(shape: QueryShape, facts: Mapping[str, RelationFacts]) -> list[PlanWarning]:
    """ClickHouse loads the right-hand table into memory; size order matters."""
    if not shape.parsed or not shape.joined_tables or len(shape.tables) < 2:
        return []

    left = _rows(shape.tables[0], facts)
    warnings: list[PlanWarning] = []
    for table in shape.joined_tables:
        right = _rows(table, facts)
        if left is None or right is None or right <= left:
            continue
        warnings.append(
            PlanWarning(
                code=WarningCode.JOIN_ORDER_SUSPECT,
                severity=Severity.WARNING,
                relation=table,
                human_message=(
                    f"{table} (~{right:,} rows) is on the build side of the join against "
                    f"{shape.tables[0]} (~{left:,} rows); ClickHouse holds the right side in memory"
                ),
                suggested_rewrite=f"put {shape.tables[0]} on the right of the join instead",
            )
        )
    return warnings


def _result_size_warnings(
    summary: PlanSummary,
    shape: QueryShape,
    facts: Mapping[str, RelationFacts],
    config: Config,
) -> list[PlanWarning]:
    """An unbounded row-returning query on a large relation.

    ClickHouse's ``EXPLAIN`` reports no row estimate, so the projected size is
    the relation's own row count scaled by whatever the plan says was pruned.
    It is an approximation and the message says "roughly" for that reason.
    """
    if not shape.parsed or shape.has_limit or shape.has_aggregate:
        return []

    rows = max((_rows(table, facts) or 0 for table in shape.tables), default=0)
    projected = round(rows * (summary.pruning_ratio if summary.pruning_ratio is not None else 1.0))
    if projected <= config.unbounded_row_threshold:
        return []
    return [
        PlanWarning(
            code=WarningCode.NO_LIMIT_UNBOUNDED,
            severity=Severity.WARNING,
            human_message=(
                f"the query has no LIMIT and is projected to return roughly {projected:,} rows"
            ),
            suggested_rewrite="add a LIMIT, or aggregate instead of returning raw rows",
        )
    ]


def _rows(table: str, facts: Mapping[str, RelationFacts]) -> int | None:
    known = facts.get(table) or facts.get(table.rpartition(".")[2])
    return known.approx_rows if known is not None else None
