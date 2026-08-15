"""The Databricks rules that turn a plan into advice (SPEC §7, §8.2).

Same discipline as the ClickHouse rules: deterministic, citing the facts they
fired on, silent when those facts are missing. What differs is the mechanism they
reason about. ClickHouse prunes granules through a sort key; Delta prunes files
through a clustering key *and* through per-file statistics that exist for only a
bounded set of columns. The second half has no ClickHouse counterpart and is the
highest-value warning in this module: a filter on an unindexed column of a wide
table skips no files at all, however selective it is, and neither a schema dump
nor the query itself reveals that.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentdb.adapters import PhysicalLayout
from agentdb.config import Config
from agentdb.core.plan_ir import PlanOp, PlanSummary, PlanWarning, Severity, WarningCode
from agentdb.core.query_shape import QueryShape, mentions

UNQUALIFIED_NAME_PARTS = 3
"""Unity Catalog names have three parts. Fewer resolves against session state."""


def relation_warnings(
    summary: PlanSummary,
    shape: QueryShape,
    relation: str,
    layout: PhysicalLayout,
    column_ordinals: Mapping[str, int],
    config: Config,
) -> list[PlanWarning]:
    """Every Databricks warning that this relation's evidence supports."""
    warnings: list[PlanWarning] = []
    warnings.extend(_clustering_warning(shape, relation, layout))
    warnings.extend(_statistics_warning(shape, relation, layout, column_ordinals, config))
    warnings.extend(_partition_warning(summary, relation, layout))
    warnings.extend(_small_files_warning(relation, layout, config))
    return warnings


def plan_warnings(summary: PlanSummary, config: Config) -> list[PlanWarning]:
    """Warnings about the plan as a whole rather than about one relation."""
    return _photon_warning(summary, config)


def query_warnings(shape: QueryShape) -> list[PlanWarning]:
    """Warnings readable from the query text alone.

    Under-qualification is a correctness hazard, not a performance one, so it is
    checked even when the plan is unavailable — an agent that wrote
    ``FROM lineitem`` may have read a different table than the one it meant.
    """
    if not shape.parsed:
        return []
    unqualified = [
        table for table in shape.qualified_tables if len(table.split(".")) < UNQUALIFIED_NAME_PARTS
    ]
    if not unqualified:
        return []
    return [
        PlanWarning(
            code=WarningCode.UNQUALIFIED_RELATION,
            severity=Severity.CRITICAL,
            columns=tuple(unqualified),
            human_message=(
                f"{', '.join(unqualified)} is referenced with fewer than three name parts; "
                "under Unity Catalog that resolves against the session's USE context and may "
                "read a different table than the one intended"
            ),
            suggested_rewrite="qualify every table as catalog.schema.table",
        )
    ]


def _clustering_warning(
    shape: QueryShape, relation: str, layout: PhysicalLayout
) -> list[PlanWarning]:
    """Whether the filters can reach the clustering key at all.

    The direct analogue of ``SORT_KEY_UNUSED``: Delta skips files by comparing a
    predicate against per-file statistics, and clustering is what makes those
    statistics selective. A filter on a non-clustered column reads every file the
    partition predicate left behind.
    """
    keys = layout.clustering_columns or layout.zorder_columns
    if not shape.parsed or not keys or not shape.filter_columns:
        return []
    if any(mentions(key, shape.filter_columns) for key in keys):
        return []
    return [
        PlanWarning(
            code=WarningCode.CLUSTERING_KEY_UNUSED,
            severity=Severity.WARNING,
            relation=relation,
            columns=tuple(sorted(shape.filter_columns)),
            human_message=(
                f"none of the filtered columns are in the clustering key of {relation} "
                f"({', '.join(keys)}), so the scan opens every file"
            ),
            suggested_rewrite=(
                f"add a predicate on {keys[0]} if the question allows one, even a wide range"
            ),
        )
    ]


def _statistics_warning(
    shape: QueryShape,
    relation: str,
    layout: PhysicalLayout,
    column_ordinals: Mapping[str, int],
    config: Config,
) -> list[PlanWarning]:
    """Filters on columns Delta collects no statistics for.

    This is the silent one. ``delta.dataSkippingNumIndexedCols`` defaults to 32,
    so on a wide table a filter on column 40 cannot skip a single file — and
    nothing in the query, the schema, or the error output says so.
    """
    if not shape.parsed or not shape.filter_columns or not column_ordinals:
        return []

    unskippable = sorted(
        column
        for column in shape.filter_columns
        if column in column_ordinals
        and not layout.has_file_statistics(
            column, column_ordinals[column], default_indexed=config.delta_default_stats_columns
        )
    )
    if not unskippable:
        return []

    limit = layout.stats_indexed_columns or config.delta_default_stats_columns
    reason = (
        f"the table collects statistics only for {', '.join(layout.stats_columns)}"
        if layout.stats_columns
        else f"Delta indexes only the first {limit} columns in schema order"
    )
    return [
        PlanWarning(
            code=WarningCode.STATS_NOT_COLLECTED,
            severity=Severity.CRITICAL,
            relation=relation,
            columns=tuple(unskippable),
            human_message=(
                f"{', '.join(unskippable)} has no per-file statistics on {relation} ({reason}), "
                "so filtering on it cannot skip any file"
            ),
            suggested_rewrite=(
                "filter on an indexed column as well, or ask the table owner to set "
                "delta.dataSkippingStatsColumns"
            ),
        )
    ]


def _partition_warning(
    summary: PlanSummary, relation: str, layout: PhysicalLayout
) -> list[PlanWarning]:
    """A partitioned table scanned with no partition predicate *pushed down*.

    Read from the plan rather than from the query, because Databricks silently
    declines to push a predicate wrapped in a function: ``year(ts) = 2026`` looks
    like a partition filter and prunes nothing (SPEC §7).
    """
    if not layout.partition_by:
        return []
    scans = [
        node
        for node in summary.root.walk()
        if node.op is PlanOp.SCAN and _same_relation(node.relation, relation)
    ]
    if not scans or any(scan.partition_filters for scan in scans):
        return []
    return [
        PlanWarning(
            code=WarningCode.MISSING_PARTITION_PREDICATE,
            severity=Severity.WARNING,
            relation=relation,
            columns=layout.partition_by,
            human_message=(
                f"{relation} is partitioned by {', '.join(layout.partition_by)} and the plan "
                "pushed no partition filter, so every partition is opened; a predicate wrapped "
                "in a function does not push down"
            ),
            suggested_rewrite=(
                f"write a range on {layout.partition_by[0]} itself rather than on a function of it"
            ),
        )
    ]


def _small_files_warning(
    relation: str, layout: PhysicalLayout, config: Config
) -> list[PlanWarning]:
    """A fragmented table, where per-file overhead dominates and skipping is coarse."""
    if layout.num_files is None or layout.avg_file_bytes is None:
        return []
    if (
        layout.num_files <= config.small_file_count_threshold
        or layout.avg_file_bytes >= config.small_file_bytes_threshold
    ):
        return []
    return [
        PlanWarning(
            code=WarningCode.SMALL_FILES,
            severity=Severity.INFO,
            relation=relation,
            human_message=(
                f"{relation} holds {layout.num_files:,} files averaging "
                f"{layout.avg_file_bytes / 1024 / 1024:.1f} MiB; per-file overhead dominates and "
                "file-level skipping is coarse"
            ),
            suggested_rewrite=(
                "this is a table-maintenance issue, not a query one: OPTIMIZE the table, or "
                "enable predictive optimization if it is managed"
            ),
        )
    ]


def _photon_warning(summary: PlanSummary, config: Config) -> list[PlanWarning]:
    """Part of the query shape fell off the vectorized engine.

    Photon fallback is never reported as such — it is the absence of a ``Photon``
    prefix on a node — so the warning names the nodes that lacked it and says
    that the reading is an inference.
    """
    if summary.photon_coverage is None:
        return []
    if summary.photon_coverage >= config.photon_coverage_threshold:
        return []
    fell_back = [node.node_type for node in summary.root.walk() if node.photon is False]
    return [
        PlanWarning(
            code=WarningCode.PHOTON_FALLBACK,
            severity=Severity.WARNING,
            columns=tuple(fell_back),
            human_message=(
                f"only {summary.photon_coverage:.0%} of plan nodes ran on Photon; "
                f"{', '.join(fell_back)} did not, inferred from the node names — Databricks "
                "reports a fallback only by omitting the Photon prefix"
            ),
        )
    ]


def _same_relation(candidate: str | None, relation: str) -> bool:
    if candidate is None:
        return False
    return candidate == relation or candidate.rpartition(".")[2] == relation.rpartition(".")[2]
