"""Value objects rendered as the JSON the tool schemas promise (SPEC §13.1).

Written out by hand, one function per type, for the same reason the schemas are:
what crosses the wire is an interface, and an interface that changes shape
whenever someone adds a private field to a dataclass is not one. The contract
tests validate everything here against :mod:`agentdb.server.schemas`, so the two
halves cannot drift apart in silence.

One rule runs through all of it: ``None`` is preserved. A field the engine could
not report stays ``null`` rather than becoming ``0`` or ``""``, because an agent
that cannot tell "no skip indexes" from "we did not look" will eventually make a
confident claim about a table nobody profiled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from agentdb.adapters.models import (
    ColumnDef,
    ColumnProfile,
    DialectRules,
    PhysicalLayout,
    Projection,
    Relation,
    RelationDetail,
    RelationRef,
    ResultSet,
    SkipIndex,
    WorkloadEntry,
)
from agentdb.core.context import GroundedContext, RelationContext
from agentdb.core.plan_ir import PlanNode, PlanSummary, PlanWarning
from agentdb.server.schemas import JsonValue


def relation_ref(ref: RelationRef) -> dict[str, JsonValue]:
    """A reference plus the name to actually write into SQL.

    ``fqn`` is not redundant with the parts beside it: on Unity Catalog a
    two-part name resolves against session state the stateless server does not
    have, so the tool hands the agent the full name rather than the pieces to
    reassemble (SPEC §8.2).
    """
    return {
        "catalog": ref.catalog,
        "namespace": ref.namespace,
        "name": ref.name,
        "fqn": str(ref),
    }


def relation(value: Relation) -> dict[str, JsonValue]:
    return {
        "ref": relation_ref(value.ref),
        "kind": value.kind,
        "engine_type": value.engine_type,
        "approx_rows": value.approx_rows,
        "on_disk_bytes": value.on_disk_bytes,
        "comment": value.comment,
    }


def column_def(value: ColumnDef, ordinal: int) -> dict[str, JsonValue]:
    """One column. ``ordinal`` is passed in because it is a fact about the *list*."""
    return {
        "name": value.name,
        "data_type": value.data_type,
        "is_nullable": value.is_nullable,
        "ordinal": ordinal,
        "default_expression": value.default_expression,
        "comment": value.comment,
    }


def relation_detail(value: RelationDetail) -> dict[str, JsonValue]:
    return {
        "ref": relation_ref(value.ref),
        "columns": [
            column_def(column, ordinal) for ordinal, column in enumerate(value.columns, start=1)
        ],
        "create_statement": value.create_statement,
        "comment": value.comment,
    }


def skip_index(value: SkipIndex) -> dict[str, JsonValue]:
    return {
        "name": value.name,
        "index_type": value.index_type,
        "expression": value.expression,
        "granularity": value.granularity,
        "compressed_bytes": value.compressed_bytes,
    }


def projection(value: Projection) -> dict[str, JsonValue]:
    return {"name": value.name, "query": value.query}


def physical_layout(value: PhysicalLayout) -> dict[str, JsonValue]:
    return {
        "engine": value.engine,
        "ref": relation_ref(value.ref),
        "create_statement": value.create_statement,
        "table_engine": value.table_engine,
        "order_by": _optional_list(value.order_by),
        "partition_by": _optional_list(value.partition_by),
        "primary_key": _optional_list(value.primary_key),
        "sampling_key": value.sampling_key,
        "skip_indexes": [skip_index(index) for index in value.skip_indexes],
        "projections": [projection(item) for item in value.projections],
        "ttl": value.ttl,
        "table_format": value.table_format,
        "clustering_columns": _optional_list(value.clustering_columns),
        "zorder_columns": _optional_list(value.zorder_columns),
        "is_managed": value.is_managed,
        "deletion_vectors_enabled": value.deletion_vectors_enabled,
        "num_files": value.num_files,
        "avg_file_bytes": value.avg_file_bytes,
        "stats_indexed_columns": value.stats_indexed_columns,
        "stats_columns": _optional_list(value.stats_columns),
        "approx_rows": value.approx_rows,
        "on_disk_bytes": value.on_disk_bytes,
        "compression_ratio": value.compression_ratio,
    }


def column_profile(value: ColumnProfile) -> dict[str, JsonValue]:
    return {
        "name": value.name,
        "data_type": value.data_type,
        "sample_method": value.sample_method,
        "sampled_rows": value.sampled_rows,
        "approx_distinct": value.approx_distinct,
        "null_ratio": value.null_ratio,
        "min_value": value.min_value,
        "max_value": value.max_value,
        "top_values": [{"value": item, "count": count} for item, count in value.top_values],
        "avg_bytes": value.avg_bytes,
    }


def dialect_rules(value: DialectRules) -> dict[str, JsonValue]:
    """Reserved words are sorted: a set's iteration order is not an interface."""
    return {
        "engine": value.engine,
        "version": value.version,
        "identifier_quote": value.identifier_quote,
        "string_quote": value.string_quote,
        "supports_ilike": value.supports_ilike,
        "reserved_words": strings(sorted(value.reserved_words)),
        "quirks": strings(value.quirks),
    }


def plan_warning(value: PlanWarning) -> dict[str, JsonValue]:
    return {
        "code": value.code.value,
        "severity": value.severity.value,
        "human_message": value.human_message,
        "relation": value.relation,
        "columns": list(value.columns),
        "suggested_rewrite": value.suggested_rewrite,
    }


def plan_node(value: PlanNode) -> dict[str, JsonValue]:
    """One node and its subtree.

    ``pruning_ratio`` is computed here rather than left to the client: it is the
    number the whole plan IR exists to carry, and a client that has to divide two
    nullable fields to get it will sooner or later divide by a zero total.
    """
    return {
        "op": value.op.value,
        "node_type": value.node_type,
        "relation": value.relation,
        "estimated_rows": value.estimated_rows,
        "actual_rows": value.actual_rows,
        "estimated_cost": value.estimated_cost,
        "filters": list(value.filters),
        "granules_total": value.granules_total,
        "granules_selected": value.granules_selected,
        "parts_total": value.parts_total,
        "parts_selected": value.parts_selected,
        "index_used": list(value.index_used),
        "projection_used": value.projection_used,
        "files_total": value.files_total,
        "files_selected": value.files_selected,
        "partitions_total": value.partitions_total,
        "partitions_selected": value.partitions_selected,
        "partition_filters": list(value.partition_filters),
        "pushed_filters": list(value.pushed_filters),
        "data_filters": list(value.data_filters),
        "photon": value.photon,
        "join_strategy": value.join_strategy,
        "pruning_ratio": value.pruning_ratio,
        "children": [plan_node(child) for child in value.children],
    }


def plan_summary(value: PlanSummary) -> dict[str, JsonValue]:
    """The plan, plus the same short text a grounded arm would put in a prompt."""
    return {
        "engine": value.engine,
        "sql": value.sql,
        "root": plan_node(value.root),
        "pruning_ratio": value.pruning_ratio,
        "pruning_unit": value.pruning_unit,
        "pruning_source": value.pruning_source,
        "full_scan_relations": list(value.full_scan_relations),
        "estimated_bytes_read": value.estimated_bytes_read,
        "measured_bytes_read": value.measured_bytes_read,
        "bytes_ratio": value.bytes_ratio,
        "photon_coverage": value.photon_coverage,
        "warnings": [plan_warning(warning) for warning in value.warnings],
        "rendered": value.render(),
    }


def result_set(value: ResultSet) -> dict[str, JsonValue]:
    return {
        "columns": list(value.columns),
        "rows": [[_cell(cell) for cell in row] for row in value.rows],
        "row_count": value.row_count,
        "truncated": value.truncated,
        "duration_ms": value.duration_ms,
        "rows_read": value.rows_read,
        "bytes_read": value.bytes_read,
        "query_id": value.query_id,
    }


def workload_entry(value: WorkloadEntry) -> dict[str, JsonValue]:
    return {
        "normalized_sql": value.normalized_sql,
        "calls": value.calls,
        "relations": list(value.relations),
        "total_duration_ms": value.total_duration_ms,
        "mean_duration_ms": value.mean_duration_ms,
        "rows_read": value.rows_read,
        "bytes_read": value.bytes_read,
        "sample_sql": value.sample_sql,
    }


def relation_context(value: RelationContext) -> dict[str, JsonValue]:
    return {
        "detail": relation_detail(value.detail),
        "layout": physical_layout(value.layout) if value.layout is not None else None,
        "profiles": [column_profile(profile) for profile in value.profiles],
        "profiled_columns_available": value.profiled_columns_available,
    }


def grounded_context(value: GroundedContext) -> dict[str, JsonValue]:
    """The assembled bundle, both structured and rendered.

    Both, because they serve different callers: an agent loop pastes ``rendered``
    into a prompt, and a program that wants the sort key reads ``relations``. The
    two are generated from one object, so they cannot disagree.
    """
    return {
        "engine": value.engine,
        "namespace": value.namespace,
        "level": value.level.value,
        "relations": [relation_context(item) for item in value.relations],
        "rendered": value.render(),
        "size_bytes": value.size_bytes,
    }


def strings(values: Iterable[str]) -> list[JsonValue]:
    """A list of strings, typed as JSON. Python's invariance needs it said out loud."""
    return list(values)


def _optional_list(value: Sequence[str] | None) -> JsonValue:
    """Keep ``None`` distinct from ``[]``: unknown is not the same as empty."""
    return None if value is None else strings(value)


def _cell(value: object) -> JsonValue:
    """One result cell as JSON.

    Anything JSON has no room for — a date, a Decimal, a UUID — becomes its
    string form rather than being dropped or coerced to a float. Losing
    precision on a Decimal silently would corrupt exactly the aggregates this
    benchmark grades.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def json_rows(values: Iterable[Relation]) -> list[JsonValue]:
    """Serialize a relation listing. Named for the shape, used by the tools."""
    return [relation(value) for value in values]
