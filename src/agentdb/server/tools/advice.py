"""Advice tools: what to change about the table, as text (SPEC §13.1, §9).

Every tool here returns DDL as a *string*. agentdb never runs it: SPEC §13.3
requires an elicited confirmation before any DDL executes, and a server that
quietly altered a table because an agent asked a question would be the last time
anybody trusted it with a connection.

**Engine-specific advice, engine-neutral catalogue.** Sort keys exist on
ClickHouse and clustering keys on Databricks, but both tools are served on both
engines and the one that does not apply says so in a structured error. A client
that had to ask which engine it was talking to before knowing which tools it had
would make the capability flags decorative — the same rule the rest of the
catalogue follows.

**Where the demand signal comes from.** A recommendation about physical design is
only as good as its picture of what the table is asked. These tools take the
query at hand and, where the connection can read the engine's log, the recent
workload. That is the deployment story; the *benchmark* uses a committed
reference workload instead, because on a benchmark instance the log holds the
project's own gold executions (SPEC §11.3, arm A6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from agentdb.adapters import (
    Capability,
    ColumnProfile,
    PhysicalLayout,
    RelationDetail,
    RelationRef,
    SamplePolicy,
    TimeWindow,
)
from agentdb.core.advisor import (
    ClickHouseAdvisor,
    DatabricksAdvisor,
    Demand,
    Kind,
    Recommendation,
    ShadowError,
    ShadowValidator,
    demand_from_queries,
    measured,
    rewrites,
    workload_shapes,
)
from agentdb.core.query_shape import analyze
from agentdb.server import serialize
from agentdb.server.base import (
    ServerContext,
    ToolDef,
    ToolError,
    optional_int,
    optional_str,
    require_str,
)
from agentdb.server.schemas import JsonValue, array_of, object_schema

DEFAULT_WORKLOAD_HOURS: Final = 24 * 7
"""A week of traffic. Long enough to hold a weekly report, short enough that the
advice describes how the table is used now rather than last quarter."""

MAX_WORKLOAD_HOURS: Final = 24 * 30
WORKLOAD_TOP_N: Final = 200
"""Query shapes mined per advice call. The demand signal is a distribution, and a
couple of hundred shapes is well past where its head stops moving."""

CLICKHOUSE_KINDS: Final = {
    "advise_sort_key": Kind.ORDER_BY,
    "advise_indexes": Kind.SKIP_INDEX,
    "advise_projection": Kind.PROJECTION,
}
DATABRICKS_KINDS: Final = {
    "advise_clustering": Kind.CLUSTER_BY,
    "advise_skipping_stats": Kind.STATS_COLUMNS,
    "advise_compaction": Kind.COMPACTION,
}


@dataclass(frozen=True, slots=True)
class AdviceInputs:
    """Everything an advisor needs about one relation, gathered once."""

    ref: RelationRef
    layout: PhysicalLayout
    detail: RelationDetail
    profiles: tuple[ColumnProfile, ...]
    demand: Demand


def advice_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Build the advice group against ``context``."""
    return (
        *(_design_tool(context, name, kind) for name, kind in CLICKHOUSE_KINDS.items()),
        *(_design_tool(context, name, kind) for name, kind in DATABRICKS_KINDS.items()),
        _suggest_rewrite(context),
    )


# --------------------------------------------------------------------------
# gathering the evidence
# --------------------------------------------------------------------------


async def gather(context: ServerContext, args: Mapping[str, JsonValue]) -> AdviceInputs:
    """Read the layout, profile the filtered columns, and build the demand signal."""
    ref = context.parse_relation(require_str(args, "relation"))
    sql = optional_str(args, "sql")
    hours = optional_int(args, "hours")
    if hours is not None and hours > MAX_WORKLOAD_HOURS:
        raise ToolError(
            f"hours must be <= {MAX_WORKLOAD_HOURS}",
            suggestion="the query log has usually rotated past a month",
        )

    layout = await context.adapter.physical_layout(ref)
    detail = await context.adapter.describe_relation(ref)
    demand = await _demand(context, ref, sql, hours)
    profiles = await _profiles(context, ref, demand, detail)
    return AdviceInputs(ref=ref, layout=layout, detail=detail, profiles=profiles, demand=demand)


async def _demand(
    context: ServerContext, ref: RelationRef, sql: str | None, hours: int | None
) -> Demand:
    """What this relation is asked, from the query at hand and the log behind it."""
    shapes = []
    calls = []
    if sql is not None:
        shapes.append(analyze(sql, context.adapter.engine))
        calls.append(1)

    if context.adapter.supports(Capability.WORKLOAD_LOG):
        window_hours = hours or DEFAULT_WORKLOAD_HOURS
        end = datetime.now(UTC)
        entries = await context.adapter.workload(
            TimeWindow(start=end - timedelta(hours=window_hours), end=end), WORKLOAD_TOP_N
        )
        mined, mined_calls = workload_shapes(entries, context.adapter.engine)
        shapes.extend(mined)
        calls.extend(mined_calls)

    if not shapes:
        raise ToolError(
            "nothing to advise from: this connection cannot read the engine's query log",
            suggestion="pass the query you are asking about as 'sql'",
        )
    return demand_from_queries(ref.name, shapes, calls)


async def _profiles(
    context: ServerContext, ref: RelationRef, demand: Demand, detail: RelationDetail
) -> tuple[ColumnProfile, ...]:
    """Sample the columns the demand signal actually weighs, and no others.

    Profiling is the expensive half of advising, so it is bounded twice: to the
    filtered columns, and then to the configured column budget. A recommendation
    about a column nobody filters on is not worth a scan.
    """
    known = {column.name for column in detail.columns}
    wanted = [item.column for item in demand.filtered() if item.column in known]
    if not wanted:
        return ()

    policy = SamplePolicy(
        fraction=context.config.default_sample_fraction,
        max_rows=context.config.profile_max_rows,
        timeout_s=context.config.query_timeout_s,
    )
    profiled = await context.adapter.column_profile(
        ref, wanted[: context.config.max_profiled_columns], policy
    )
    return tuple(profiled)


def recommend(context: ServerContext, inputs: AdviceInputs) -> tuple[Recommendation, ...]:
    """Run the advisor this engine has."""
    if context.adapter.engine == "clickhouse":
        return ClickHouseAdvisor(config=context.config).advise(
            ref=inputs.ref,
            layout=inputs.layout,
            profiles=inputs.profiles,
            demand=inputs.demand,
        )
    return DatabricksAdvisor(config=context.config).advise(
        ref=inputs.ref,
        layout=inputs.layout,
        detail=inputs.detail,
        profiles=inputs.profiles,
        demand=inputs.demand,
    )


# --------------------------------------------------------------------------
# the tools
# --------------------------------------------------------------------------


def _design_tool(context: ServerContext, name: str, kind: Kind) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        _require_engine(context, name)
        inputs = await gather(context, args)
        found = [item for item in recommend(context, inputs) if item.kind is kind]
        if _wants_validation(args):
            found = await _validate(context, args, inputs, found)
        return {
            "relation": serialize.relation_ref(inputs.ref),
            "recommendations": [serialize.recommendation(item) for item in found],
            "workload_queries": inputs.demand.queries,
        }

    return ToolDef(
        name=name,
        title=_TITLES[name],
        description=_DESCRIPTIONS[name],
        input_schema=_advice_input_schema(),
        output_schema=_advice_output_schema(),
        handler=handler,
    )


def _suggest_rewrite(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        sql = require_str(args, "sql")
        ref = context.parse_relation(require_str(args, "relation"))
        rules = await context.adapter.dialect_rules()
        layout = await context.adapter.physical_layout(ref)
        detail = await context.adapter.describe_relation(ref)
        found = rewrites(sql=sql, ref=ref, rules=rules, layout=layout, detail=detail)
        return {
            "relation": serialize.relation_ref(ref),
            "recommendations": [serialize.recommendation(item) for item in found],
            "workload_queries": 1,
        }

    return ToolDef(
        name="suggest_rewrite",
        title="Suggest rewrite",
        description=(
            "Deterministic rewrites of one query, each either exactly right or "
            "absent: a name qualified in full, a function-wrapped date predicate "
            "turned into a range the engine can prune with, a reserved word quoted "
            "the way this engine quotes. SELECT * on a wide table is costed and "
            "explained but never rewritten — only the author knows which columns "
            "the answer needs."
        ),
        input_schema=object_schema(
            {
                "sql": {"type": "string", "description": "The query to rewrite."},
                "relation": {
                    "type": "string",
                    "description": "The relation it reads, fully qualified.",
                },
            },
            required=["sql", "relation"],
        ),
        output_schema=_advice_output_schema(),
        handler=handler,
    )


def _wants_validation(args: Mapping[str, JsonValue]) -> bool:
    value = args.get("validate")
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ToolError("argument 'validate' must be a boolean")
    return value


async def _validate(
    context: ServerContext,
    args: Mapping[str, JsonValue],
    inputs: AdviceInputs,
    found: Sequence[Recommendation],
) -> list[Recommendation]:
    """Measure the top candidate on a shadow table, and only the top candidate.

    One, not all of them: each validation copies a sample of the table, and on a
    warehouse that is real money. The rest keep their estimates and say so, which
    is the honest state — nobody measured them.
    """
    if not found:
        return list(found)
    if context.shadow is None:
        raise ToolError(
            "validation needs a writable connection this server was not given",
            suggestion=(
                "configure a second, write-privileged principal and a scratch schema; "
                "the read-only role the tools use cannot create a shadow table"
            ),
        )
    probe = optional_str(args, "sql")
    if probe is None:
        raise ToolError(
            "validation needs the query to measure",
            suggestion="pass the query you want pruning measured for as 'sql'",
        )

    try:
        validator = ShadowValidator(
            runner=context.shadow,
            config=context.config,
            scratch_schema=context.scratch_schema,
        )
    except ShadowError as exc:
        raise ToolError(str(exc), suggestion="set AGENTDB_ALLOW_SHADOW=true to opt in") from exc

    best = found[0]
    measurement = await validator.measure(
        ref=inputs.ref,
        layout=inputs.layout,
        probe_sql=probe,
        baseline=best.evidence.pruning_ratio,
        order_by=_key_columns(best, Kind.ORDER_BY),
        index_ddl=best.ddl if best.kind is Kind.SKIP_INDEX else None,
        cluster_by=_key_columns(best, Kind.CLUSTER_BY),
        stats_columns=_key_columns(best, Kind.STATS_COLUMNS),
    )
    return [measured(best, measurement), *found[1:]]


def _key_columns(recommendation: Recommendation, kind: Kind) -> tuple[str, ...]:
    """The columns a candidate proposes, for the kinds whose DDL names a key."""
    if recommendation.kind is not kind:
        return ()
    return tuple(column for column, _ in recommendation.evidence.distinct_counts)


def _require_engine(context: ServerContext, name: str) -> None:
    """Refuse advice this engine has no concept of, by name."""
    engine = context.adapter.engine
    if name in CLICKHOUSE_KINDS and engine != "clickhouse":
        raise ToolError(
            f"{name} is ClickHouse-specific and this connection is {engine}",
            suggestion="on Databricks use advise_clustering or advise_skipping_stats",
        )
    if name in DATABRICKS_KINDS and engine != "databricks":
        raise ToolError(
            f"{name} is Databricks-specific and this connection is {engine}",
            suggestion="on ClickHouse use advise_sort_key, advise_indexes or advise_projection",
        )


def _advice_input_schema() -> dict[str, JsonValue]:
    return object_schema(
        {
            "relation": {
                "type": "string",
                "description": "The relation to advise on, fully qualified.",
            },
            "sql": {
                "type": "string",
                "description": (
                    "The query prompting the question. Counted alongside the mined "
                    "workload, and the only demand signal where the connection "
                    "cannot read the engine's log."
                ),
            },
            "validate": {
                "type": "boolean",
                "description": (
                    "Measure the top candidate on a shadow table rather than "
                    "estimating it: a sampled copy of the relation carrying the "
                    "proposed design, planned and dropped. Off by default, needs a "
                    "writable connection and AGENTDB_ALLOW_SHADOW, and costs a real "
                    "table copy — on a warehouse, real money."
                ),
            },
            "hours": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_WORKLOAD_HOURS,
                "description": (
                    f"Workload window ending now. Defaults to {DEFAULT_WORKLOAD_HOURS} "
                    "hours — long enough for a weekly report, short enough to describe "
                    "how the table is used now."
                ),
            },
        },
        required=["relation"],
    )


def _advice_output_schema() -> dict[str, JsonValue]:
    return object_schema(
        {
            "relation": {"$ref": "#/$defs/relation_ref"},
            "recommendations": array_of(
                "recommendation", "Best first: confidence, then expected effect."
            ),
            "workload_queries": {
                "type": "integer",
                "description": (
                    "Queries the demand signal covers. A recommendation derived from "
                    "one query is a materially weaker claim than one derived from a "
                    "week of traffic, and this is how a reader tells them apart."
                ),
            },
        },
        required=["relation", "recommendations", "workload_queries"],
    )


_TITLES: Final[Mapping[str, str]] = {
    "advise_sort_key": "Advise sort key",
    "advise_indexes": "Advise skip indexes",
    "advise_projection": "Advise projection",
    "advise_clustering": "Advise clustering key",
    "advise_skipping_stats": "Advise data-skipping statistics",
    "advise_compaction": "Advise compaction",
}

_DESCRIPTIONS: Final[Mapping[str, str]] = {
    "advise_sort_key": (
        "ClickHouse only. A sort-key proposal with the cardinality reasoning behind "
        "it and a migration, stated as the rebuild it is — ORDER BY cannot be "
        "altered in place. Dropping a leading column the workload depends on is "
        "flagged as a regression rather than buried."
    ),
    "advise_indexes": (
        "ClickHouse only. Skip-index candidates for filters the sort key cannot "
        "serve, with the index type chosen by predicate shape: bloom filters for "
        "high-cardinality equality, set indexes for low, minmax for ranges, token "
        "filters for text search."
    ),
    "advise_projection": (
        "ClickHouse only. A projection for a recurring GROUP BY the base sort key "
        "cannot serve, with the storage cost stated: a projection is a second "
        "physical copy and every insert writes twice."
    ),
    "advise_clustering": (
        "Databricks only. A liquid clustering key ranked by filter frequency then "
        "selectivity — deliberately not the ClickHouse cardinality rule, which "
        "exists for a sparse primary index and does not transfer. Says so when the "
        "table is managed and predictive optimization may already be choosing keys."
    ),
    "advise_skipping_stats": (
        "Databricks only, and the one with no counterpart in any other server. "
        "Reports which filter columns fall outside delta.dataSkippingStatsColumns "
        "or past dataSkippingNumIndexedCols and therefore cannot prune a single "
        "file, with the DDL to widen the set and the warning that it is not "
        "retroactive."
    ),
    "advise_compaction": (
        "Databricks only. An OPTIMIZE and target-file-size proposal derived from "
        "numFiles and average size. Projects a file count and claims nothing about "
        "latency, which this project does not measure on a shared warehouse."
    ),
}
