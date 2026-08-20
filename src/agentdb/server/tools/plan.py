"""Plan tools: what the engine would do, before it does it (SPEC §13.1).

Neither engine has an executing EXPLAIN, which is exactly what makes these two
tools safe to hand an agent mid-loop: they cost a plan, not a scan of a hundred
million rows. ``actual_rows`` is therefore always null here and is filled in
only by ``run_query`` afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agentdb.core import (
    DatabricksPlanParseError,
    PlanParseError,
    PlanSummary,
    Severity,
)
from agentdb.server import serialize
from agentdb.server.base import ServerContext, ToolDef, ToolError, require_str, require_str_list
from agentdb.server.schemas import (
    JsonValue,
    definition_schema,
    object_schema,
    ref,
    schema,
)

MIN_CANDIDATES = 2
"""``explain_diff`` compares drafts; one draft is ``explain_plan``."""


def plan_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Build the plan group against ``context``."""
    return (_explain_plan(context), _explain_diff(context))


def _explain_plan(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        sql = require_str(args, "sql")
        namespace = require_str(args, "namespace")
        return serialize.plan_summary(await _explain(context, sql, namespace))

    return ToolDef(
        name="explain_plan",
        title="Explain plan",
        description=(
            "Run the engine's own EXPLAIN, normalize it, and say what is wrong in "
            "terms the agent can act on: how much of the data the filters prune, "
            "which relations are full-scanned, and which sort-key, clustering-key "
            "or data-skipping-statistics fact makes that so. The query is never "
            "executed — both engines' EXPLAIN is estimate-only, so actual_rows is "
            "null until run_query has run. On Databricks the plan states no file "
            "counts at all, so a pruning ratio there is only ever measured, "
            "afterwards, and pruning_source says which."
        ),
        input_schema=object_schema(
            {
                "sql": {
                    "type": "string",
                    "description": (
                        "The draft query. It will not be run. Qualify every "
                        "relation in it — the connection's default namespace is "
                        "not changed by the namespace argument below."
                    ),
                },
                "namespace": {
                    "type": "string",
                    "description": (
                        "Namespace whose layout and profiles are gathered as "
                        "evidence for the warnings. It does not rewrite the query."
                    ),
                },
            },
            required=["sql", "namespace"],
        ),
        output_schema=definition_schema("plan_summary"),
        handler=handler,
    )


def _explain_diff(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        namespace = require_str(args, "namespace")
        candidates = require_str_list(args, "candidates")
        if len(candidates) < MIN_CANDIDATES:
            raise ToolError(
                f"explain_diff compares at least {MIN_CANDIDATES} candidates,"
                f" got {len(candidates)}",
                suggestion="use explain_plan for a single query",
            )
        return _diff([await _explain(context, sql, namespace) for sql in candidates])

    return ToolDef(
        name="explain_diff",
        title="Compare candidate queries",
        description=(
            "Plan two or more of your own drafts and compare them on pruning "
            "ratio, estimated bytes and warnings, so the choice between them is "
            "made on the engine's evidence rather than on which one reads better. "
            "Nothing is executed. When the plans are indistinguishable on the "
            "evidence the engine reported, that is what it says."
        ),
        input_schema=object_schema(
            {
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": MIN_CANDIDATES,
                    "description": (
                        "Candidate queries, in the order you want them indexed. "
                        "Qualify every relation in them."
                    ),
                },
                "namespace": {
                    "type": "string",
                    "description": (
                        "Namespace whose layout and profiles are gathered as "
                        "evidence. It does not rewrite the queries."
                    ),
                },
            },
            required=["candidates", "namespace"],
        ),
        output_schema=schema(
            {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "sql": {"type": "string"},
                                "summary": ref("plan_summary"),
                            },
                            "required": ["index", "sql", "summary"],
                            "additionalProperties": False,
                        },
                    },
                    "recommended_index": {
                        "type": ["integer", "null"],
                        "description": (
                            "The candidate with the strongest plan evidence, or "
                            "null when the plans cannot be told apart."
                        ),
                    },
                    "reason": {"type": "string"},
                },
                "required": ["candidates", "recommended_index", "reason"],
                "additionalProperties": False,
            }
        ),
        handler=handler,
    )


async def _explain(context: ServerContext, sql: str, namespace: str) -> PlanSummary:
    """Plan one query, turning an unreadable plan into a result rather than a crash.

    Both analyzers refuse to half-understand a plan, which is the right call —
    warnings derived from a guess are worse than none. At the tool boundary that
    refusal becomes an error the agent can act on: the query is still runnable,
    it just cannot be reviewed first.
    """
    try:
        return await context.explainer.explain(sql, namespace)
    except (PlanParseError, DatabricksPlanParseError) as exc:
        raise ToolError(
            f"the engine's plan output could not be parsed: {exc}",
            suggestion="the query is still runnable; run_query works without a plan review",
        ) from exc


@dataclass(frozen=True, slots=True, order=True)
class _Rank:
    """How good a plan looks, worst-first fields ordered by how much they matter.

    Warnings outrank the pruning ratio on purpose: a ``STATS_NOT_COLLECTED`` or
    ``SORT_KEY_UNUSED`` finding is a statement about the mechanism, while a
    missing ratio is often just an engine that reported nothing.
    """

    critical: int
    warnings: int
    pruning_ratio: float
    estimated_bytes: float


def _rank(summary: PlanSummary) -> _Rank:
    return _Rank(
        critical=sum(1 for w in summary.warnings if w.severity is Severity.CRITICAL),
        warnings=len(summary.warnings),
        pruning_ratio=1.0 if summary.pruning_ratio is None else summary.pruning_ratio,
        estimated_bytes=(
            math.inf if summary.estimated_bytes_read is None else summary.estimated_bytes_read
        ),
    )


def _diff(summaries: Sequence[PlanSummary]) -> dict[str, JsonValue]:
    """Rank the candidates, and decline to pick when nothing separates them."""
    ranks = [_rank(summary) for summary in summaries]
    best = min(range(len(ranks)), key=lambda index: ranks[index])
    tied = [index for index, rank in enumerate(ranks) if rank == ranks[best]]
    candidates: JsonValue = [
        {"index": index, "sql": summary.sql, "summary": serialize.plan_summary(summary)}
        for index, summary in enumerate(summaries)
    ]
    if len(tied) > 1:
        return {
            "candidates": candidates,
            "recommended_index": None,
            "reason": (
                "the plans are indistinguishable on the evidence the engine "
                f"reported: candidates {tied} rank identically"
            ),
        }
    return {
        "candidates": candidates,
        "recommended_index": best,
        "reason": _reason(ranks[best]),
    }


def _reason(rank: _Rank) -> str:
    parts = [f"fewest plan warnings ({rank.warnings}, {rank.critical} critical)"]
    if rank.pruning_ratio < 1.0:
        parts.append(f"lowest share of data read after pruning ({rank.pruning_ratio:.1%})")
    if math.isfinite(rank.estimated_bytes):
        parts.append(f"lowest estimated bytes read ({int(rank.estimated_bytes):,})")
    return "; ".join(parts)
