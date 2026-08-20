"""Grounding tools: the facts that go into the prompt before the agent writes SQL.

``grounded_context`` is the tool an agent calls first, and it is also the exact
payload the Family A arms of the benchmark measure (SPEC §11.3). Serving the
same bundle here that the harness scores there is deliberate: the number in the
report is a number about this tool, not about a demo written to flatter it.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentdb.adapters import SamplePolicy
from agentdb.core import GroundingLevel
from agentdb.server import serialize
from agentdb.server.base import (
    ServerContext,
    ToolDef,
    ToolError,
    optional_str,
    require_str,
    require_str_list,
)
from agentdb.server.schemas import (
    JsonValue,
    array_of,
    definition_schema,
    object_schema,
)


def grounding_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Build the grounding group against ``context``."""
    return (
        _profile_columns(context),
        _grounded_context(context),
        _dialect_rules(context),
    )


def _profile_columns(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        relation = context.parse_relation(require_str(args, "relation"))
        columns = require_str_list(args, "columns")
        policy = SamplePolicy(
            fraction=context.config.default_sample_fraction,
            max_rows=context.config.profile_max_rows,
            timeout_s=context.config.query_timeout_s,
        )
        profiles = await context.adapter.column_profile(relation, columns, policy)
        return {
            "profiles": [serialize.column_profile(profile) for profile in profiles],
            "sample_fraction": policy.fraction,
            "max_sampled_rows": policy.max_rows,
        }

    return ToolDef(
        name="profile_columns",
        title="Profile columns",
        description=(
            "Sampled cardinality, null ratio, min/max and top-k values for named "
            "columns. Every figure carries sample_method and sampled_rows, so an "
            "estimate is never mistaken for an exact count. Profiling reads a "
            "bounded sample and never full-scans."
        ),
        input_schema=object_schema(
            {
                "relation": {"type": "string", "description": "Fully-qualified relation name."},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Columns to profile. Each one costs a probe.",
                },
            },
            required=["relation", "columns"],
        ),
        output_schema=object_schema(
            {
                "profiles": array_of("column_profile", "One per column that exists."),
                "sample_fraction": {
                    "type": "number",
                    "description": "Fraction requested, where the engine supports sampling.",
                },
                "max_sampled_rows": {
                    "type": "integer",
                    "description": "Ceiling applied where it does not.",
                },
            },
            required=["profiles", "sample_fraction", "max_sampled_rows"],
        ),
        handler=handler,
    )


def _grounded_context(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        namespace = require_str(args, "namespace")
        level = _level(optional_str(args, "level"))
        relations = args.get("relations")
        names = require_str_list(args, "relations") if relations is not None else None
        bundle = await context.builder.build(namespace, level, names)
        return serialize.grounded_context(bundle)

    return ToolDef(
        name="grounded_context",
        title="Grounded context",
        description=(
            "The assembled context bundle for a namespace: relations, physical "
            "layout, column profiles, rendered as one block ready to put in a "
            "prompt and also returned structured. The tool an agent calls first. "
            "Asking for a level the engine cannot honestly serve fails rather "
            "than quietly returning a thinner payload under the level's name."
        ),
        input_schema=object_schema(
            {
                "namespace": {"type": "string", "description": "Fully-qualified namespace."},
                "question": {
                    "type": "string",
                    "description": (
                        "The question being answered. Accepted and currently "
                        "unused: the bundle is per-namespace, and question-aware "
                        "selection is a separate measured arm."
                    ),
                },
                "level": {
                    "enum": [level.value for level in GroundingLevel],
                    "description": (
                        "schema: DDL only. stats: plus column profiles. layout: "
                        "plus physical design. Cumulative; defaults to layout."
                    ),
                },
                "relations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Bare relation names to narrow the bundle to.",
                },
            },
            required=["namespace"],
        ),
        output_schema=object_schema(
            {
                "engine": {"enum": ["clickhouse", "databricks"]},
                "namespace": {"type": "string"},
                "level": {"enum": [level.value for level in GroundingLevel]},
                "relations": array_of("relation_context", ""),
                "rendered": {"type": "string", "description": "The payload as prompt-ready text."},
                "size_bytes": {
                    "type": "integer",
                    "description": "UTF-8 size of rendered. Grounding is not free.",
                },
            },
            required=["engine", "namespace", "level", "relations", "rendered", "size_bytes"],
        ),
        handler=handler,
    )


def _dialect_rules(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:  # noqa: ARG001
        return serialize.dialect_rules(await context.adapter.dialect_rules())

    return ToolDef(
        name="dialect_rules",
        title="Dialect rules",
        description=(
            "Identifier quoting, reserved words, string quoting and known quirks "
            "for the connected engine version. Cheaper to tell an agent the rule "
            "than to let it discover the rule by failing."
        ),
        input_schema=object_schema({}, required=[]),
        output_schema=definition_schema("dialect_rules"),
        handler=handler,
    )


def _level(value: str | None) -> GroundingLevel:
    """Resolve the requested grounding level, naming the valid ones on a miss."""
    if value is None:
        return GroundingLevel.LAYOUT
    try:
        return GroundingLevel(value)
    except ValueError as exc:
        valid = ", ".join(level.value for level in GroundingLevel)
        raise ToolError(
            f"unknown grounding level {value!r}", suggestion=f"use one of: {valid}"
        ) from exc
