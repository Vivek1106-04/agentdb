"""Discovery tools: what exists, and how it is stored (SPEC §13.1).

``physical_layout`` is the differentiating one. Every SQL MCP server on the
registry can list tables and dump DDL; none of them tells the agent that the
sort key is ``(CounterID, EventDate)`` and that its ``WHERE UserID = …`` filter
will therefore read every granule in the table.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentdb.server import serialize
from agentdb.server.base import ServerContext, ToolDef, optional_str, require_str
from agentdb.server.schemas import (
    JsonValue,
    array_of,
    definition_schema,
    object_schema,
)

_NO_ARGS: dict[str, JsonValue] = object_schema({}, required=[])

_RELATION_ARG: dict[str, JsonValue] = object_schema(
    {
        "relation": {
            "type": "string",
            "description": (
                "Fully-qualified relation name: database.table on ClickHouse, "
                "catalog.schema.table on Databricks."
            ),
        }
    },
    required=["relation"],
)


def discovery_tools(context: ServerContext) -> tuple[ToolDef, ...]:
    """Build the discovery group against ``context``."""
    return (
        _list_namespaces(context),
        _list_relations(context),
        _describe_relation(context),
        _physical_layout(context),
    )


def _list_namespaces(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:  # noqa: ARG001
        relations = await context.adapter.list_relations(None)
        seen = {_namespace_of(item.ref.catalog, item.ref.namespace) for item in relations}
        namespaces = serialize.strings(sorted(seen))
        return {"engine": context.adapter.engine, "namespaces": namespaces}

    return ToolDef(
        name="list_namespaces",
        title="List namespaces",
        description=(
            "Namespaces visible to this connection, as fully-qualified strings — "
            "one level on ClickHouse, catalog.schema on Databricks — so the agent "
            "never has to guess the depth of the namespace."
        ),
        input_schema=_NO_ARGS,
        output_schema=object_schema(
            {
                "engine": {"enum": ["clickhouse", "databricks"]},
                "namespaces": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sorted, and each one usable as-is in list_relations.",
                },
            },
            required=["engine", "namespaces"],
        ),
        handler=handler,
    )


def _list_relations(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        namespace = optional_str(args, "namespace")
        relations = await context.adapter.list_relations(namespace)
        return {"relations": serialize.json_rows(relations)}

    return ToolDef(
        name="list_relations",
        title="List relations",
        description=(
            "Tables and views in a namespace with approximate row counts and "
            "on-disk bytes. Sizes are the catalog's own estimates, and are null "
            "where the engine does not hold them rather than zero."
        ),
        input_schema=object_schema(
            {
                "namespace": {
                    "type": "string",
                    "description": "Omit to list the connection's default namespace.",
                }
            },
            required=[],
        ),
        output_schema=object_schema(
            {"relations": array_of("relation", "In the order the engine lists them.")},
            required=["relations"],
        ),
        handler=handler,
    )


def _describe_relation(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        relation = context.parse_relation(require_str(args, "relation"))
        detail = await context.adapter.describe_relation(relation)
        return serialize.relation_detail(detail)

    return ToolDef(
        name="describe_relation",
        title="Describe relation",
        description=(
            "Columns with types, comments and 1-based ordinal positions, plus the "
            "engine's own CREATE statement. Ordinal position is not cosmetic on "
            "Databricks: Delta collects data-skipping statistics only for the "
            "leading columns, so a column's ordinal decides whether a filter on "
            "it can prune files at all."
        ),
        input_schema=_RELATION_ARG,
        output_schema=definition_schema("relation_detail"),
        handler=handler,
    )


def _physical_layout(context: ServerContext) -> ToolDef:
    async def handler(args: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        relation = context.parse_relation(require_str(args, "relation"))
        layout = await context.adapter.physical_layout(relation)
        return serialize.physical_layout(layout)

    return ToolDef(
        name="physical_layout",
        title="Physical layout",
        description=(
            "How the relation is actually stored. ClickHouse: table engine, sort "
            "key, partition key, skip indexes, projections, TTL, compression "
            "ratio. Databricks: format, clustering and partition columns, file "
            "count and average size, deletion vectors, and the effective "
            "data-skipping statistics column set. None of this appears in a DDL "
            "dump, and all of it decides how much data a filter reads."
        ),
        input_schema=_RELATION_ARG,
        output_schema=definition_schema("physical_layout"),
        handler=handler,
    )


def _namespace_of(catalog: str | None, namespace: str) -> str:
    return namespace if catalog is None else f"{catalog}.{namespace}"
